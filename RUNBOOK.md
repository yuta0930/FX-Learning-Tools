# RUNBOOK

このドキュメントは運用時のクイックリファレンスです。

## Patterns: Triangle / Rectangle / Asia Box（最短）

グリッド評価（London/NY前提）

- VS Code: Patterns: grid (LDN/NY)（Flag/Pennant 用）
- VS Code:
	- Patterns: grid (Triangle LDN/NY)
	- Patterns: grid (Rectangle LDN/NY)
	- Patterns: grid (AsiaBox LDN/NY)
→ reports/patterns_grid_YYYYMMDD/summary.csv

## Flag/Pennant 最適化（15m, London/NY セッション）

推奨フロー（精度寄り）:

1) 事前設定の確認
- `config/patterns.yml` の flag_pennant 既定（pole_min_atr=3.0, flag_slope_max_atr=0.10, contraction_percentile=0.15 など）
- `config/patterns_session.yml` のセッション別上書き（Tokyo は厳しめ / LDN/NY はやや緩め）

2) グリッド評価（LDN/NY）
- VS Code: Patterns: grid (LDN/NY) を実行
- 出力: `reports/patterns_grid_YYYYMMDD/summary.csv` と各 `metrics.json`
- 指標: EV_net / hit_rate / n_signals と HTF フィルタ適用後の `ev_R_htf`, `uplift_pct`

3) しきい値の適用
- VS Code: Patterns: emit thresholds (from summary) を実行し、最上位候補の品質しきい値を `config/patterns_quality_thresholds.yml` に反映

4) 追加の軽量最適化（任意）
- `python scripts/optimize_flag_pennant_15m.py --data data/USDJPY_15m.csv --out reports/opt_flag_pennant_15m`
- London/NY 限定で detector パラメータ（pole_min_atr, flag_slope_max_atr, contraction_percentile 等）をグリッド探索
- 出力: `grid_results.csv` と `best.json`

採用ガイドライン（初期）:
- EV_net > 0 かつ サンプル数 n ≥ 100（希少パターンは ≥ 80 でも可）
- hit_rate の Wilson 下限が方向性ベースライン超え
- ニュースウィンドウ回避・HTF トレンド整合を満たすこと

しきい値の自動反映（人手転記ゼロ）
- Patterns: emit thresholds (from summary) を実行し、summary.csv の先頭行を採用
→ config/patterns_quality_thresholds.yml（.bak 自動バックアップ）

プリフライト → Paper → Live
- health_check.py で直近1–7日を確認（block_rate / unknown_ratio、任意で drift PSI/JS）
- Paper で数日 → 問題なければ Live（異常時は YAML を .bak で即ロールバック）

採用ガード（初期基準）

- evcurve_EV_net > 0 かつ evcurve_n ≥ 100（レアなパターンは ≥80 でも可）
- 可能なら hit_rate_ci.lo がベースラインを明確に上回る
- Tokyo は厳しめ（ニュース/スプレッド/収縮を強化）

### Rectangle Detector 最適化 (2025-11)

レクタングル検出が他パターンより著しく遅かったため最適化を実施しました。

主な変更:
1. ピボット不足ウィンドウの `argsort` 補完を原則スキップし、必要時のみ `np.argpartition` による O(L) 補完へ（`enable_low_pivot_fallback`）。
2. 幅配列を生成せず、幅の平均/標準偏差を閉形式計算（O(1)）。
3. ピボット数・価格平均を累積和で O(1) 取得しウィンドウ探索コストを削減。
4. 走査ステップ縮小: `last_N=2000`, `e_step=3`, `len_step=2` をデフォルト化（網羅性と速度のトレードオフ）。

ベンチマーク例 (USDJPY 15m, last_N=2000):
```
python scripts/benchmark_rectangle.py --baseline
Optimized Rectangle: time≈0.9s
Baseline Rectangle:  time≈6.7s
Speedup ≈ 7x
```

並列化（オプション）:
- `detect_rectangles(..., use_parallel=True, n_jobs=-1, parallel_backend="threads")` を実装済み。小さなワークロードではオーバーヘッドで遅くなるためガードを入れています。
- 本アルゴリズムは Python 側のループが残るため、スレッド並列は GIL の影響で有効でないケースが多いです。プロセス並列（`parallel_backend="loky"`）は配列のコピー/シリアライズが重く、メモリマップ（joblibのmmap）併用が必要です。現状は既定OFF（シリアル）推奨。

ベンチの使い方（並列比較）:
```cmd
python scripts/benchmark_rectangle.py --data data/USDJPY_15m.csv --last-N 4000 --win-max 120 --e-step 1 --len-step 1 --repeat 1
python scripts/benchmark_rectangle.py --data data/USDJPY_15m.csv --last-N 4000 --win-max 120 --e-step 1 --len-step 1 --repeat 1 --parallel --jobs 8
```
注: 上記の通り、環境やデータによりスレッド並列は遅くなる場合があります（特に検出0件のセグメント）。その場合はシリアル運用が最速です。

Numba（実験）:
- `scripts/benchmark_rectangle.py --use-numba` でベンチのみ numba JIT を試せます。
- 導入: `requirements.txt` に numba を追加済み。Windows では llvmlite のホイールが必要で、インストール済タスクで自動取得されます。
- アプリ本体への常時適用は未実装（安定性優先）。ベンチで効果が十分であれば今後統合可能です。

検出カバレッジを優先したい場合:
```
python scripts/benchmark_rectangle.py --last-N 3000 --repeat 1 --win-max 48 --baseline
```
またはアプリ側で `detect_rectangles(..., e_step=1, len_step=1, last_N=3000)` に調整。

回帰チェック（品質影響）:
- パターン無しケースで false positive 増は確認されず（既存ルール厳格化でむしろ減る可能性）。
- パターン件数が極端に減る場合は `enable_low_pivot_fallback=True` を維持したまま `e_step=1` に下げてください。

既知のトレードオフ:
- 高ボラ短期間に非常に短いボックス（< win_min_bars）が多発する場合、ステップ間引きで取り逃がす可能性。
- 旧補完ロジックによる「弱い矩形」検出は抑制される（品質スコア平均は上昇傾向）。

必要なら将来的に `quality_score` 分布の簡易統計をベンチに追加し、品質ドリフト監視を実装予定。
今後の強化候補:
- joblib + memmap によるプロセス並列の安定化（大規模データでのスケール）。
- 検索空間の追加削減（セッション/ボラティリティでの L の動的制限）。
- numba による `_eval_e` 内部のループ高速化。

## 6. 参考: ヘルスチェック CLI（例）

- 1日分、理由正規化あり（マッピングがある場合）

```cmd
python scripts/health_check.py --logs logs/executions.parquet --days 1 --max-block-rate 0.8 --max-unknown-ratio 0.3 --reasons-map config/reasons_map.yml
```

- ドリフト（PSI/JS）も判定する例（価格CSVの close 列を使用）

```cmd
python scripts/health_check.py --logs logs/executions.parquet --days 1 --max-block-rate 0.8 --max-unknown-ratio 0.3 --reasons-map config/reasons_map.yml --drift-source data/USDJPY_15m.csv --drift-column close --drift-ref-n 2000 --drift-cur-n 500 --drift-bins 20 --max-drift-psi 0.5 --max-drift-js 0.3
```

### VS Code からワンクリック
- Health (with drift example) タスク：代表的なドリフト設定でヘルスチェックを実行（Problems パネル連携あり）

## 7. Observability ドリフト確認

- Observability ページの「Drift quick check (PSI/JS)」カードから、任意のCSV/Parquetの列でPSI/JSを即時計算できます。
- 既定のしきい値は .env の `HEALTH_MAX_DRIFT_PSI` / `HEALTH_MAX_DRIFT_JS` で調整可能です。
 - ヘルスゲートは同カード内の「Run health gate with current settings」から実行できます（結果は `reports/health_YYYYMMDDHHMM/health.json` に保存）。

## ATR・kSL・kTP 常時表示パネル

- すべてのページのタイトル直下に、現状の ATR(15m)・kSL・kTP・RR(kTP/kSL)・SL/TP距離（pips/価格）・timeout bars を表示します。
- 入力データや設定が無い場合でもページは落ちません（No-Op）。説明キャプションのみ表示されます。
- セッション別の係数は `config/atr_targets.yml` を編集して反映されます（未配置でもデフォルトを使用）。
- spread_pips 列がある場合、SL が 2×spread 以上か（または min_spread_pips 以上か）を判定し、OK/NG を表示します。

使い方メモ（3行）
- どのページでも、タイトル直下に ATR・kSL・kTP・RR・SL/TP距離・timeout が出ます。
- セッション別の係数は `config/atr_targets.yml` を編集して反映（未配置でもデフォルトで表示）。
- スプレッドが広すぎる/SLが短すぎる場合は警告タグが赤表示になります（見送り判断の材料に）。

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

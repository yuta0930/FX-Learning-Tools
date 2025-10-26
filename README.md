# FX-Learning-Tools

確率的なレベルブレイク検出と意思決定を支援するトレーディング・ツール群です。既存の学習・推論フローに後方互換を保ちながら、レジーム判定・確率校正・執行・リスク管理をプラガブルに拡張しました。

## 主要機能

- 予測/推論
	- 既存の `ai_train_break.py` と `inference_break.py` を踏襲
	- 校正済みモデルが存在する場合は自動で優先利用（互換API）
- レジーム（市場状態）
	- ATR・実現ボラ・高低差比・セッションで簡易クラスタ
	- `src/features/regime.py`（薄いトランスフォーマー）
- 確率校正
	- `scripts/calibrate_break.py` で既存モデルを `CalibratedClassifierCV` で校正
	- レポート（信頼度プロット、Brier/ECE）を `reports/` に出力
- 執行/リスク
	- `src/execution/executor.py`: スリッページ/クールダウン/許可レジーム
	- `src/risk/risk_manager.py`: 日次損失上限・DDゲート・ケリー上限/サイズ
- バックテスト（WF）
	- `scripts/run_backtest.py`: ウォークフォワード評価、方向別（ロング/ショート）
	- 取引メトリクス（PF/Sharpe/MaxDD/PnL）と確率メトリクス（Brier/ECE）集計
	- 出力: `reports/wf_report.json`, `reports/equity_wf.png`
- アプリ（Streamlit）
	- `app.py` にモジュールのオン/オフとバックテストサマリ表示を追加
	- ログ（Parquet/JSONL）保存オプションあり

## ディレクトリ構成（抜粋）

```
.
├─ app.py
├─ ai_train_break.py / inference_break.py
├─ config/
│  ├─ config.yml  # 新設定（優先）
│  └─ loader.py   # 環境変数 FX_CFG で上書き可、training.yaml を後方互換で読む
├─ scripts/
│  ├─ calibrate_break.py
│  ├─ run_backtest.py
│  └─ train_regime.py
├─ src/
│  ├─ backtest/walkforward.py
│  ├─ execution/executor.py
│  ├─ features/regime.py
│  ├─ monitoring/{logs.py,metrics.py}
│  └─ risk/risk_manager.py
├─ reports/  # 出力先（wf_report.json / equity_wf.png 等）
└─ tests/    # 主要ユニットテスト
```

## クイックスタート（Windows）

1) 依存インストール（ワークスペースのタスクから）

- VS Code のタスク「Install requirements」を実行（仮想環境: `env/`）

2) データとモデルを配置

- データ: `data/USDJPY_15m.csv`（例）
- モデル: `models/break_model.joblib`（必須）、`break_model_calibrated.joblib`（任意）
- メタ: `models/break_meta.json`（threshold など）

3) バックテスト（WF）

- 実行後、`reports/` に以下が生成されます:
	- `wf_report.json`: 確率/取引メトリクス（base/cal、long/short 別を含む）
	- `equity_wf.png`: エクイティ曲線

4) アプリ起動

- Streamlit ターミナルで `app.py` を起動（既にセットアップ済みであれば自動）
- 設定パネル内に「バックテストサマリ (WF)」が表示されます

### pre-commit（推奨）

```
pip install pre-commit
pre-commit install
```

以降はコミット前に ruff/black/mypy（軽）と pytest fast が自動実行されます。

## 本番昇格までの最短プレイブック（パターン品質ゲート）

1) グリッド評価（London/NY前提）

- VS Code タスク「Patterns: grid (LDN/NY)」を実行
- 出力: `reports/patterns_grid_YYYYMMDD/summary.csv`

2) 推奨しきい値の YAML 生成（人手転記ゼロ）

- VS Code タスク「Patterns: emit thresholds (from summary)」を実行し、summary.csv のパスを入力
- 例: `reports/patterns_grid_YYYYMMDD/summary.csv`
- 出力: `config/patterns_quality_thresholds.yml`（`.bak` バックアップ自動作成）

3) 品質ゲートは「設定時のみ有効化」

- しきい値が入ると最終ゲートで AND 適用され、`trade_ok=False` と `reason=... | pattern_quality<thr` が付与されます
- 未設定/空/読込失敗時は完全 No-Op（既存運用を壊しません）

4) プリフライト & ソフトランディング

- `scripts/health_check.py` で直近 1–7 日の block_rate / unknown_ratio を確認（任意でドリフト閾値も）
- MODE=paper + KILL_SWITCH=1 で数日 Paper 試運用 → 04/05 ページで block 件数と理由を確認

5) 採用ガード（初期目安）

- `summary.csv`: `evcurve_EV_net > 0` かつ `evcurve_n ≥ 100`
- 可能なら Wilson 95% 下限 `hit_rate_ci.lo` がベースラインを明確に上回ること
- Tokyo は 0.08–0.10 目安で厳しめ、London/NY を中心に更新

6) 週次運用の型

- `patterns_grid.py` → `emit_quality_thresholds.py` → `health_check.py`
- しきい値更新は `git commit`、ロールバックは `.bak` を置換

一括実行（自動検出）:

```
python scripts/weekly_quality_gate.py --tokyo-thr 0.10 --sessions "London,NewYork"
```

最新の `reports/patterns_grid_YYYYMMDD/summary.csv` を検出し、しきい値出力・1日ヘルスチェックを順に実行します。
通知（任意）: Slack/Teams の Incoming Webhook を使う場合は、環境変数 `WEEKLY_WEBHOOK_URL` を設定するか `--webhook-url` を渡してください。

## 設定（config/config.yml）

- general: symbol, timeframe, tz, seed
- paths: models_dir, logs_dir, data_dir, reports_dir
- regime: method, atr_window, rv_window, session_dummies など
- calibration: enabled, method
- execution: slippage_alpha/beta, cool_down_bars, allowed_regimes
- risk: daily_loss_limit_pct, max_drawdown_pct, kelly_cap, base_risk_per_trade_pct
- backtest: wf_window_days, wf_step_days, metrics
- logging: parquet/jsonl/keep_raw_probs
- quality: ap_min_* / brier_max / coverage_warn_min
- cost_sensitivity: costs（EVのコスト感度）

## パフォーマンス調整（TTL・プロファイル）

- PRICE_FETCH_TTL（秒）
	- 価格データ取得のキャッシュTTL（Streamlit `st.cache_data`）を環境変数で制御できます。
	- 既定: 60（未設定時）
	- 例: `set PRICE_FETCH_TTL=120`

- DEV_PROFILING（開発用）
	- `1/true` を設定すると、開発用の軽量プロファイリング表示がサイドバーに出ます（現状は価格取得の処理時間のみ）。
	- 例: `set DEV_PROFILING=1`

いずれもアプリ起動前に設定してください（起動後の動的変更には対応しません）。

追加のTTLノブ:
- EVENTS_TTL: ニュース影ウィンドウ生成のキャッシュTTL（秒）。既定 300。
- MODEL_CACHE_TTL: モデル読み込みリソースキャッシュTTL（秒）。既定 600。

```
set EVENTS_TTL=300
set MODEL_CACHE_TTL=600
```

## E2Eスモークテスト（CI/ローカル）

`tests/test_e2e_smoke.py` はネットワークに依存せず、以下を最小限で通します。
- ローカルCSV（`data/USDJPY_15m.csv`）の一部で特徴量生成（build_features → augment_features）
- モデル/メタ読込 → 予測（predict_with_session_theta）
- 最終ゲート適用（policy.gate.apply_final_gate）

ローカル実行（任意）:
```
python -m pytest -q -k e2e_smoke
```

## 重大ゲート失敗の即時通知（任意ON）

ヘルスチェック `scripts/health_check.py` に即時通知オプションを追加しました。
- 引数 `--webhook-url`、または環境変数 `HEALTH_WEBHOOK_URL` に Slack/Teams の Incoming Webhook を指定すると、
	失敗時（閾値違反）と成功時のステータスを通知します。

例（Windows cmd）:
```
set HEALTH_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
python scripts/health_check.py --logs logs/executions.parquet --days 1 --max-block-rate 0.8 --max-unknown-ratio 0.3
```

週次オーケストレータ（`scripts/weekly_quality_gate.py`）は既に `--webhook-url`/`WEEKLY_WEBHOOK_URL` に対応済みです。

## モデルのバージョニング/署名チェック（簡易）

学習後にモデルのSHA-256署名を `models/break_meta.json` に記録し、推論時に整合性を確認します。

1) 署名の付与（学習/更新後に実行）
```
python scripts/model_sign.py --model models/break_model_calibrated.joblib
# 未指定なら校正版が存在する場合はそちらを優先
```

2) 推論時の検証
- 既定では不一致時に警告（動作は継続）
- 本番で厳格にする場合は `STRICT_MODEL_SIGNATURE=1` を設定（不一致なら起動失敗）

```
set STRICT_MODEL_SIGNATURE=1
```

3) ローリング更新の手順（簡易）
- 新しいモデル（と必要なら校正モデル）を `models/` に配置
- `scripts/model_sign.py` で署名を更新（`models/break_meta.json` に反映）
- `models/break_meta.json` の閾値やセッション別θも必要に応じて更新
- アプリ/ジョブを再起動（署名チェックが通ることを確認）

## レポートの見方

- prob（確率メトリクス）
	- Brier: 0 に近いほど良い
	- ECE: 較正の良さ（小さいほど良い）
- trade（取引メトリクス）
	- PF: Profit Factor（>1 でプラス優位）
	- Sharpe: 平均/標準偏差に基づく簡易 Sharpe
	- MaxDD: 最大ドローダウン
	- PnL: 総損益（割合集計）
	- base_by_side / cal_by_side: ロング/ショート別の同指標

## よくあるトラブル

- 設定が読めない: `FX_CFG` 環境変数で明示パス指定可。既定は `config/config.yml`、無ければ `config/training.yaml`。
- Parquet 書き込みエラー: `pyarrow` のインストールを確認（`requirements.txt` に含む）。
- モデル不在: `models/break_model.joblib` が無いとバックテストが動きません。

## 設定の検証（オプション）

Pydantic スキーマで `config/config.yml` を検証できます。既定は後方互換のため検証オフですが、CLI でいつでもチェック可能です。

```
python scripts/validate_config_cli.py            # 非strict（警告）
python scripts/validate_config_cli.py --strict   # 失敗時に非ゼロ終了
```

## Docker（任意）

最小の再現用 Dockerfile を用意しています。

```
docker build -t fxlt .
docker run --rm -it -p 8501:8501 --env-file .env fxlt
```

fast テストのみを先行実行するヘルスチェックを含みます。大規模データはホスト側に置き、`-v` で共有する運用を推奨します。

## ライセンス

本リポジトリのコードはプロジェクトの目的に準じて利用してください。
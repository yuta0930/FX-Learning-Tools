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

## ライセンス

本リポジトリのコードはプロジェクトの目的に準じて利用してください。
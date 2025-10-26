## 0. 要約（1分で把握）

- 本リポジトリは、為替（USDJPY 15m想定）のレベルブレイク確率推定と意思決定を行うStreamlitアプリとバッチ群です。学習済みモデル（joblib）とメタ情報（JSON）、設定（YAML）を用いて、校正・レジーム判定・リスクガード・ドリフト監視をプラガブルに適用します。
- フロントはStreamlit単一ページ（`app.py`）、バックは純Pythonモジュール（`src/`配下）で構成。永続DBは使わず、CSV/Parquet/JSONLとモデルファイルを入出力に用いる軽量構成です。
- 外部連携はOpenAI（説明生成等）とOANDA（任意の発注API、環境変数で有効化）。品質・監視はpytest、簡易メトリクス、PSI等で補助します。

## 1. 技術スタック

- 言語/ランタイム: Python 3.11（`pyproject.toml` mypy設定）
- フロント: Streamlit 1.41
- 機械学習/データ: scikit-learn 1.7, pandas 2.3, numpy 2.3, shap, matplotlib/plotly
- 品質/開発: pytest 8.3, ruff, black, mypy（型は緩め）
- ユーティリティ: PyYAML, yfinance, requests/httpx, python-dotenv
- 永続化: joblib（モデル）, CSV/Parquet/JSONL（ログ/レポート）。DB（peewee）は現状未使用の可能性が高い

## 2. システム構成とデータフロー（簡易）

```
[Browser]
  ↕ Streamlit (app.py)
  ├─ UI操作/描画/設定
  └─ Core呼び出し
       ↓
[src/* モジュール]
  - monitoring(PSI/JS/KL/health)
  - features/regime（レジーム付与）
  - execution（スリッページ/クールダウン）
  - risk（リスク管理）
       ↓
[models/*.joblib + break_meta.json]
       ↓
[data/*.csv 入力] → [reports/*.json|*.png 出力]
       ↓
[logs/*.parquet|*.jsonl（任意）]
外部: OpenAI API, OANDA API（任意）
```

## 3. ルーティング/API一覧

- HTTPベースのAPIは存在しません（サーバフレームワーク未使用）。
- UI: Streamlit単一アプリ `app.py`（約7.5k行、機能多数；例: News窓抑制、リスクゲート、OpenAI補助、OANDA発注）。
- バッチ/スクリプト: `scripts/calibrate_break.py`（校正）、`scripts/run_backtest.py`（WF）、`scripts/train_regime.py`。VS Codeタスクで実行可。

## 4. データモデル要約

- 学習モデル: `models/break_model.joblib`（scikit-learn互換）、方向別`break_up/down_model.joblib`等。
- メタ情報: `models/break_meta.json`（閾値`threshold`、`ev_per_trade`、`theta_by_session` など）。
- 入力データ: `data/USDJPY_15m.csv`（OHLCV想定）。
- 出力: `reports/wf_report.json`, `reports/equity_wf.png`, `reports/break_calibration.json`。
- ログ: `logs/trades.parquet|jsonl`（`src/monitoring/logs.py`）。RDBスキーマはなし。

## 5. セキュリティ

- 認証/認可: サーバAPIなし。Streamlitローカル利用前提。
- Secret管理: `.env`想定（例: `OPENAI_API_KEY`、`OANDA_TOKEN`/`OANDA_ACCOUNT` を `app.py`が参照）。キー値は保存/出力しない設計。
- 入力検証: pandas/numpyベースの型/列チェックが中心。CORS/CSP/Helmet/RateLimitは非適用（サーバ未使用）。
- 組織利用時はStreamlit共有/認証やプロキシ越しの鍵管理（環境変数スコープ制限）を推奨。

## 6. パフォーマンス

- クライアント側UIはStreamlit標準。チャート/テーブル等の描画が多く、データ量による再実行コストに注意。
- モデル推論は軽量（scikit-learn）で、確率校正やドリフト計算（PSI/JS/KL/Hellinger）を適宜キャッシュ化可能。
- 画像最適化/コード分割/SSR等のWeb最適化は対象外。I/OはCSV/Parquet中心で高速。
- 既知ボトルネック候補: `app.py` 内の大規模処理の再実行、ニュース/外部API待ち、巨大CSVの読み込み。

## 7. アクセシビリティ/国際化

- UIはStreamlit標準コンポーネント中心。ARIA属性/キーボード操作の詳細制御は限定的。
- 言語: 日本語中心（UI文言/コメント）。時系列はJSTを基本（`utils/time_utils`）。
- ダークモード等のテーマ切替はStreamlit依存。多言語i18nは未導入。

## 8. 観測可能性（ログ/メトリクス/トレース）

- ロギング: アプリ最上部で`logging`設定（`fx.app`）。取引ログはParquet/JSONLに記録可能（`src/monitoring/logs.py`）。
- メトリクス: Brier/ECE（`src/monitoring/metrics.py`）。ドリフト: PSI/JS/KL/Hellinger（`monitoring.py`）。
- トレース/分散トレーシングは未導入。異常検知は簡易ヘルスチェック（モデル/メタ/価格鮮度）。

## 9. テスト/品質/CI

- テスト: pytest。`tests/test_app_small_units.py` ほか、RiskGuard等の小粒テストあり。
- Lint/Format: ruff/black、mypy（寛容設定）。
- CI: 設定ファイルは未検出（手元タスクで実行想定）。カバレッジは不明だがユニット中心のスモーク・境界テストを用意。

## 10. デプロイ/運用

- ローカル実行: VS Codeタスクで依存導入/テスト実行、Streamlitターミナルから `app.py` を起動。
- 環境差分: `config/config.yml` を既定、`FX_CFG` 環境変数で外部ファイルを指定可能（後方互換で `training.yaml` も読込）。
- コンテナ/Docker/CICDは未提供。運用はWindowsローカル/スケジュールタスク等を想定（`scripts/run_daily_jobs.cmd`）。

## 11. 既知の課題/リスク

- 単一巨大`app.py`に機能が集約（再実行と状態管理の複雑化、保守性低下）。
- セキュリティ境界が曖昧（ローカル想定だが、外部API鍵の保護・権限設計が必要）。
- DB未使用により履歴/分析の拡張性は限定的（ロングターム集計や監査要件が伸びると限界）。

## 12. 不明点（外部レビュアーへの質問テンプレ）

- 本番運用の形態（単純ローカル/社内サーバ/クラウド共有）とアクセス権限は？
- OANDA発注は本番で使う想定か（鍵の保管、監査ログ、ドライラン/紙トレの境界）？
- モデル更新と校正の運用手順、ローリング/リリース基準は？

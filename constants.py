from __future__ import annotations
import os

# Timezone and base directories
TZ = "Asia/Tokyo"
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")

# Log file paths
SIGNALS_LOG = os.path.join(LOG_DIR, "signals.parquet")
ORDERS_LOG = os.path.join(LOG_DIR, "orders.parquet")
TRADES_LOG = os.path.join(LOG_DIR, "trades.parquet")
TCA_FEATS_LOG = os.path.join(LOG_DIR, "tca_features.parquet")

# Artifacts
CALIBRATION_DIR = os.path.join(ARTIFACTS_DIR, "calibration")
TCA_DIR = os.path.join(ARTIFACTS_DIR, "tca")

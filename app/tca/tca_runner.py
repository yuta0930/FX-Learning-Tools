from __future__ import annotations
from typing import Dict
import os
import pandas as pd

from constants import TCA_DIR, TCA_FEATS_LOG
from app.tca.slippage_model import SlippageModel, SlippageModelConfig


def fit_from_logs(cfg: Dict) -> str | None:
    tca_cfg = cfg.get("tca", {}) if cfg else {}
    if not tca_cfg.get("enabled", True):
        return None
    if not os.path.exists(TCA_FEATS_LOG):
        return None
    df = pd.read_parquet(TCA_FEATS_LOG)
    if len(df) < int(tca_cfg.get("min_samples", 500)):
        return None
    m = SlippageModel(SlippageModelConfig(model=tca_cfg.get("model", "quantile"),
                                          use_features=tca_cfg.get("use_features", [])))
    m.fit(df)
    return m.save()

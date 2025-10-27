from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import TCA_DIR

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


@dataclass
class SlippageModelConfig:
    model: str = "quantile"  # or "linear"
    use_features: List[str] = None  # type: ignore


class SlippageModel:
    """Predict slippage pips using quantile regression (p50/p90) or linear model."""

    def __init__(self, cfg: SlippageModelConfig):
        self.cfg = cfg
        self._p50 = None
        self._p90 = None
        self._lin = None

    def fit(self, df: pd.DataFrame) -> None:
        use = self.cfg.use_features or []
        X = df[use].copy()
        X = self._encode(X)
        y = df["slip_pips"].astype(float).to_numpy()
        if self.cfg.model == "quantile" and sm is not None:
            Xc = sm.add_constant(X)
            self._p50 = sm.QuantReg(y, Xc).fit(q=0.5)
            self._p90 = sm.QuantReg(y, Xc).fit(q=0.9)
        else:
            self._lin = LinearRegression().fit(X, y)

    def predict(self, row: Dict) -> Dict[str, float]:
        x = self._encode(pd.DataFrame([row]))
        out = {"p50": 0.0, "p90": 0.0}
        if self._p50 is not None and self._p90 is not None:
            Xc = sm.add_constant(x)
            out["p50"] = float(self._p50.predict(Xc)[0])
            out["p90"] = float(self._p90.predict(Xc)[0])
        elif self._lin is not None:
            yhat = float(self._lin.predict(x)[0])
            out["p50"] = yhat
            out["p90"] = yhat * 1.8
        return out

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        # minimal encoding for session/category
        X = X.copy()
        if "session" in X.columns:
            ses = X.pop("session").astype(str)
            for s in ["tokyo", "london", "ny"]:
                X[f"sess_{s}"] = (ses == s).astype(int)
        return X.fillna(0.0)

    def save(self) -> str:
        os.makedirs(TCA_DIR, exist_ok=True)
        path = os.path.join(TCA_DIR, "slippage_model.joblib")
        joblib.dump({"cfg": self.cfg, "p50": self._p50, "p90": self._p90, "lin": self._lin}, path)
        return path

    @staticmethod
    def load() -> Optional["SlippageModel"]:
        path = os.path.join(TCA_DIR, "slippage_model.joblib")
        if not os.path.exists(path):
            return None
        obj = joblib.load(path)
        m = SlippageModel(cfg=obj["cfg"])
        m._p50 = obj.get("p50")
        m._p90 = obj.get("p90")
        m._lin = obj.get("lin")
        return m

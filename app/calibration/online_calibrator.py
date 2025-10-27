from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable
import os
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from datetime import datetime

from app.utils.stats import bin_ece
from constants import CALIBRATION_DIR


@dataclass
class CalibReport:
    ece: float
    ks: Optional[float]
    bins: Dict


class OnlineCalibrator:
    """Lightweight online calibrator supporting Isotonic or Platt scaling.

    Expected input for fit_and_eval: DataFrame with columns ['p_raw','y'].
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._iso: Optional[IsotonicRegression] = None
        self._platt: Optional[LogisticRegression] = None

    def fit_and_eval(self, df_signals_recent) -> Tuple[object, CalibReport]:
        y = df_signals_recent["y"].to_numpy(dtype=float)
        p = df_signals_recent["p_raw"].to_numpy(dtype=float)
        if self.method == "isotonic":
            m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            m.fit(p, y)
            p_cal = m.predict(p)
            self._iso = m
        else:
            x = p.reshape(-1, 1)
            m = LogisticRegression(solver="lbfgs")
            m.fit(x, y)
            p_cal = m.predict_proba(x)[:, 1]
            self._platt = m
        ece, detail = bin_ece(y, p_cal, n_bins=10)
        rep = CalibReport(ece=float(ece), ks=None, bins=detail)
        return self, rep

    def predict(self, p_raw: np.ndarray | float) -> np.ndarray | float:
        if isinstance(p_raw, (float, int)):
            arr = np.array([float(p_raw)], dtype=float)
            out = self._predict_array(arr)
            return float(out[0])
        return self._predict_array(np.asarray(p_raw, dtype=float))

    def _predict_array(self, p: np.ndarray) -> np.ndarray:
        if self._iso is not None:
            return self._iso.predict(p)
        if self._platt is not None:
            return self._platt.predict_proba(p.reshape(-1, 1))[:, 1]
        return p

    # --- artifacts ---
    def save_artifact(self) -> str:
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        path = os.path.join(CALIBRATION_DIR, f"isotonic_{stamp}.joblib" if self.method=="isotonic" else f"platt_{stamp}.joblib")
        joblib.dump({"method": self.method, "model": self._iso or self._platt}, path)
        # update current pointer
        cur = os.path.join(CALIBRATION_DIR, "current.joblib")
        try:
            if os.path.exists(cur):
                os.remove(cur)
            os.symlink(os.path.basename(path), cur)
        except Exception:
            # Windows fallback: copy
            joblib.dump({"method": self.method, "model": self._iso or self._platt}, cur)
        return path

    @staticmethod
    def load_current() -> Optional["OnlineCalibrator"]:
        cur = os.path.join(CALIBRATION_DIR, "current.joblib")
        if not os.path.exists(cur):
            return None
        obj = joblib.load(cur)
        method = obj.get("method", "isotonic")
        m = OnlineCalibrator(method=method)
        if method == "isotonic":
            m._iso = obj.get("model")
        else:
            m._platt = obj.get("model")
        return m

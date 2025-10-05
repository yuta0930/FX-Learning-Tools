# model_wrappers.py
# 互換性維持のためのモデルラッパーモジュール
# - 温度スケーリング済み確率分類器の薄いラッパー
# - 学習時は本クラスを使って保存し、推論時は本モジュールから解決されるようにする
# - 既存の __main__._TemperatureScaledModel で保存されたpickleにも対応するため、
#   下部で別名（エイリアス）を公開する

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any

class TemperatureScaledModel:
    def __init__(self, base, T: float = 1.0):
        self.base = base
        self.T = float(T)

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)
        logit = np.log(p / (1 - p))
        logit_scaled = logit / self.T
        p_new = 1 / (1 + np.exp(-logit_scaled))
        return np.vstack([1 - p_new, p_new]).T

# 後方互換: 旧pickleが参照する可能性のあるシンボル名
_TemperatureScaledModel = TemperatureScaledModel


# ===== 特徴寄与（SHAP/代替）の簡易ユーティリティ =====
def _unwrap_model(model):
    """TemperatureScaledModelなどのラッパーから中身を取り出す。"""
    try:
        return model.base if hasattr(model, "base") else model
    except Exception:
        return model


def _is_tree_model(model) -> bool:
    m = _unwrap_model(model)
    attrs = (
        hasattr(m, "get_booster")  # xgboost
        or hasattr(m, "feature_importances_")  # sklearn tree/gbm
        or (m.__class__.__name__.lower().startswith("lgbm"))
    )
    return bool(attrs)


def _try_shap_contribs(model, X) -> Optional[Dict[str, Any]]:
    try:
        import importlib
        shap = importlib.import_module('shap')  # optional dependency
    except Exception:
        return None
    try:
        m = _unwrap_model(model)
        explainer = None
        if _is_tree_model(m):
            try:
                explainer = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
            except Exception:
                explainer = shap.Explainer(m)
        else:
            explainer = shap.Explainer(m)
        sv = explainer(X)
        # sv.values shape: (n_samples, n_features)
        values = np.array(sv.values)
        base = None
        try:
            base = float(np.array(sv.base_values).mean())
        except Exception:
            base = None
        return {"contribs": values, "base": base, "method": "shap"}
    except Exception:
        return None


def _linear_contribs(model, X) -> Optional[Dict[str, Any]]:
    try:
        m = _unwrap_model(model)
        # ロジスティック/線形の係数から近似寄与
        if hasattr(m, "coef_"):
            coef = np.ravel(m.coef_)
            contribs = X * coef.reshape(1, -1)
            base = float(m.intercept_[0]) if hasattr(m, "intercept_") else 0.0
            return {"contribs": contribs, "base": base, "method": "linear_coef"}
    except Exception:
        pass
    return None


def compute_feature_contributions(model, X, *, feature_names: Optional[list] = None) -> Dict[str, Any]:
    """
    直近N本の特徴寄与を返す。優先: SHAP → 線形係数 → 失敗時は空。
    Returns dict: {contribs: np.ndarray[n,m], base: float|None, method: str, feature_names: list|None}
    """
    res = _try_shap_contribs(model, X)
    if res is None:
        res = _linear_contribs(model, X)
    if res is None:
        return {"contribs": None, "base": None, "method": "none", "feature_names": feature_names}
    res["feature_names"] = feature_names
    return res

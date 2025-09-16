# model_wrappers.py
# 互換性維持のためのモデルラッパーモジュール
# - 温度スケーリング済み確率分類器の薄いラッパー
# - 学習時は本クラスを使って保存し、推論時は本モジュールから解決されるようにする
# - 既存の __main__._TemperatureScaledModel で保存されたpickleにも対応するため、
#   下部で別名（エイリアス）を公開する

from __future__ import annotations
import numpy as np

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

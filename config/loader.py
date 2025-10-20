"""設定ローダ

YAML の階層をドット記法/属性アクセスで扱える軽量ユーティリティ。

Usage:
    from config.loader import load_config
    cfg = load_config()
    ap_floor = cfg.quality.ap_min_floor

環境変数:
    FX_CFG (任意): 指定があればそのパスを優先
"""
from __future__ import annotations
import os
try:
    import yaml  # type: ignore
except ModuleNotFoundError as e:  # 明示的ガードでユーザーに指示
    raise ModuleNotFoundError("PyYAML が未インストールです。`pip install PyYAML` を実行してください。") from e
from typing import Any, Mapping

"""後方互換: config/config.yml があれば優先、無ければ従来の training.yaml を読む。

優先順:
1) 環境変数 FX_CFG
2) config/config.yml
3) config/training.yaml
"""
_DEFAULT_PATH = os.environ.get("FX_CFG") or (
    "config/config.yml" if os.path.exists("config/config.yml") else "config/training.yaml"
)

class _Node:
    def __init__(self, data: Any):
        self._data = data
    def __getattr__(self, item):
        v = self._data[item]
        return _Node(v) if isinstance(v, Mapping) else v
    def __getitem__(self, item):
        v = self._data[item]
        return _Node(v) if isinstance(v, Mapping) else v
    def to_dict(self):
        return self._data
    def get(self, key, default=None):
        return self._data.get(key, default)
    def __repr__(self):
        return f"_Node({self._data!r})"

def load_config(path: str | None = None) -> _Node:
    path = path or _DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _Node(data)

# シングルトン的キャッシュ（import単位）
_cfg_cache: _Node | None = None

def get_config() -> _Node:
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config()
    return _cfg_cache

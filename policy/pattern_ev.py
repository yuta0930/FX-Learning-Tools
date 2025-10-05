from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

@dataclass
class PatternEV:
    kind: str
    direction: str  # "up" or "down"
    quality: float  # 0..1 or 0..100 を想定、後で正規化
    measured_up: float
    measured_down: float

    def expected_R(self, *, normalize_quality: bool = True, clip: float = 5.0) -> float:
        """
        簡易EV（R単位）を計算。
        - measured_up/down は price 差（同一単位）想定
        - Rは stop幅を1とした相対幅（ここでは幅を同じ重みで近似）
        - qualityは0..1で使う。0..100なら100で割る。
        """
        q = float(self.quality)
        if q > 1.5:  # 100スケールとみなし正規化
            q = q / 100.0
        q = float(np.clip(q, 0.0, 1.0)) if normalize_quality else float(q)
        # 方向に応じた測定幅をRと見なす（簡易）
        r_raw = self.measured_up if self.direction == "up" else self.measured_down
        # 正規化のためクリップ
        r_raw = float(np.clip(r_raw, -clip, clip))
        # 品質を重みとして乗算
        return float(q * r_raw)


def pattern_list_expected_R(patterns: List[Dict[str, Any]]) -> Optional[float]:
    """
    patterns: app.pyで作っている Pattern dict/obj のリスト想定。
    直近の高品質パターンの期待Rを代表値として返す（なければNone）。
    """
    if not patterns:
        return None
    rows = []
    for p in patterns:
        try:
            kind = p.get('kind') if hasattr(p, 'get') else getattr(p, 'kind', '')
            dir_bias = p.get('direction_bias') if hasattr(p, 'get') else getattr(p, 'direction_bias', '')
            q = float(p.get('quality') if hasattr(p, 'get') else getattr(p, 'quality', 0.0))
            from app import measured_targets  # reuse existing helper
            mt = measured_targets(p)
            ev = PatternEV(kind=kind, direction='up' if str(dir_bias).lower()=="up" else 'down',
                           quality=q, measured_up=float(mt.get('up', 0.0)), measured_down=float(mt.get('down', 0.0)))
            rows.append((q, ev.expected_R()))
        except Exception:
            continue
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    return float(rows[0][1])


def pattern_expected_R_for_dir(patterns: List[Dict[str, Any]], direction: str) -> Optional[float]:
    """
    指定方向（'up'|'down'）のパターンのみから代表的な期待Rを返す。
    """
    if not patterns or direction not in ("up","down"):
        return None
    rows = []
    for p in patterns:
        try:
            kind = p.get('kind') if hasattr(p, 'get') else getattr(p, 'kind', '')
            dir_bias = p.get('direction_bias') if hasattr(p, 'get') else getattr(p, 'direction_bias', '')
            if str(dir_bias).lower() not in ("up","down"):
                continue
            if str(dir_bias).lower() != direction:
                continue
            q = float(p.get('quality') if hasattr(p, 'get') else getattr(p, 'quality', 0.0))
            from app import measured_targets
            mt = measured_targets(p)
            ev = PatternEV(kind=kind, direction=direction,
                           quality=q, measured_up=float(mt.get('up', 0.0)), measured_down=float(mt.get('down', 0.0)))
            rows.append((q, ev.expected_R()))
        except Exception:
            continue
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    return float(rows[0][1])

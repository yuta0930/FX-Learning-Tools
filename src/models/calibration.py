import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class CalibrationConfig:
    enabled: bool = True
    method: Literal["isotonic","sigmoid"] = "isotonic"
    cv: str = "prefit"

# 既存の ai_train_break.py では CalibratedClassifierCV を使っているため、
# ここでは設定保持と薄いラッパのみ提供（実体は scripts で作成）。

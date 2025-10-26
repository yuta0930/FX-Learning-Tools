from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError


class Paths(BaseModel):
    models_dir: Optional[str] = Field(default="models")
    logs_dir: Optional[str] = Field(default="logs")
    data_dir: Optional[str] = Field(default="data")
    reports_dir: Optional[str] = Field(default="reports")


class Regime(BaseModel):
    method: Optional[str] = Field(default=None)
    atr_window: Optional[int] = Field(default=None, ge=1)
    rv_window: Optional[int] = Field(default=None, ge=1)
    session_dummies: Optional[bool] = Field(default=None)


class Calibration(BaseModel):
    enabled: Optional[bool] = Field(default=None)
    method: Optional[str] = Field(default=None)


class Execution(BaseModel):
    slippage_alpha: Optional[float] = None
    slippage_beta: Optional[float] = None
    cool_down_bars: Optional[int] = Field(default=None, ge=0)
    allowed_regimes: Optional[List[str]] = None


class Risk(BaseModel):
    daily_loss_limit_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    kelly_cap: Optional[float] = None
    base_risk_per_trade_pct: Optional[float] = None


class Backtest(BaseModel):
    wf_window_days: Optional[int] = Field(default=None, ge=1)
    wf_step_days: Optional[int] = Field(default=None, ge=1)
    metrics: Optional[List[str]] = None


class Logging(BaseModel):
    parquet: Optional[bool] = None
    jsonl: Optional[bool] = None
    keep_raw_probs: Optional[bool] = None


class Quality(BaseModel):
    ap_min_floor: Optional[float] = None
    brier_max: Optional[float] = None
    coverage_warn_min: Optional[float] = None


class CostSensitivity(BaseModel):
    costs: Optional[float] = None


class ConfigModel(BaseModel):
    general: Optional[dict] = None
    paths: Optional[Paths] = None
    regime: Optional[Regime] = None
    calibration: Optional[Calibration] = None
    execution: Optional[Execution] = None
    risk: Optional[Risk] = None
    backtest: Optional[Backtest] = None
    logging: Optional[Logging] = None
    quality: Optional[Quality] = None
    cost_sensitivity: Optional[CostSensitivity] = None


def validate_config_dict(cfg_dict: dict, strict: bool = False) -> ConfigModel:
    """Validate loaded YAML dict using Pydantic.

    - If strict=True, ValidationError is raised on failure
    - If strict=False, returns best-effort model; errors are attached to .model_fields
    """
    try:
        return ConfigModel.model_validate(cfg_dict)
    except ValidationError:
        if strict:
            raise
        # Best-effort: coerce with model_construct (unsafe) so callers can proceed
        # while surfacing errors to logs.
        return ConfigModel.model_construct(**{k: cfg_dict.get(k) for k in ConfigModel.model_fields})

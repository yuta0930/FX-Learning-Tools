from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
import os
import json
import time
import tempfile
import datetime as dt

import constants as C


@dataclass
class PersistedRiskState:
    day_key: str
    equity_day_start_ccy: float
    realized_pnl_today_ccy: float
    losstreak: int
    cooldown_until_epoch: float


@dataclass
class RiskConfig:
    daily_loss_cap_pct: float = 0.03
    cooldown_on_losstreak: int = 3
    cooldown_minutes: int = 45


class RiskManager:
    """
    Persisted daily risk state with loss cap and loss-streak cooldown.

    - State file: artifacts/risk/state_{YYYYMMDD}.json
    - allow_new_trade(...): apply checks before new entry
    - on_trade_close(pnl_ccy): update realized PnL and loss streak
    - reset_if_new_day(): ensure state file for today
    """

    def __init__(self, cfg: Optional[RiskConfig] = None, artifacts_dir_override: Optional[str] = None):
        """Construct RiskManager.

        Backward compatible with previous usage (cfg as RiskConfig).
        Also supports alternative signature used in tests: RiskManager(cfg_dict, art_dir).

        If cfg is a dict-like with key 'gate', interpret it as gate.yml node; else use RiskConfig dataclass.
        """
        # Interpret cfg variants
        if isinstance(cfg, dict) and "gate" in cfg:
            g = cfg.get("gate", {}) or {}
            self.cfg = RiskConfig(
                daily_loss_cap_pct=float(g.get("daily_loss_cap_pct", 0.03)),
                cooldown_on_losstreak=int(g.get("cooldown_on_losstreak", 3)),
                cooldown_minutes=int(g.get("cooldown_minutes", 45)),
            )
            self._halt_on_weekend = bool(g.get("halt_on_weekend", False))
        else:
            self.cfg = cfg or RiskConfig()
            # read halt_on_weekend from gate.yml lazily in allow_new_trade via components if needed; default False here
            self._halt_on_weekend = False
        # artifacts directory
        self._art_dir = artifacts_dir_override or getattr(C, "ARTIFACTS_DIR", "artifacts")
        self._state: Optional[PersistedRiskState] = None
        self._load_state_for_today()

    def reset_if_new_day(self) -> None:
        today_key = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")
        if self._state is None or self._state.day_key != today_key:
            self._load_state_for_today()

    # ---------- helpers for weekend / atomic IO ----------
    def _is_weekend(self) -> bool:
        if not getattr(self, "_halt_on_weekend", False):
            return False
        jst = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=9)))
        # Monday=0 .. Sunday=6
        return jst.weekday() in (5, 6)

    # ---------- internal ----------
    @property
    def _state_dir(self) -> str:
        d = os.path.join(self._art_dir, "risk")
        os.makedirs(d, exist_ok=True)
        return d

    def _state_path_for(self, day_key: str) -> str:
        return os.path.join(self._state_dir, f"state_{day_key}.json")

    def _load_state_for_today(self) -> None:
        day_key = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")
        path = self._state_path_for(day_key)
        if os.path.exists(path):
            try:
                obj = json.load(open(path, "r", encoding="utf-8"))
                self._state = PersistedRiskState(**obj)
                return
            except Exception:
                pass
        self._state = PersistedRiskState(
            day_key=day_key,
            equity_day_start_ccy=0.0,
            realized_pnl_today_ccy=0.0,
            losstreak=0,
            cooldown_until_epoch=0.0,
        )
        self._save_state()

    def _save_state(self) -> None:
        assert self._state is not None
        path = self._state_path_for(self._state.day_key)
        # atomic write: write to tmp then replace
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp", dir=self._state_dir)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self._state), f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def allow_new_trade(self, equity_day_start_ccy: float | None, realized_pnl_today_ccy: float | None, current_losstreak: Optional[int] = None) -> Tuple[bool, str]:
        """Return whether a new trade is allowed and reason code when rejected.

        Reasons (unified): weekend_halt | losstreak_cooldown | risk_daily_cap | ok
        """
        self.reset_if_new_day()
        if self._is_weekend():
            return False, "weekend_halt"
        st = self._state
        assert st is not None
        # initialize equity_day_start if provided
        if equity_day_start_ccy is not None and st.equity_day_start_ccy <= 0:
            st.equity_day_start_ccy = float(equity_day_start_ccy)
            self._save_state()
        # update realized if caller provides
        if realized_pnl_today_ccy is not None:
            st.realized_pnl_today_ccy = float(realized_pnl_today_ccy)
            self._save_state()
        # sync losstreak hint
        if current_losstreak is not None:
            st.losstreak = int(current_losstreak)
            self._save_state()
        # cooldown check
        now = time.time()
        if now < st.cooldown_until_epoch:
            return False, "losstreak_cooldown"
        # daily loss cap check
        cap_pct = float(self.cfg.daily_loss_cap_pct)
        if cap_pct > 0 and st.equity_day_start_ccy > 0:
            pnl_pct = st.realized_pnl_today_ccy / max(st.equity_day_start_ccy, 1e-9)
            if pnl_pct <= -cap_pct:
                return False, "risk_daily_cap"
        # Optional: if caller provides losstreak and threshold hit, apply cooldown immediately
        if (
            current_losstreak is not None
            and self.cfg.cooldown_on_losstreak > 0
            and st.losstreak >= int(self.cfg.cooldown_on_losstreak)
        ):
            st.cooldown_until_epoch = now + int(self.cfg.cooldown_minutes) * 60
            st.losstreak = 0
            self._save_state()
            return False, "losstreak_cooldown"
        return True, "ok"

    def on_trade_close(self, pnl_ccy: Optional[float]) -> None:
        self.reset_if_new_day()
        st = self._state
        assert st is not None
        if pnl_ccy is None:
            return
        pnl_ccy = float(pnl_ccy)
        st.realized_pnl_today_ccy += pnl_ccy
        if pnl_ccy <= 0:
            st.losstreak += 1
        else:
            st.losstreak = 0
        # trigger cooldown if needed
        if self.cfg.cooldown_on_losstreak > 0 and st.losstreak >= int(self.cfg.cooldown_on_losstreak):
            st.cooldown_until_epoch = time.time() + int(self.cfg.cooldown_minutes) * 60
            st.losstreak = 0
        self._save_state()

"""UI: sidebar settings (partial extraction).

This file extracts a focused, relatively self-contained chunk of app.py sidebar:
- cost inputs
- auto refresh
- auto params status block

We keep behavior identical by continuing to read/write st.session_state keys.
"""

from __future__ import annotations

import numpy as np
import streamlit as st


def render_cost_and_refresh_and_auto_params(*, interval: str) -> float:
    """Render sidebar blocks and return extra_cost_pips.

    Side effects (session_state):
      - refresh_secs
      - auto_params_force_enabled

    Returns:
      extra_cost_pips: float
    """

    st.sidebar.subheader("取引コスト設定（pips）")
    fee_commission = st.sidebar.number_input(
        "手数料（往復）", min_value=0.0, max_value=5.0, value=0.00, step=0.01
    )
    fee_slippage = st.sidebar.number_input(
        "スリッページ（平均）", min_value=0.0, max_value=5.0, value=0.20, step=0.01
    )
    fee_gap = st.sidebar.number_input(
        "ギャップ控除（期待値）", min_value=0.0, max_value=10.0, value=0.00, step=0.01
    )
    extra_cost_pips = float(fee_commission + fee_slippage + fee_gap)

    # --- auto refresh ---
    st.sidebar.subheader("自動更新")
    auto_refresh = st.sidebar.checkbox("自動で再取得（ページ再読み込み）", value=True)

    # recommended_refresh_secs is defined in app.py; we import lazily to avoid circular deps.
    def recommended_refresh_secs(interval: str) -> int:
        """足種に応じた推奨の自動更新間隔（秒）を返す。

        app.py 側の関数への依存を避ける（循環import防止）ため、このモジュールに局所実装する。
        既存ロジックと同等の意図（短い足ほど短め、長い足ほど長め）を保つ。
        """

        # 既存アプリの前提: interval は "15m" / "60m" / "1h" 等の表記が混在しうる
        s = (interval or "").strip().lower()
        if s in {"15m", "m15", "15"}:
            return 180
        if s in {"30m", "m30", "30"}:
            return 300
        if s in {"60m", "1h", "h1", "60"}:
            return 600
        # デフォルト（保守的）
        return 300

    _default_refresh = int(st.session_state.get("refresh_secs", recommended_refresh_secs(interval)))
    refresh_secs = st.sidebar.slider(
        "更新間隔（秒）",
        30,
        600,
        _default_refresh,
        help="15分足は60〜180秒が目安 / 1時間足は300〜600秒が目安",
    )
    st.session_state["refresh_secs"] = int(refresh_secs)

    try:
        from streamlit_autorefresh import st_autorefresh

        if auto_refresh:
            st_autorefresh(interval=refresh_secs * 1000, limit=None, key="fx_autorefresh")
    except Exception:
        if auto_refresh:
            from streamlit.components.v1 import html

            html(
                f"""<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_secs*1000)});</script>""",
                height=0,
            )

    # --- auto params ---
    st.sidebar.subheader("Auto params")
    auto_force = st.sidebar.checkbox(
        "マーケットプロファイル自動調整を強制ON",
        value=bool(st.session_state.get("auto_params_force_enabled", False)),
        help="ConfigでOFFでも、このチェックを入れるとMarketProfile由来のθ/ATR/リスク調整を適用します。",
    )
    st.session_state["auto_params_force_enabled"] = bool(auto_force)

    with st.sidebar.expander("⚙️ Auto params status", expanded=False):
        auto_enabled = bool(st.session_state.get("auto_params_enabled", False))
        auto_cfg_enabled = bool(st.session_state.get("auto_params_config_enabled", auto_enabled))
        auto_override = bool(st.session_state.get("auto_params_override_active", False))
        status_badge = "🟢 ON" if auto_enabled else "⚪ OFF"
        st.markdown(f"**Enabled:** {status_badge}")
        st.write(
            f"config: {'ON' if auto_cfg_enabled else 'OFF'}  |  override toggle: {'ON' if auto_override else 'OFF'}"
        )

        profile = st.session_state.get("market_profile")
        session_label = getattr(profile, "session", None) if profile else None
        atr_regime = getattr(profile, "atr_regime", None) if profile else None
        theta_effective = st.session_state.get("theta_base_effective")
        theta_offset = st.session_state.get("theta_auto_offset")
        atr_override = st.session_state.get("atr_filter_override", {}) or {}
        atr_min = atr_override.get("min")
        atr_max = atr_override.get("max")
        guard_override = bool(st.session_state.get("risk_guard_override_active", False))

        def _fmt_num(val, digits: int = 3) -> str:
            if isinstance(val, (int, float)) and np.isfinite(val):
                return f"{float(val):.{digits}f}"
            return "-"

        st.caption("現在のマーケットプロファイル")
        st.write(f"Session: {session_label or '-'}  |  ATR regime: {atr_regime or '-'}")

        st.caption("θ & ATR オーバーライド")
        st.write(f"θ base eff: {_fmt_num(theta_effective)}")
        st.write(f"θ offset: {_fmt_num(theta_offset)}")
        st.write(f"ATR filter: {_fmt_num(atr_min, 4)} 〜 {_fmt_num(atr_max, 4)}")

        st.caption("Risk guard")
        guard_badge = "🟢 override active" if guard_override else "⚪ base config"
        st.write(guard_badge)
        if guard_override:
            cfg = st.session_state.get("risk_guard_current_cfg")
            if cfg is not None:
                st.write(
                    f"max/day: {cfg.max_trades_per_day} | max/session: {cfg.max_trades_per_session} | loss→cooldown: {cfg.max_consecutive_losses}"
                )

    return extra_cost_pips

import pandas as pd
import numpy as np
from pathlib import Path

# Diagnostic replica of app.detect_asia_box_break with verbose reasons
# NOTE: Keep defaults aligned with app.py as of 2025-11-07

def detect_asia_box_break_diag(
    df: pd.DataFrame,
    *,
    asia_start: str = "09:00",
    asia_end: str = "15:45",
    break_buffer: float = 0.05,
    retest_offset: float = 0.02,
    sl_buffer_min: float = 0.02,
    atr_window: int = 14,
    min_range_pips: float = 0.15,
    min_range_atrK: float = 0.8,
    max_range_atrK: float = 2.5,
    wickiness_max: float = 0.6,
    vol_ratio_min: float = 0.8,
    consider_dst: bool = True,
    tz: str = "Asia/Tokyo",
    now_ts: pd.Timestamp | None = None,
    windows_df: pd.DataFrame | None = None,
):
    reasons = []
    try:
        if df.empty:
            reasons.append("df.empty")
            return [], reasons, {}
        df2 = df.copy()
        if not isinstance(df2.index, pd.DatetimeIndex):
            if "timestamp" in df2.columns:
                df2 = df2.set_index(pd.to_datetime(df2["timestamp"]))
            else:
                reasons.append("no datetime index")
                return [], reasons, {}
        if df2.index.tz is None:
            df2.index = df2.index.tz_localize(tz)
        else:
            df2.index = df2.index.tz_convert(tz)

        now_ts = now_ts or pd.Timestamp.now(tz=df2.index.tz)
        day = now_ts.normalize()
        t_asia_start = pd.Timestamp(f"{day.date()} {asia_start}", tz=df2.index.tz)
        t_asia_end   = pd.Timestamp(f"{day.date()} {asia_end}",   tz=df2.index.tz)

        asia = df2[(df2.index >= t_asia_start) & (df2.index <= t_asia_end)]
        if asia.empty or len(asia) < 5:
            reasons.append(f"asia empty/short (len={len(asia)})")
            return [], reasons, {}

        AsiaHigh = float(asia["high"].max())
        AsiaLow  = float(asia["low"].min())
        Range    = float(AsiaHigh - AsiaLow)

        high = asia["high"].astype(float)
        low  = asia["low"].astype(float)
        open_ = asia["open"].astype(float)
        close = asia["close"].astype(float)
        body_max = pd.concat([open_, close], axis=1).max(axis=1)
        body_min = pd.concat([open_, close], axis=1).min(axis=1)
        upper_wick = (high - body_max).clip(lower=0)
        lower_wick = (body_min - low).clip(lower=0)
        denom = (high - low).replace(0, np.nan)
        wickiness = float(((upper_wick + lower_wick) / denom).mean()) if denom.notna().any() else 1.0
        if wickiness > float(wickiness_max):
            reasons.append(f"wickiness {wickiness:.3f} > max {wickiness_max}")
            return [], reasons, {}

        pc = df2["close"].astype(float).shift(1)
        tr = pd.concat([
            (df2["high"]-df2["low"]).abs(),
            (df2["high"]-pc).abs(),
            (df2["low"]-pc).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(atr_window, min_periods=max(8, atr_window//2)).mean().ffill()
        atr14_asia = float(atr14.loc[asia.index].median()) if not atr14.loc[asia.index].empty else float("nan")
        lookback_bars = min(len(df2), 96*20)
        atr14_med20d = float(atr14.tail(lookback_bars).median()) if lookback_bars >= atr_window else float("nan")
        if np.isfinite(atr14_asia) and np.isfinite(atr14_med20d) and atr14_med20d > 0:
            vol_ratio = atr14_asia / atr14_med20d
            if vol_ratio < float(vol_ratio_min):
                reasons.append(f"vol_ratio {vol_ratio:.3f} < min {vol_ratio_min}")
                return [], reasons, {}
        else:
            reasons.append("atr14_asia or atr14_med20d not finite")
            return [], reasons, {}

        min_range = max(float(min_range_pips), float(min_range_atrK) * (atr14_asia if np.isfinite(atr14_asia) else 0.0))
        max_range = float(max_range_atrK) * (atr14_asia if np.isfinite(atr14_asia) else np.inf)
        if Range < min_range or Range > max_range:
            reasons.append(f"Range {Range:.5f} outside [{min_range:.5f}, {max_range:.5f}] (atr14_asia={atr14_asia:.5f})")
            return [], reasons, {}

        pre = df2[(df2.index >= t_asia_start) & (df2.index < t_asia_end)]
        if not pre.empty:
            if (pre["close"] > AsiaHigh + break_buffer).any() or (pre["close"] < AsiaLow - break_buffer).any():
                reasons.append("pre-break occurred before box end")
                return [], reasons, {}

        # Time window for London open
        def is_bst(d: pd.Timestamp) -> bool:
            if not consider_dst:
                return True
            m = d.month
            return 4 <= m <= 10

        if is_bst(day):
            open_start = pd.Timestamp(f"{day.date()} 15:45", tz=df2.index.tz)
            open_end   = pd.Timestamp(f"{day.date()} 16:45", tz=df2.index.tz)
        else:
            open_start = pd.Timestamp(f"{day.date()} 15:45", tz=df2.index.tz)
            open_end   = pd.Timestamp(f"{day.date()} 17:45", tz=df2.index.tz)

        last_row = df2.iloc[-1]
        last_ts: pd.Timestamp = last_row.name
        last_close = float(last_row["close"])
        in_window = (open_start <= last_ts <= open_end)

        # News suppression check (only affects signal emission, not filters)
        def _in_news(ts: pd.Timestamp) -> bool:
            try:
                if windows_df is not None and not windows_df.empty:
                    from is_in_any_window import is_in_any_window
                    return bool(is_in_any_window(pd.Series([ts]), windows_df[["start","end"]])[0])
                return False
            except Exception:
                return False

        details = {
            "AsiaHigh": AsiaHigh,
            "AsiaLow": AsiaLow,
            "Range": Range,
            "wickiness": wickiness,
            "atr14_asia": atr14_asia,
            "atr14_med20d": atr14_med20d,
            "last_ts": str(last_ts),
            "in_window": bool(in_window),
            "open_start": str(open_start),
            "open_end": str(open_end),
            "last_close": last_close,
            "buy_stop": AsiaHigh + break_buffer,
            "sell_stop": AsiaLow - break_buffer,
            "buy_limit": AsiaHigh + retest_offset,
            "sell_limit": AsiaLow - retest_offset,
            "news": bool(_in_news(last_ts)),
        }

        # Mimic outputs: if filters pass but not in window, retest_limit would be emitted by app
        out = [
            ("asia_box", {"mode": "retest_limit", "buy_limit": details["buy_limit"], "sell_limit": details["sell_limit"]})
        ]
        if in_window and not details["news"]:
            if last_close > details["buy_stop"]:
                out.append(("asia_box_bull", {"mode":"close_break"}))
            elif last_close < details["sell_stop"]:
                out.append(("asia_box_bear", {"mode":"close_break"}))
            else:
                # also would propose stop_pending in app
                out.append(("asia_box", {"mode": "stop_pending", "buy_stop": details["buy_stop"], "sell_stop": details["sell_stop"]}))
        return out, reasons, details
    except Exception as e:
        reasons.append(f"exception: {e}")
        return [], reasons, {}


def load_usdjpy_15m(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # Expect columns: time or timestamp, open, high, low, close
    ts_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if ts_col is None:
        # try typical naming
        for c in df.columns:
            if "time" in c.lower():
                ts_col = c
                break
    if ts_col is None:
        raise ValueError("No time/timestamp column found in CSV")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.set_index(ts_col, inplace=True)
    for c in ["open","high","low","close"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
        else:
            raise ValueError(f"Missing column: {c}")
    return df


def main():
    # Use repo default CSV
    data_path = Path("data/USDJPY_15m.csv")
    df = load_usdjpy_15m(data_path)
    # Ensure tz for index; app localizes to Asia/Tokyo when absent
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Tokyo")
    # Evaluate for today by default
    out, reasons, details = detect_asia_box_break_diag(df)
    print("=== Asia Box Diagnose (today) ===")
    if reasons:
        print("Filters blocked:", "; ".join(reasons))
    else:
        print("Filters passed.")
    print("Details:")
    for k, v in details.items():
        print(f"  {k}: {v}")
    print("Signals:")
    for kind, params in out:
        print("  -", kind, params)

    # Also try rolling back to London window in the latest day to simulate in-window
    tz = df.index.tz
    today = pd.Timestamp.now(tz=tz).normalize()
    for hour in [16, 17]:
        trial_now = today + pd.Timedelta(hours=hour, minutes=10)
        out2, reasons2, details2 = detect_asia_box_break_diag(df[df.index <= trial_now], now_ts=trial_now)
        print(f"\n=== Simulate at {trial_now} ===")
        if reasons2:
            print("Filters blocked:", "; ".join(reasons2))
        else:
            print("Filters passed.")
        print("Details:")
        for k, v in details2.items():
            print(f"  {k}: {v}")
        print("Signals:")
        for kind, params in out2:
            print("  -", kind, params)

    # Summary over last 7 days
    print("\n=== Summary: last 7 days (diagnose at ~16:10 JST) ===")
    reasons_count: dict[str,int] = {}
    rows = []
    for i in range(7):
        day_i = today - pd.Timedelta(days=i)
        trial_now = day_i + pd.Timedelta(hours=16, minutes=10)
        dfi = df[(df.index >= day_i) & (df.index <= trial_now)]
        outi, ri, det = detect_asia_box_break_diag(dfi, now_ts=trial_now)
        if ri:
            key = ri[0]
            reasons_count[key] = reasons_count.get(key, 0) + 1
            status = f"blocked: {key}"
        else:
            status = "passed"
        rows.append((str(day_i.date()), status, det.get("Range"), det.get("atr14_asia")))
    for r in rows:
        print("  ", r)
    print("Reasons tally:")
    for k, v in reasons_count.items():
        print(f"  {k}: {v}")

    # Analyze Range/ATR ratio distribution for last 30 days
    print("\n=== Range/ATR14_asia ratio (last 30 days) ===")
    ratios = []
    tz = df.index.tz
    for i in range(30):
        d = today - pd.Timedelta(days=i)
        t0 = pd.Timestamp(f"{d.date()} 09:00", tz=tz)
        t1 = pd.Timestamp(f"{d.date()} 15:45", tz=tz)
        asia = df[(df.index >= t0) & (df.index <= t1)]
        if asia.empty or len(asia) < 5:
            continue
        # ATR14 median in asia window
        pc = df["close"].astype(float).shift(1)
        tr = pd.concat([
            (df["high"]-df["low"]).abs(),
            (df["high"]-pc).abs(),
            (df["low"]-pc).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=7).mean().ffill()
        atr_asia = float(atr14.loc[asia.index].median()) if not atr14.loc[asia.index].empty else np.nan
        if not np.isfinite(atr_asia) or atr_asia <= 0:
            continue
        rng = float(asia["high"].max() - asia["low"].min())
        ratios.append(rng / atr_asia)
    if ratios:
        s = pd.Series(ratios)
        for p in [50, 75, 90, 95, 99]:
            print(f"  p{p}: {s.quantile(p/100):.2f}")
        print(f"  mean: {s.mean():.2f}  min: {s.min():.2f}  max: {s.max():.2f}  n={len(ratios)}")
    else:
        print("  (no data)")


if __name__ == "__main__":
    main()

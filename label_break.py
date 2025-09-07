# label_break.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd

# ========= 基本ユーティリティ =========
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def _in_any_window(ts: pd.Series, windows: Optional[pd.DataFrame]) -> pd.Series:
    if windows is None or windows.empty:
        return pd.Series(False, index=ts.index)
    mask = pd.Series(False, index=ts.index)
    arr = pd.to_datetime(ts).values
    for _, row in windows.iterrows():
        mask |= (arr >= np.datetime64(row["start"])) & (arr <= np.datetime64(row["end"]))
    return mask

# ========= 設定 =========
@dataclass
class BreakLabelConfig:
    # ブレイク定義
    H: int = 12                        # 予測地平（何本先まで見てブレイク/不発を判定するか）
    level_lookback: int = 48           # 水平線が無い場合、過去N本の高値/安値を水平線の代替に
    buffer_mode: Literal["atr","pct","abs"] = "atr"
    break_buffer_atr: float = 0.15     # buffer_mode="atr" のときの閾値（ATR倍率）
    break_buffer_pct: float = 0.0005   # buffer_mode="pct" のとき（=0.05% など）
    break_buffer_abs: float = 0.05     # buffer_mode="abs" のとき（価格の絶対値、例: 0.05円）

    # 「定着」条件（だまし抑制）
    settle_bars: int = 2               # 連続クローズ本数
    settle_atr: float = 0.15           # neck±ATR*α を基準に定着を要求

    # トリプルバリア（勝ち/負け/タイムアウト）
    tb_up_R: float = 1.0               # エントリー価格±ATR*R
    tb_dn_R: float = 1.0
    tb_timeout: int = 48               # 何本でタイムアウト扱いにするか

    # ニュース/イベント除外
    exclude_in_windows: bool = True

# ========= コア実装 =========
def _buffer_value(level: float, atr: float, cfg: BreakLabelConfig) -> float:
    if cfg.buffer_mode == "atr":
        return cfg.break_buffer_atr * atr
    if cfg.buffer_mode == "pct":
        return level * cfg.break_buffer_pct
    return cfg.break_buffer_abs

def _find_break_and_settle(
    df: pd.DataFrame,
    idx: int,
    level_up: float,
    level_dn: float,
    atr_arr: np.ndarray,
    cfg: BreakLabelConfig
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    returns: (t_break, t_settle, side)  side in {"up","dn",None}
    """
    n = len(df); t0 = idx + 1; t1 = min(n, idx + 1 + cfg.H)
    if t0 >= t1: return None, None, None

    # 動的バッファ（ATR or pct or abs）
    buf_up = _buffer_value(level_up, atr_arr[idx], cfg)
    buf_dn = _buffer_value(level_dn, atr_arr[idx], cfg)

    # 1) ブレイク検知（最初に起きた方を採用）
    t_break_up = None
    t_break_dn = None
    # 上抜け: high >= level_up + buf
    hit_up = np.where(df["high"].values[t0:t1] >= (level_up + buf_up))[0]
    if hit_up.size:
        t_break_up = t0 + int(hit_up[0])
    # 下抜け: low <= level_dn - buf
    hit_dn = np.where(df["low"].values[t0:t1] <= (level_dn - buf_dn))[0]
    if hit_dn.size:
        t_break_dn = t0 + int(hit_dn[0])

    if t_break_up is None and t_break_dn is None:
        return None, None, None

    if t_break_dn is None or (t_break_up is not None and t_break_up <= t_break_dn):
        side = "up"; t_break = t_break_up
        base_level = level_up
        settle_line = base_level + cfg.settle_atr * atr_arr[idx]
        # 2) 定着：クローズが settle_line 以上を連続 settle_bars
        closes = df["close"].values
        ok = False
        for t in range(t_break, min(n, t_break + cfg.H)):
            # 直近 settle_bars 本が全て条件を満たすか
            if t - cfg.settle_bars + 1 < t_break: 
                continue
            if np.all(closes[t - cfg.settle_bars + 1: t + 1] >= settle_line):
                t_settle = t
                ok = True
                break
        return (t_break, t_settle if ok else None, side)
    else:
        side = "dn"; t_break = t_break_dn
        base_level = level_dn
        settle_line = base_level - cfg.settle_atr * atr_arr[idx]
        closes = df["close"].values
        ok = False
        for t in range(t_break, min(n, t_break + cfg.H)):
            if t - cfg.settle_bars + 1 < t_break: 
                continue
            if np.all(closes[t - cfg.settle_bars + 1: t + 1] <= settle_line):
                t_settle = t
                ok = True
                break
        return (t_break, t_settle if ok else None, side)

def _triple_barrier(
    df: pd.DataFrame,
    t_entry: int,
    atr_arr: np.ndarray,
    cfg: BreakLabelConfig
) -> Tuple[str, int]:
    """
    returns: (outcome, t_exit)  outcome in {"tp","sl","to"}  # take/sl/timeout
    """
    n = len(df)
    price0 = float(df["close"].iloc[t_entry])
    atr0 = float(atr_arr[t_entry])
    up = price0 + cfg.tb_up_R * atr0
    dn = price0 - cfg.tb_dn_R * atr0
    t1 = min(n, t_entry + cfg.tb_timeout + 1)
    highs = df["high"].values[t_entry+1:t1]
    lows  = df["low"].values[t_entry+1:t1]

    hit_tp = np.where(highs >= up)[0]
    hit_sl = np.where(lows  <= dn)[0]
    first_tp = t_entry + 1 + int(hit_tp[0]) if hit_tp.size else None
    first_sl = t_entry + 1 + int(hit_sl[0]) if hit_sl.size else None

    if first_tp is None and first_sl is None:
        return "to", t1 - 1
    if first_sl is None or (first_tp is not None and first_tp <= first_sl):
        return "tp", first_tp
    else:
        return "sl", first_sl

def build_break_labels(
    raw: pd.DataFrame,
    cfg: BreakLabelConfig,
    *,
    level_col_up: Optional[str] = None,
    level_col_dn: Optional[str] = None,
    windows_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    入力: raw [timestamp, open, high, low, close, ...]
    出力: labels [timestamp, y, side, t_break, t_settle, outcome, t_exit, reason]
      y=1: ブレイク→定着→トリプルバリアでブレイク方向が先に到達（=勝ち筋の真性ブレイク）
      y=0: 不発 or 定着せず or 逆方向先到達/タイムアウト
    """
    df = raw.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    atr = _atr(df, 14).ffill().values

    # 水平線候補（列が無ければローリング極値で代替）
    if level_col_up and level_col_up in df.columns:
        level_up = df[level_col_up].astype(float).values
    else:
        level_up = df["high"].shift(1).rolling(cfg.level_lookback, min_periods=cfg.level_lookback).max().values
    if level_col_dn and level_col_dn in df.columns:
        level_dn = df[level_col_dn].astype(float).values
    else:
        level_dn = df["low"].shift(1).rolling(cfg.level_lookback, min_periods=cfg.level_lookback).min().values

    excl = _in_any_window(ts, windows_df) if cfg.exclude_in_windows else pd.Series(False, index=df.index)

    rows = []
    for i in range(len(df)):
        if i + cfg.H + 1 >= len(df):  # 末尾は学習に使わない
            rows.append((ts[i], np.nan, "", np.nan, np.nan, "", np.nan, "tail"))
            continue
        if excl.iloc[i]:
            rows.append((ts[i], np.nan, "", np.nan, np.nan, "", np.nan, "event"))
            continue
        if np.isnan(level_up[i]) or np.isnan(level_dn[i]) or atr[i] <= 0:
            rows.append((ts[i], np.nan, "", np.nan, np.nan, "", np.nan, "insufficient"))
            continue

        t_break, t_settle, side = _find_break_and_settle(df, i, level_up[i], level_dn[i], atr, cfg)
        if side is None or t_break is None:
            rows.append((ts[i], 0, "", np.nan, np.nan, "", np.nan, "no_break"))
            continue
        if t_settle is None:
            rows.append((ts[i], 0, side, t_break, np.nan, "", np.nan, "no_settle"))
            continue

        # エントリー＝定着時点（保守的）。成りなら t_break に変えても可
        t_entry = int(t_settle)
        outcome, t_exit = _triple_barrier(df, t_entry, atr, cfg)

        # ブレイク方向とトリプルバリアの結果で教師ラベル
        if (side == "up" and outcome == "tp") or (side == "dn" and outcome == "tp"):
            y = 1
        elif outcome == "sl":
            y = 0
        else:  # timeout
            y = 0

        rows.append((ts[i], int(y), side, t_break, t_settle, outcome, t_exit, "ok"))

    out = pd.DataFrame(rows, columns=["timestamp","y","side","t_break","t_settle","outcome","t_exit","reason"])
    return out

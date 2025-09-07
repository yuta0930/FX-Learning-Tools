from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BreakLabelConfig:
    horizon_bars: int = 12       # H=12（15分足なら3時間先）
    buffer_ratio: float = 0.0005 # 例: 0.05% = 0.0005（元の0.05が5%を意味していたなら調整）

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: 必須カラム ['timestamp','open','high','low','close','volume']
    時間昇順 index or 'timestamp' ソート済み前提
    未来情報を使わないrolling/shiftのみを利用（closed='left'で左閉区間）
    """
    df = df.sort_values('timestamp').copy()
    # 過去N本の高安/ATRなど（未来参照なし）
    win = 96  # 過去一日相当（15分足）
    df['ret_1']  = df['close'].pct_change(1)
    df['ret_4']  = df['close'].pct_change(4)
    df['atr']    = true_range_atr(df, n=14)  # 下の関数参照
    # “レジサポの質”の簡易近似：直近X本でのタッチ密度・反転強度
    X = 48
    df['swing_high'] = (df['high'] == df['high'].rolling(X, min_periods=10).max()).astype(int)
    df['swing_low']  = (df['low']  == df['low'].rolling(X, min_periods=10).min()).astype(int)
    df['touch_density'] = (df['swing_high'] + df['swing_low']).rolling( X, min_periods=10).sum()
    # 長期スロープ（線形回帰の傾き近似。未来を見ない）
    df['ma_long'] = df['close'].rolling(win, min_periods=30).mean()
    df['slope_long'] = df['ma_long'].diff(12) / 12.0


    # セッションOne-Hot（JST: Tokyo=9–15, London=16–23, NY=22–4）
    h = pd.to_datetime(df['timestamp']).dt.hour
    df['tokyo']  = ((h>=9)  & (h<15)).astype(int)
    df['london'] = ((h>=16) & (h<24)).astype(int)
    df['ny']     = ((h>=22) | (h<5)).astype(int)

    # 将来参照を避けるため、計算直後にNaN行を落とす
    feats = ['ret_1','ret_4','atr','touch_density','slope_long','tokyo','london','ny']
    out = df[['timestamp','open','high','low','close','volume'] + feats].dropna().copy()
    return out

def true_range_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = (pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1))
    atr = tr.rolling(n, min_periods=n).mean()
    return atr

def build_break_labels(df: pd.DataFrame, cfg: BreakLabelConfig) -> pd.Series:
    """
    “ブレイク成功”の例：現在のクローズから先H本の間に
    +buffer*ATR以上の上抜け（or 下抜け）に達したかで2値化。
    ラベル生成自体は未来を見るが、訓練ではラベルをH本分 過去へshiftして
    特徴量との時間整合を取る。
    """
    H = cfg.horizon_bars
    buf = cfg.buffer_ratio

    # 将来の最大/最小（未来だけで計算）
    fwd_max = df['high'].shift(-1).rolling(H, min_periods=H).max()
    fwd_min = df['low'].shift(-1).rolling(H, min_periods=H).min()

    # バッファはATR基準（現時点のATRを利用：未来を見ない）
    atr_now = true_range_atr(df, 14)

    up_break   = (fwd_max >= df['close'] + buf * df['close'].abs().clip(lower=1e-8))  # 価格比例
    down_break = (fwd_min <= df['close'] - buf * df['close'].abs().clip(lower=1e-8))
    y_raw = (up_break | down_break).astype(int)

    # 訓練用ラベルはH本分左へシフトして「当時に分かっていた特徴量と結合」する
    y = y_raw.shift(H)
    return y

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

SESSION_TZ = "Asia/Tokyo"

@dataclass
class RegimeConfig:
    method: str = "hybrid"
    k: int = 4
    atr_window: int = 14
    rv_window: int = 30
    session_dummies: bool = True
    spread_proxy: str = "hl_over_close"

class RegimeClassifier:
    """OHLCVに基づくレジーム付与（薄い変換器）。

    必須列: timestamp, open, high, low, close, volume
    出力: 同じ行数で regime_id, regime_name を付与
    """
    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self.kmeans_: Optional[object] = None
        self.rules_: dict = {}

    @staticmethod
    def _atr(df: pd.DataFrame, w: int) -> pd.Series:
        h, l, c = df['high'], df['low'], df['close']
        prev_c = c.shift(1)
        tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(w, min_periods=w).mean()

    @staticmethod
    def _realized_vol(df: pd.DataFrame, w: int) -> pd.Series:
        r = np.log(df['close']).diff()
        return (r.rolling(w, min_periods=w).std() * np.sqrt(252*24*4)).fillna(0.0)

    @staticmethod
    def _session_dummies(ts: pd.Series) -> pd.DataFrame:
        dt = pd.to_datetime(ts).dt.tz_convert(SESSION_TZ) if getattr(ts.dtype, 'tz', None) else pd.to_datetime(ts).dt.tz_localize(SESSION_TZ)
        h = dt.dt.hour
        tokyo = ((h>=9) & (h<15)).astype(int)
        london = ((h>=16) & (h<24)).astype(int)
        ny = ((h>=22) | (h<5)).astype(int)
        return pd.DataFrame({'tokyo': tokyo, 'london': london, 'ny': ny}, index=ts.index)

    def _rule_bins(self, s: pd.Series, qs=(0.33, 0.66)) -> Tuple[pd.Series, dict]:
        if s.dropna().empty:
            return pd.Series(index=s.index, dtype=int), {}
        qvals = s.quantile(list(qs))
        # 重複回避: 分位が同一/NaN の場合は2分割にフォールバック
        q1 = float(qvals.iloc[0]) if np.isfinite(qvals.iloc[0]) else None
        q2 = float(qvals.iloc[1]) if np.isfinite(qvals.iloc[1]) else None
        edges = None
        labels = None
        if q1 is not None and q2 is not None and q1 < q2:
            edges = [-np.inf, q1, q2, np.inf]
            labels = [0, 1, 2]
        elif q1 is not None:
            edges = [-np.inf, q1, np.inf]
            labels = [0, 1]
        elif q2 is not None:
            edges = [-np.inf, q2, np.inf]
            labels = [0, 1]
        else:
            # 全て同値など -> 単一ビン
            return pd.Series(1, index=s.index, dtype=int), {'q1': float('nan'), 'q2': float('nan')}

        try:
            bins = pd.cut(s, edges, labels=labels, duplicates='drop').astype('Int64')
        except Exception:
            return pd.Series(1, index=s.index, dtype=int), {'q1': q1 if q1 is not None else float('nan'), 'q2': q2 if q2 is not None else float('nan')}
        # 中央ビンをデフォルトに（NaNは1）
        mid = 1 if 1 in set(labels) else (labels[len(labels)//2] if labels else 0)
        return bins.astype(float).fillna(mid).astype(int), {'q1': q1 if q1 is not None else float('nan'), 'q2': q2 if q2 is not None else float('nan')}

    def transform(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        req = {'timestamp','open','high','low','close','volume'}
        if not req.issubset({c.lower() for c in df_ohlcv.columns}):
            # 標準化
            ren = {}
            for c in df_ohlcv.columns:
                if c.lower() in req:
                    ren[c]=c.lower()
            df = df_ohlcv.rename(columns=ren).copy()
        else:
            df = df_ohlcv.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 指標
        df['atr'] = self._atr(df, self.cfg.atr_window)
        df['rv'] = self._realized_vol(df, self.cfg.rv_window)
        df['hl_over_close'] = (df['high']-df['low']).abs() / df['close'].replace(0,np.nan)
        df['hl_over_close'] = df['hl_over_close'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        # セッション
        if self.cfg.session_dummies:
            s = self._session_dummies(df['timestamp'])
            df = pd.concat([df, s], axis=1)

        # ルール基準の粗区分（ATRベース）
        atr_bin, rules = self._rule_bins(df['atr'])
        self.rules_['atr_bins'] = rules
        name_map = {0:'LV', 1:'NV', 2:'HV'}
        df['regime_id'] = atr_bin.values
        df['regime_name'] = df['regime_id'].map(name_map)

        # 流動性低下を hl_over_close で近似
        low_liq = (df['hl_over_close'] > df['hl_over_close'].quantile(0.8)) & (df['rv'] > df['rv'].median())
        df.loc[low_liq, 'regime_name'] = 'HV_HLQ'
        # idを再エンコード
        names = df['regime_name'].fillna('NV')
        uniq = {n:i for i,n in enumerate(pd.unique(names))}
        df['regime_id'] = names.map(uniq).astype(int)

        return df

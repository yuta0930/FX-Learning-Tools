from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class WFConfig:
    window_days: int = 60
    step_days: int = 15

class WalkForward:
    def __init__(self, cfg: WFConfig):
        self.cfg = cfg

    def split(self, df: pd.DataFrame) -> List[Dict[str, pd.Timestamp]]:
        df = df.sort_values('timestamp').reset_index(drop=True)
        ts = pd.to_datetime(df['timestamp'])
        start = ts.min()
        end = ts.max()
        out = []
        cur = start
        while cur < end:
            tr_end = cur + pd.Timedelta(days=self.cfg.window_days)
            te_end = tr_end + pd.Timedelta(days=self.cfg.step_days)
            out.append({
                'train_start': cur,
                'train_end': tr_end,
                'test_start': tr_end,
                'test_end': min(te_end, end)
            })
            cur = cur + pd.Timedelta(days=self.cfg.step_days)
        return out

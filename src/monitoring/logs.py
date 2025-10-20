from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import json
import os

@dataclass
class TradeLog:
    timestamp: str
    regime: Optional[str]
    proba_raw: float
    proba_cal: float
    side: str
    size: float
    entry: float
    exit: float
    stop: float
    take: float
    spread_proxy: float
    slippage: float
    pnl: float
    dd: float
    flags: dict

class Logger:
    def __init__(self, logs_dir: str, parquet: bool = True, jsonl: bool = True):
        self.logs_dir = logs_dir
        self.parquet = parquet
        self.jsonl = jsonl
        os.makedirs(logs_dir, exist_ok=True)
        self._rows = []

    def log_trade(self, rec: TradeLog):
        d = asdict(rec)
        self._rows.append(d)
        if self.jsonl:
            with open(os.path.join(self.logs_dir, 'trades.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    def flush(self):
        if self.parquet and self._rows:
            df = pd.DataFrame(self._rows)
            df.to_parquet(os.path.join(self.logs_dir, 'trades.parquet'), engine='pyarrow', index=False)
            self._rows.clear()

# 簡易信頼度図（バックテストやオフラインで使用）
import numpy as np
import matplotlib.pyplot as plt

def plot_reliability(y_true, p_pred, n_bins=10, title=None):
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p_pred, bins) - 1
    prob = []; acc = []
    for b in range(n_bins):
        m = idx==b
        if m.sum()==0:
            continue
        prob.append(p_pred[m].mean())
        acc.append(y_true[m].mean())
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.plot(prob, acc, 'o-')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    if title: plt.title(title)
    plt.tight_layout()
    return plt.gcf()

from __future__ import annotations
import os
import pandas as pd
from constants import SIGNALS_LOG, ORDERS_LOG, TRADES_LOG, TCA_FEATS_LOG


def ensure_logs():
    os.makedirs(os.path.dirname(SIGNALS_LOG), exist_ok=True)
    for path, cols in [
        (SIGNALS_LOG, ["ts_jst","setup_id","features","p_raw","p_cal","quality","atr_15m","session","spread","news_flag","gate_decision","variant","run_id"]),
        (ORDERS_LOG,  ["ts_jst","order_id","setup_id","side","size","price_req","price_fill","slip_pips","spread","latency_ms","broker","run_id","variant"]),
        (TRADES_LOG,  ["ts_jst_open","ts_jst_close","order_id","pnl_pips","pnl_ccy","tp_hit","sl_hit","hold_secs","r_multiple","run_id","variant"]),
        (TCA_FEATS_LOG, ["ts_jst","spread","atr_1m","atr_15m","size","session","slip_pips","run_id"]),
    ]:
        if not os.path.exists(path):
            pd.DataFrame(columns=cols).to_parquet(path, index=False)
            print("created", path)

if __name__ == "__main__":
    ensure_logs()

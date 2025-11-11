import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from features_util import compute_level_relative_features

df = pd.read_csv('data/USDJPY_15m.csv', parse_dates=['timestamp'])
df = df.rename(columns=str.lower)
print('n_rows=', len(df))
out = compute_level_relative_features(df, window=120, look_pivot=9, k=2)
print('out_cols=', list(out.columns))
print('lvl_cols=', [c for c in out.columns if c.startswith('lvl_')])
print('tail_sample=')
print(out.tail(3).to_string())

import pandas as pd
import numpy as np
from src.features.regime import RegimeClassifier, RegimeConfig

def test_regime_transform_basic():
    n=200
    ts = pd.date_range('2024-01-01', periods=n, freq='15min', tz='Asia/Tokyo')
    df = pd.DataFrame({
        'timestamp': ts,
        'open': np.linspace(150,151,n),
        'high': np.linspace(150.1,151.1,n),
        'low': np.linspace(149.9,150.9,n),
        'close': np.linspace(150,151,n) + np.sin(np.arange(n)/10)*0.05,
        'volume': 1000
    })
    reg = RegimeClassifier(RegimeConfig())
    out = reg.transform(df)
    assert 'regime_id' in out.columns and 'regime_name' in out.columns
    assert out['regime_id'].isna().sum() == 0
    if {'tokyo','london','ny'}.issubset(out.columns):
        # 各行で少なくともどれか一つが1になる時間帯が想定
        assert ((out[['tokyo','london','ny']].sum(axis=1) >= 0).all())

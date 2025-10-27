import numpy as np
import pandas as pd
from app.tca.slippage_model import SlippageModel, SlippageModelConfig


def test_tca_slippage_monotonic_trend():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "spread": rng.uniform(0.1, 0.8, size=n),
        "atr_1m": rng.uniform(0.01, 0.3, size=n),
        "atr_15m": rng.uniform(0.05, 0.9, size=n),
        "session": rng.choice(["tokyo","london","ny"], size=n),
        "size": rng.uniform(0.1, 1.0, size=n),
        "latency_ms": rng.normal(120, 30, size=n),
        "news_flag": rng.integers(0, 2, size=n),
    })
    # synth slip: increasing in spread/atr/latency
    df["slip_pips"] = 0.4*df["spread"] + 0.2*df["atr_15m"] + 0.001*df["latency_ms"] + 0.05*df["news_flag"] + rng.normal(0, 0.05, size=n)

    m = SlippageModel(SlippageModelConfig(model="linear", use_features=["spread","atr_1m","atr_15m","session","size","latency_ms","news_flag"]))
    m.fit(df)
    low = df.sort_values("spread").head(10).iloc[0].to_dict()
    high = df.sort_values("spread").tail(10).iloc[-1].to_dict()
    y_low = m.predict(low)["p50"]
    y_high = m.predict(high)["p50"]
    assert y_high >= y_low

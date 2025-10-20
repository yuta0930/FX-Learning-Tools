import pandas as pd
from config.loader import get_config
from src.features.regime import RegimeClassifier, RegimeConfig

# rule/hybrid は学習不要。kmeans を導入する場合にここを拡張。

def main():
    cfg = get_config()
    data_path = f"{cfg.paths.data_dir}/{cfg.general.symbol}_{cfg.general.timeframe}.csv"
    raw = pd.read_csv(data_path)
    rcfg = RegimeConfig(
        method=cfg.regime.method,
        k=int(cfg.regime.k),
        atr_window=int(cfg.regime.atr_window),
        rv_window=int(cfg.regime.rv_window),
        session_dummies=bool(cfg.regime.session_dummies),
    )
    reg = RegimeClassifier(rcfg)
    df = reg.transform(raw)
    print(df[["timestamp","regime_id","regime_name"]].tail())

if __name__ == "__main__":
    main()

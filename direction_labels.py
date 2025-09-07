# direction_labels.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class DirLabelConfig:
    horizon_bars: int = 12
    buffer_ratio: float = 0.0005

# Up/Down方向ラベルを生成する関数
# df: 必須カラム ['timestamp', 'open', 'high', 'low', 'close']
def build_direction_labels(df: pd.DataFrame, cfg: DirLabelConfig) -> pd.DataFrame:
    """
    指定した horizon_bars 先の価格変化で Up/Down ラベルを付与。
    buffer_ratio は閾値（上下どちらにも動かなかった場合はラベルなし）。
    戻り値: DataFrame(index=df.index, columns=['y_up', 'y_down'])
    """
    close = df['close'].values
    y_up = []
    y_down = []
    for i in range(len(df)):
        if i + cfg.horizon_bars >= len(df):
            y_up.append(None)
            y_down.append(None)
            continue
        future_close = close[i + cfg.horizon_bars]
        ret = (future_close - close[i]) / close[i]
        if ret > cfg.buffer_ratio:
            y_up.append(1)
            y_down.append(0)
        elif ret < -cfg.buffer_ratio:
            y_up.append(0)
            y_down.append(1)
        else:
            y_up.append(0)
            y_down.append(0)
    return pd.DataFrame({'y_up': y_up, 'y_down': y_down}, index=df.index)

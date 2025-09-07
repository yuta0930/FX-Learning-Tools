import pandas as pd

def is_in_any_window(timestamps: pd.Series, windows: pd.DataFrame) -> pd.Series:
    """
    各timestampがwindowsのいずれかの[start, end]区間に含まれるかを返す。
    windows: DataFrame with columns ["start", "end"] (both pd.Timestamp, same tz)
    戻り値: pd.Series[bool] (timestampsと同じ長さ)
    """
    if windows is None or windows.empty:
        return pd.Series([False]*len(timestamps), index=timestamps.index)
    # 各timestampについて、どれかのウィンドウに入っているか
    arr = timestamps.values
    mask = pd.Series(False, index=timestamps.index)
    for _, row in windows.iterrows():
        mask |= (arr >= row["start"]) & (arr <= row["end"])
    return mask

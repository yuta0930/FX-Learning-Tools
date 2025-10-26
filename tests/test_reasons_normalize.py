import pandas as pd
from pathlib import Path
from src.policy.reasons_normalize import load_reason_map, normalize_reasons_series


def test_normalize_series_basic(tmp_path: Path):
    yml = tmp_path / "reasons_map.yml"
    yml.write_text(
        "categories:\n spread: ['spread>', 'スプレッド']\n news: ['news_window']\n",
        encoding="utf-8",
    )
    mapping = load_reason_map(yml)
    s = pd.Series(["spread>1.0", "news_window±10/10m", "other"])
    out = normalize_reasons_series(s, mapping)
    assert out.iloc[0] == "spread"
    assert out.iloc[1] == "news"
    assert out.iloc[2] == "other"

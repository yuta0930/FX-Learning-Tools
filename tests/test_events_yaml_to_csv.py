from pathlib import Path

from src.core.events import load_events_yaml, events_to_df


def test_events_yaml_to_df(tmp_path: Path):
    yml = tmp_path / "events.yml"
    yml.write_text(
        "events:\n"
        "  - {title: 'X', date: 2025-01-02, time: '09:00', importance: low}\n"
        "  - {title: 'Y', date: 2025-01-03}\n",
        encoding="utf-8",
    )

    items = load_events_yaml(yml)
    df = events_to_df(items)
    assert list(df.columns) == ["time", "title", "importance"]
    assert len(df) == 2
    assert df.iloc[0]["title"] == "X"
    assert df.iloc[1]["title"] == "Y"

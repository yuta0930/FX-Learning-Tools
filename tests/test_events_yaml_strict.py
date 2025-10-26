from pathlib import Path
import subprocess
import sys
import yaml


def test_make_events_csv_strict(tmp_path: Path):
    yml = tmp_path / "events.yml"
    out = tmp_path / "events.csv"

    # 正常（strict）
    yml.write_text(
        yaml.safe_dump({"events": [{"title": "X", "date": "2025-01-02", "time": "09:00"}]}),
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, "scripts/make_events_csv.py", "--in", str(yml), "--out", str(out), "--strict"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0

    # 異常（strictで落ちる）
    yml.write_text(
        yaml.safe_dump({"events": [{"title": "X", "date": "2025-01-02", "time": "9:00"}]}),
        encoding="utf-8",
    )
    r2 = subprocess.run(
        [sys.executable, "scripts/make_events_csv.py", "--in", str(yml), "--out", str(out), "--strict"],
        capture_output=True,
        text=True,
    )
    assert r2.returncode != 0

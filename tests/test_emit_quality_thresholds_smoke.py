from pathlib import Path
import json
import subprocess
import sys
import yaml


def test_emit_quality_thresholds_from_metrics(tmp_path: Path):
    m = tmp_path / "metrics.json"
    m.write_text(json.dumps({"ev_curve_best": {"quality_threshold": 0.58}}), encoding="utf-8")
    out = tmp_path / "config" / "patterns_quality_thresholds.yml"

    r = subprocess.run(
        [
            sys.executable,
            "scripts/emit_quality_thresholds.py",
            "--metrics",
            str(m),
            "--sessions",
            "London,NewYork",
            "--tokyo-thr",
            "0.10",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    y = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert y["London"] == 0.58 and y["NewYork"] == 0.58 and y["Tokyo"] == 0.10

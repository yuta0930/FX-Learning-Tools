from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime
import pandas as pd


def test_health_check_cli_drift_noop(tmp_path: Path):
    # 最小ログ（block率=0）を用意
    log = tmp_path / "exec.csv"
    pd.DataFrame(
        {
            "time": [datetime.now()],
            "trade_ok": [True],
            "reason": [""],
        }
    ).to_csv(log, index=False)

    # ドリフト用の合成CSV（close列）を用意（しきい値未設定なのでNo-Op）
    drift = tmp_path / "drift.csv"
    pd.DataFrame({"close": list(range(200))}).to_csv(drift, index=False)

    # しきい値を指定しない→ドリフトは完全No-Op（exit=0想定）
    r = subprocess.run(
        [
            sys.executable,
            "scripts/health_check.py",
            "--logs",
            str(log),
            "--days",
            "1",
            "--drift-source",
            str(drift),
            "--drift-column",
            "close",
            "--drift-ref-n",
            "100",
            "--drift-cur-n",
            "50",
            "--drift-bins",
            "10",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    data = json.loads(r.stdout)
    assert "ok" in data and data["ok"] is True

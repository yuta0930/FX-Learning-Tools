from __future__ import annotations
import os
from constants import LOG_DIR, ARTIFACTS_DIR, CONFIGS_DIR


def ensure_dirs() -> None:
    """Create minimal directory structure required for logging and artifacts."""
    for d in [LOG_DIR, ARTIFACTS_DIR, CONFIGS_DIR]:
        os.makedirs(d, exist_ok=True)
    # subdirs
    os.makedirs(os.path.join(ARTIFACTS_DIR, "calibration"), exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, "tca"), exist_ok=True)

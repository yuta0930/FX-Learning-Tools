"""
Provide backward-compatible exports expected by tests that import from `app` module
when both a top-level app.py (Streamlit app) and an `app/` package coexist.

Historically, tests do: `from app import _load_and_validate_baseline`.
Now that `app/` exists as a package, we re-export a thin wrapper that delegates
to utils.app_core to preserve the old import path.
"""

from typing import Any, Tuple
import numpy as _np

try:
	# Preferred: use the stable UI-free core implementation
	from utils.app_core import _load_and_validate_baseline as _core__load_and_validate_baseline  # type: ignore
except Exception:  # pragma: no cover
	_core__load_and_validate_baseline = None  # type: ignore


def _load_and_validate_baseline(*args: Any, **kwargs: Any) -> Tuple[float, _np.ndarray | None, list[str]]:
	"""Backward-compatible export for baseline loader used by tests.

	Falls back to a safe default if core is unavailable.
	Returns: (baseline_proba, baseline_probs_or_None, warnings)
	"""
	if _core__load_and_validate_baseline is not None:
		return _core__load_and_validate_baseline(*args, **kwargs)
	# Safe fallback: default baseline with no distribution
	return 0.5, None, ["core baseline loader unavailable: using defaults"]

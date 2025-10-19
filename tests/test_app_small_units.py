import numpy as np
from importlib import import_module


def _import_targets():
    core = import_module('utils.app_core')
    return core.compute_drift_score, core._load_and_validate_baseline, core.compute_final_theta_components


def test_compute_drift_score_basic():
    compute_drift_score, _, _ = _import_targets()
    dm = {"psi": 0.25, "kl": 0.1, "js": 0.05, "hellinger": 0.2}
    s = compute_drift_score(dm)
    assert 0.0 <= s <= 1.0


def test_compute_drift_score_caps_and_weights():
    compute_drift_score, _, _ = _import_targets()
    dm = {"psi": 1.0, "kl": 1.0, "js": 1.0, "hellinger": 2.0}
    s = compute_drift_score(dm, w={"psi":0.5,"kl":0.2,"js":0.2,"h":0.1}, caps={"psi":0.5,"kl":0.5,"js":0.5,"h":1.0})
    # すべて上限で頭打ち → 正規化後は 1.0
    assert 0.99 <= s <= 1.0


def test_load_and_validate_baseline_handles_missing(tmp_path):
    _, _load, _ = _import_targets()
    # 存在しないパスを渡す → 警告は返るが関数は落ちない
    base_p, probs, warns = _load(
        meta_path=str(tmp_path/"no_meta.json"),
        calib_path=str(tmp_path/"no_calib.json"),
    )
    assert 0.0 <= base_p <= 1.0
    assert probs is None or (isinstance(probs, np.ndarray) and probs.ndim==1)
    assert isinstance(warns, list)


def test_compute_final_theta_components_basic():
    _, _, comp = _import_targets()
    tf, br = comp(base_theta=0.6, session_theta=0.65, drift_bump=0.03, news_bump=0.02, theta_min=0.4, theta_max=0.85)
    assert 0.4 <= tf <= 0.85
    # session が base より高いのでそれを採用し、バンプを加算
    assert abs(tf - (0.65 + 0.03 + 0.02)) < 1e-9


def test_compute_final_theta_components_clip_min():
    _, _, comp = _import_targets()
    tf, br = comp(base_theta=0.45, session_theta=0.40, drift_bump=-0.1, news_bump=0.0, theta_min=0.4, theta_max=0.85)
    # base vs session の max は 0.45、そこに -0.1 → 0.35 を min=0.4 でクリップ
    assert abs(tf - 0.4) < 1e-9


def test_compute_final_theta_components_clip_max():
    _, _, comp = _import_targets()
    tf, br = comp(base_theta=0.8, session_theta=0.82, drift_bump=0.1, news_bump=0.1, theta_min=0.4, theta_max=0.85)
    # 0.82 + 0.2 = 1.02 を max=0.85 でクリップ
    assert abs(tf - 0.85) < 1e-9

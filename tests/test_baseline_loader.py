import json
import os
import numpy as np
from app import _load_and_validate_baseline


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def test_baseline_loader_happy(tmp_path):
    meta = tmp_path / 'models'
    rep = tmp_path / 'reports'
    meta.mkdir(); rep.mkdir()
    write_json(meta / 'break_meta.json', {"baseline_proba": 0.62})
    write_json(rep / 'break_calibration.json', {"prob_mean": [0.1, 0.3, 0.5, 0.7, 0.9]})
    b, arr, warns = _load_and_validate_baseline(str(meta / 'break_meta.json'), str(rep / 'break_calibration.json'))
    assert b == 0.62
    assert arr is not None and len(arr) == 5
    assert not warns


def test_baseline_loader_edge_cases(tmp_path):
    meta = tmp_path / 'models'
    rep = tmp_path / 'reports'
    meta.mkdir(); rep.mkdir()
    # meta missing baseline_proba, calibration has NaN & out-of-range values
    write_json(meta / 'break_meta.json', {})
    write_json(rep / 'break_calibration.json', {"prob_mean": [0.2, float('nan'), 1.2, -0.1]})
    b, arr, warns = _load_and_validate_baseline(str(meta / 'break_meta.json'), str(rep / 'break_calibration.json'))
    assert 0.0 <= b <= 1.0
    # After cleaning, arr should have only valid clipped values
    assert arr is not None
    assert all(0.0 <= x <= 1.0 for x in arr)
    assert any('NaN/inf' in w or 'クリップ' in w for w in warns)


def test_baseline_loader_insufficient_samples(tmp_path):
    meta = tmp_path / 'models'
    rep = tmp_path / 'reports'
    meta.mkdir(); rep.mkdir()
    write_json(meta / 'break_meta.json', {"baseline_proba": 0.4})
    write_json(rep / 'break_calibration.json', {"prob_mean": [0.5]})  # only 1 sample
    b, arr, warns = _load_and_validate_baseline(str(meta / 'break_meta.json'), str(rep / 'break_calibration.json'))
    assert arr is None  # insufficient -> None
    assert any('サンプル不足' in w for w in warns)

import pandas as pd
import numpy as np

# UI非依存の純粋関数を直接インポート
from utils.app_core import compute_final_theta_for_time_pure


def _now_jst():
    return pd.Timestamp.now(tz="Asia/Tokyo")


def _make_window_covering(ts, minutes=10, importance=5):
    return pd.DataFrame([
        {
            "start": ts - pd.Timedelta(minutes=minutes),
            "end": ts + pd.Timedelta(minutes=minutes),
            "importance": importance,
            "title": "test",
        }
    ])



def test_theta_final_applies_news_bump_when_soft_suppress_active():
    ts = _now_jst()
    windows_df = _make_window_covering(ts)

    settings = {
        'theta_min': 0.40,
        'theta_max': 0.90,
        'theta_base': 0.60,
        'theta_drift_bump_active': False,
        'use_soft_suppress': True,
        'theta_bump_in_news': 0.05,
        'news_win': 30,
        'news_imp_min': 3,
    'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.55}  # session_theta=0.55 < base

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    assert np.isclose(theta, 0.65, atol=1e-6)
    assert br['in_news'] is True
    assert np.isclose(br['news_bump'], 0.05, atol=1e-9)
    assert np.isclose(br['session_used'], 0.55, atol=1e-9)


def test_theta_final_uses_session_when_higher():
    ts = _now_jst()
    windows_df = pd.DataFrame(columns=["start","end","importance","title"])  # ニュースなし

    settings = {
        'theta_min': 0.40,
        'theta_max': 0.90,
        'theta_base': 0.60,
        'theta_drift_bump_active': False,
        'use_soft_suppress': False,
        'theta_bump_in_news': 0.03,
        'news_win': 30,
        'news_imp_min': 3,
    'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.80}  # session_theta=0.80 > base

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    assert np.isclose(theta, 0.80, atol=1e-6)
    assert np.isclose(br['session_bump'], 0.20, atol=1e-9)
    assert np.isclose(br['news_bump'], 0.0, atol=1e-9)
    assert br['in_news'] is False


def test_theta_final_clips_to_max_with_drift_bump():
    ts = _now_jst()
    windows_df = pd.DataFrame(columns=["start","end","importance","title"])  # ニュースなし

    settings = {
        'theta_min': 0.40,
        'theta_max': 0.90,
        'theta_base': 0.85,
        'theta_drift_bump_active': True,
        'theta_bump_drift': 0.10,
        'use_soft_suppress': False,
        'theta_bump_in_news': 0.02,
        'news_win': 30,
        'news_imp_min': 3,
    'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.90}  # session_theta=0.90, base_after_session=0.90

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    assert np.isclose(theta, 0.90, atol=1e-6)  # 0.90 + 0.10 -> 0.90 (clip)
    assert np.isclose(br['news_bump'], 0.0, atol=1e-9)
    assert np.isclose(br['drift_bump'], 0.10, atol=1e-9)
    assert br['in_news'] is False


def test_theta_final_clips_to_min():
    ts = _now_jst()
    windows_df = pd.DataFrame(columns=["start","end","importance","title"])  # ニュースなし

    settings = {
        'theta_min': 0.50,
        'theta_max': 0.90,
        'theta_base': 0.40,  # base below min
        'theta_drift_bump_active': False,
        'use_soft_suppress': False,
        'theta_bump_in_news': 0.00,
        'news_win': 30,
        'news_imp_min': 3,
        'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.45}  # session 0.45 < min 0.50

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    assert np.isclose(theta, 0.50, atol=1e-6)
    assert np.isclose(br['min'], 0.50, atol=1e-9)
    assert np.isclose(br['max'], 0.90, atol=1e-9)


def test_theta_min_greater_than_max_is_swapped():
    ts = _now_jst()
    windows_df = pd.DataFrame(columns=["start","end","importance","title"])  # ニュースなし

    settings = {
        'theta_min': 0.95,  # min > max
        'theta_max': 0.60,
        'theta_base': 0.80,
        'theta_drift_bump_active': False,
        'use_soft_suppress': False,
        'theta_bump_in_news': 0.00,
        'news_win': 30,
        'news_imp_min': 3,
        'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.70}

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    # min/max が入れ替わり min=0.60, max=0.95 となること
    assert np.isclose(br['min'], 0.60, atol=1e-9)
    assert np.isclose(br['max'], 0.95, atol=1e-9)
    # base/session=0.80 は範囲内なのでそのまま
    assert np.isclose(theta, 0.80, atol=1e-6)


def test_theta_final_handles_NaT_timestamp_and_none_windows():
    ts = pd.NaT  # NaT を渡しても落ちずに処理できること
    windows_df = None

    settings = {
        'theta_min': 0.40,
        'theta_max': 0.90,
        'theta_base': 0.50,
        'theta_drift_bump_active': False,
        'use_soft_suppress': True,
        'theta_bump_in_news': 0.05,
        'news_win': 30,
        'news_imp_min': 3,
        'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {}  # threshold 未指定 -> デフォルト 0.93 が使われる

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    # base=0.50 と session(=0.93) の高い方 0.93 が採用され、ニュースなし/バンプなし
    assert np.isclose(theta, 0.90, atol=1e-6)  # 0.93 は max=0.90 でクリップ
    assert br['in_news'] is False
    assert np.isclose(br['session_used'], 0.93, atol=1e-9)


def test_theta_final_respects_negative_drift_bump_but_clips_min():
    ts = _now_jst()
    windows_df = pd.DataFrame(columns=["start","end","importance","title"])  # ニュースなし

    settings = {
        'theta_min': 0.40,
        'theta_max': 0.90,
        'theta_base': 0.50,
        'theta_drift_bump_active': True,
        'theta_bump_drift': -0.20,  # 負のバンプ
        'use_soft_suppress': False,
        'theta_bump_in_news': 0.00,
        'news_win': 30,
        'news_imp_min': 3,
        'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.55}  # session=0.55 -> base_after_session=0.55

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    # 0.55 + (-0.20) = 0.35 -> min=0.40 でクリップ
    assert np.isclose(theta, 0.40, atol=1e-6)
    assert np.isclose(br['drift_bump'], -0.20, atol=1e-9)


def test_theta_final_ignores_malformed_windows():
    ts = _now_jst()
    # start/end カラムが欠けていてもエラーにせず False 判定
    windows_df = pd.DataFrame({"begin": [ts], "finish": [ts]})

    settings = {
        'theta_min': 0.40,
        'theta_max': 0.90,
        'theta_base': 0.60,
        'theta_drift_bump_active': False,
        'use_soft_suppress': True,
        'theta_bump_in_news': 0.05,
        'news_win': 30,
        'news_imp_min': 3,
        'news_filter_mode': "重要度別（赤影と同じ）",
    }

    meta = {"threshold": 0.50}

    theta, br = compute_final_theta_for_time_pure(ts, meta, windows_df, settings)
    # ニュースは無視され bump 0、session/baseの高い方 0.60 が採用
    assert np.isclose(theta, 0.60, atol=1e-6)
    assert br['in_news'] is False

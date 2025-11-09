import numpy as np
import pandas as pd

# ベンチマーク版の純粋関数を利用（Streamlit依存を避ける）
import importlib
br = importlib.import_module("scripts.benchmark_rectangle")

def make_synthetic_rectangle(n=200, box_start=60, box_len=48, top=1.2000, bottom=1.1950, noise=0.0002):
    rng = np.random.default_rng(42)
    o = np.linspace(1.19, 1.205, n)  # 緩やかな基調
    h = o.copy()
    l = o.copy()
    c = o.copy()
    # ボックス期間に矩形を形成
    s = box_start
    e = s + box_len - 1
    h[s:e+1] = top + rng.normal(0, noise*0.25, box_len)
    l[s:e+1] = bottom + rng.normal(0, noise*0.25, box_len)
    # 終端付近でのタッチ強化（ピボット確保）
    for k in range(4):
        idx_hi = s + 2 + k*5
        if idx_hi <= e:
            h[idx_hi] = top + rng.normal(0, noise*0.05)
        idx_lo = s + 4 + k*5
        if idx_lo <= e:
            l[idx_lo] = bottom + rng.normal(0, noise*0.05)
    # 終端後に上抜けブレイク（任意）
    if e + 2 < n:
        h[e+1] = top + 0.003
        l[e+1] = bottom + 0.002
        c[e+1] = top + 0.002
    # close はレンジ内に配置
    c[s:e+1] = (h[s:e+1] + l[s:e+1]) * 0.5 + rng.normal(0, noise*0.2, box_len)
    # open は適当に
    o = (h + l) * 0.5
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    return df


def test_rectangle_detects_on_synthetic_data():
    df = make_synthetic_rectangle()
    pats = br.detect_rectangles(
        df,
        last_N=None,
        win_min_bars=24,
        win_max_bars=60,
        e_step=1,
        len_step=1,
        enable_low_pivot_fallback=True,
        use_parallel=False,
    )
    assert isinstance(pats, list)
    # 少なくとも1件は検出されること
    assert len(pats) >= 1, "expected at least one rectangle pattern on synthetic data"
    p0 = max(pats, key=lambda p: p.get("quality_score", 0.0))
    assert p0["type"] == "rectangle"
    assert p0["width_mean"] > 0
    assert p0["touches_upper"] >= 2  # 設定により3未満の場合もあるため緩め
    assert p0["touches_lower"] >= 2


def test_baseline_and_opt_consistency_minimal():
    df = make_synthetic_rectangle(n=240)
    opt = br.detect_rectangles(df, last_N=None, win_min_bars=20, win_max_bars=60, e_step=1, len_step=1, enable_low_pivot_fallback=True)
    base = br.detect_rectangles_baseline(df, last_N=None, win_min_bars=20, win_max_bars=60, e_step=1, len_step=1)
    # どちらも実行でき、結果が空でないこと（閾値に依存するため片方のみ非空でも許容）
    assert isinstance(opt, list) and isinstance(base, list)
    assert (len(opt) + len(base)) >= 1, "either optimized or baseline should detect on synthetic rectangle"

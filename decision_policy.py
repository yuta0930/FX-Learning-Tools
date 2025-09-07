# decision_policy.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import math

# 単位RベースのEV。学習メタに合わせてR_win=R_loss=1.0前提（必要ならUIで変更）
@dataclass
class EVConfig:
    R_win: float = 1.0
    R_loss: float = 1.0
    cost_per_trade: float = 0.02  # メタのコストと整合させる

@dataclass
class DecisionParams:
    # --- ゲート ---
    min_ev_r: float = 0.0                   # EV/Rの下限（>=0を推奨）
    theta_bump_by_regime: Dict[str, float] = None  # {"low":+0.00,"mid":+0.02,"high":+0.03} 等
    theta_bump_in_news: float = 0.03        # ニュース時の上乗せ（softモード時）
    news_mode: str = "hard"                 # "hard" = 無条件見送り, "soft" = θを上げる
    spread_max: float = 0.03                # 許容最大スプレッド（pips表記のときはアプリ側で同単位に揃える）
    wick_ratio_max: float = 2.5             # 長ヒゲ: ヒゲ/実体 > この閾値なら成行を抑制
    prefer_limit_retest: bool = True        # 原則リテスト指値を優先
    # --- 表示 ---
    rationale_verbose: bool = True

def _ev_r(p: float, ev: EVConfig) -> float:
    # 単位R（勝ち=+1, 負け=-1）での期待値
    return p*ev.R_win - (1.0 - p)*ev.R_loss - ev.cost_per_trade

def _wick_ratio(o: float, h: float, l: float, c: float) -> Tuple[float, float]:
    body = max(abs(c - o), 1e-9)
    up = max(h - max(o, c), 0.0)
    dn = max(min(o, c) - l, 0.0)
    return (up/body, dn/body)

def _pick_regime_from_feats(feat_row: dict) -> Optional[str]:
    # featxで作ったreg_atr_*を優先使用（学習列に含まれている想定）:contentReference[oaicite:0]{index=0}
    for lab in ("low","mid","high"):
        if feat_row.get(f"reg_atr_{lab}", 0.0) > 0.5:
            return lab
    return None

def effective_theta(theta: float, regime: Optional[str], in_news: bool, params: DecisionParams) -> float:
    th = float(theta)
    if params.theta_bump_by_regime and regime in params.theta_bump_by_regime:
        th += float(params.theta_bump_by_regime[regime])
    if in_news and params.news_mode == "soft":
        th += float(params.theta_bump_in_news)
    return min(max(th, 0.50), 0.99)

def recommend_action(
    *,
    p: float,
    theta: float,
    session: str,
    feat_row: dict,
    ohlc: Tuple[float,float,float,float],
    in_news: bool,
    spread: float,
    ev_cfg: EVConfig,
    params: DecisionParams
) -> Tuple[str, List[str], float, float]:
    """
    return: (action, reasons, theta_eff, ev_r)
    action in {"見送り","監視","指値(リテスト)","成行(小)"}
    """
    reasons: List[str] = []
    # 0) ハードニュース: 無条件見送り（UIで切替） / ソフトは後段でθを上げる
    if in_news and params.news_mode == "hard":
        return "見送り", ["重要イベント近傍（hard）"], theta, float("nan")

    # 1) コスト/スプレッド
    if spread is not None and spread > params.spread_max:
        return "見送り", [f"スプレッド {spread:.3f} > {params.spread_max:.3f}"], theta, float("nan")

    # 2) θとEVのゲート
    regime = _pick_regime_from_feats(feat_row)
    th_eff = effective_theta(theta, regime, in_news, params)
    if p < th_eff:
        return "見送り", [f"p={p:.3f} < θ_eff={th_eff:.3f}"], th_eff, float("nan")

    evr = _ev_r(p, ev_cfg)
    if evr < params.min_ev_r:
        return "見送り", [f"EV/R={evr:.3f} < {params.min_ev_r:.3f}"], th_eff, evr

    # 3) 長ヒゲ直後は成行抑制 → 指値/監視へ誘導（“入らない勇気”）
    o,h,l,c = ohlc
    wr_up, wr_dn = _wick_ratio(o,h,l,c)
    if max(wr_up, wr_dn) >= params.wick_ratio_max:
        reasons.append(f"長ヒゲ {max(wr_up,wr_dn):.1f}x（直後）")

    # 4) 最終推奨
    if params.prefer_limit_retest or reasons:
        return "指値(リテスト)", reasons + ["リテスト優先"], th_eff, evr
    else:
        return "成行(小)", reasons, th_eff, evr

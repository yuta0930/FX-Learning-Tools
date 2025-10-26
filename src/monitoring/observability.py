from __future__ import annotations

from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Iterable, List, Optional
import json
import hashlib
import subprocess

import pandas as pd

from src.policy.reasons import unify_reasons_df
from src.policy.reasons_normalize import load_reason_map, normalize_reasons_df


def _git_commit_short() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _sha256_of(p: Path) -> Optional[str]:
    try:
        if not p.exists():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _read_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        suf = path.suffix.lower()
        if suf == ".parquet":
            return pd.read_parquet(path)
        elif suf == ".jsonl":
            return pd.read_json(path, lines=True)
        elif suf == ".csv":
            return pd.read_csv(path)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_logs(paths: Iterable[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        df = _read_one(p)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True, sort=False)

    # 統一：理由列・時刻・日付
    df = unify_reasons_df(df)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["day"] = df["time"].dt.date
    else:
        df["day"] = pd.NaT

    return df


def daily_allow_block(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "allow", "block", "total", "block_rate"])

    g = df.groupby("day", dropna=True)
    allow = g.apply(lambda x: x.get("trade_ok", pd.Series(False, index=x.index)).astype(bool).sum())
    block = g.apply(lambda x: (x.get("trade_ok", pd.Series(False, index=x.index)).astype(bool) == False).sum())  # noqa: E712
    out = pd.DataFrame({"allow": allow, "block": block}).reset_index()
    out["total"] = (out["allow"] + out["block"]).astype(int)
    out["block_rate"] = out.apply(lambda r: (r["block"] / r["total"]) if r["total"] else 0.0, axis=1)
    out = out.sort_values("day")
    return out


def top_reasons(
    df: pd.DataFrame,
    top_n: int = 10,
    since: Optional[datetime] = None,
    *,
    normalized: bool = False,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="int64")

    s = df
    if since is not None and "time" in s.columns:
        s = s[pd.to_datetime(s["time"], errors="coerce") >= since]

    col = None
    if normalized and "reason_norm" in s.columns:
        col = "reason_norm"
    if col is None:
        col = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in s.columns), None)
    if col is None:
        return pd.Series(dtype="int64")

    ser = (
        s[col]
        .astype(str)
        .replace({"": None, "nan": None})
        .dropna()
        .value_counts()
        .head(top_n)
    )
    return ser


def export_monthly_report(
    df: pd.DataFrame,
    *,
    as_of: Optional[datetime] = None,
    out_root: Path = Path("reports"),
    top_n: int = 50,
    sources: Optional[List[Path]] = None,
    reasons_map_path: Optional[Path] = None,
) -> Path:
    """
    月次ディレクトリ reports/monthly_YYYYMM/ に日次集計と理由TopをCSV保存。
    df が空なら空レポートを出すが、ファイル作成は行う（No-Opに近い安全動作）。
    """
    as_of = as_of or datetime.now()
    ym = as_of.strftime("%Y%m")
    out_dir = out_root / f"monthly_{ym}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日次集計
    daily = (
        daily_allow_block(df)
        if not df.empty
        else pd.DataFrame(columns=["day", "allow", "block", "total", "block_rate"])
    )
    daily_path = out_dir / "daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8")

    # 理由Top
    reasons = top_reasons(df, top_n=top_n)
    reasons_path = out_dir / "top_reasons.csv"
    # 理由Top（normalized）— マップがある場合のみ
    norm_path = None
    unk_path = None
    if reasons_map_path:
        mapping = load_reason_map(reasons_map_path)
        if mapping:
            df_norm = normalize_reasons_df(df, mapping=mapping)
            norm = top_reasons(df_norm, top_n=top_n, normalized=True)
            norm_path = out_dir / "top_reasons_normalized.csv"
            if norm.empty:
                pd.DataFrame(columns=["reason", "count"]).to_csv(norm_path, index=False, encoding="utf-8")
            else:
                norm.to_frame("count").to_csv(norm_path, index=True, header=True, encoding="utf-8")
            # 未分類理由抽出（normalizedと元理由が同じ=未分類とみなす）
            try:
                if "reason_norm" in df_norm.columns:
                    src_col = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in df_norm.columns), "reason_norm")
                    unk_mask = (df_norm["reason_norm"] == df_norm[src_col]) & df_norm[src_col].astype(str).ne("")
                    unk = df_norm.loc[unk_mask, [src_col]].rename(columns={src_col: "reason"})
                    if isinstance(unk, pd.DataFrame) and not unk.empty:
                        unk_path = out_dir / "unknown_reasons.csv"
                        unk["count"] = 1
                        unk = unk.groupby("reason", as_index=False)["count"].sum().sort_values("count", ascending=False)
                        unk.to_csv(unk_path, index=False, encoding="utf-8")
            except Exception:
                pass
    if reasons.empty:
        pd.DataFrame(columns=["reason", "count"]).to_csv(reasons_path, index=False, encoding="utf-8")
    else:
        reasons.to_frame("count").to_csv(reasons_path, index=True, header=True, encoding="utf-8")

    # メタデータ（生成時刻/対象期間/ログソース）
    time_min = None
    time_max = None
    if (not df.empty) and ("time" in df.columns):
        ts = pd.to_datetime(df["time"], errors="coerce")
        if not ts.isna().all():
            tmin = ts.min()
            tmax = ts.max()
            time_min = tmin.isoformat() if pd.notna(tmin) else None
            time_max = tmax.isoformat() if pd.notna(tmax) else None

    cfg_candidates = [Path("config/config.yml"), Path("config/events.yml"), Path("config/reasons_map.yml")]
    cfg_hashes = {str(p): _sha256_of(p) for p in cfg_candidates if p.exists()}

    meta = {
        "generated_at": as_of.isoformat(),
        "year_month": ym,
        "time_range": {
            "min": time_min,
            "max": time_max,
        },
        "rows": int(len(df)),
        "sources": [str(p) for p in (sources or [])],
        "git_commit": _git_commit_short(),
        "config_hashes": cfg_hashes,
        "artifacts": {
            "daily_csv": str(daily_path),
            "top_reasons_csv": str(reasons_path),
            **({"top_reasons_normalized_csv": str(norm_path)} if norm_path else {}),
            **({"unknown_reasons_csv": str(unk_path)} if unk_path else {}),
        },
        "notes": "Empty inputs produce empty CSVs with headers; safe No-Op by design.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_dir


def _filter_by_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """naiveなtime列を start/end（ともにdate, 両端含む）でフィルタ。time列が無い場合は空DFを返す。"""
    if df.empty or "time" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    start_dt = datetime.combine(start, time.min)
    end_dt = datetime.combine(end, time.max)
    m = (d["time"] >= start_dt) & (d["time"] <= end_dt)
    return d.loc[m]


def export_periodic_report(
    df: pd.DataFrame,
    *,
    start: date,
    end: date,
    out_root: Path = Path("reports"),
    top_n: int = 50,
    sources: Optional[List[Path]] = None,
    reasons_map_path: Optional[Path] = None,
) -> Path:
    """
    開始・終了日で区間固定したレポートを reports/periodic_YYYYMMDD_YYYYMMDD/ にCSV+metadata出力。
    time列が無ければ空CSVを出す安全動作。
    """
    dfp = _filter_by_date_range(df, start, end)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    out_dir = out_root / f"periodic_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = (
        daily_allow_block(dfp)
        if not dfp.empty
        else pd.DataFrame(columns=["day", "allow", "block", "total", "block_rate"])
    )
    daily_path = out_dir / "daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8")

    reasons = top_reasons(dfp, top_n=top_n)
    reasons_path = out_dir / "top_reasons.csv"
    # normalized（あれば）
    norm_path = None
    unk_path = None
    if reasons_map_path:
        mapping = load_reason_map(reasons_map_path)
        if mapping:
            dfp_norm = normalize_reasons_df(dfp, mapping=mapping)
            norm = top_reasons(dfp_norm, top_n=top_n, normalized=True)
            norm_path = out_dir / "top_reasons_normalized.csv"
            if norm.empty:
                pd.DataFrame(columns=["reason", "count"]).to_csv(norm_path, index=False, encoding="utf-8")
            else:
                norm.to_frame("count").to_csv(norm_path, index=True, header=True, encoding="utf-8")
            # 未分類理由抽出
            try:
                if "reason_norm" in dfp_norm.columns:
                    src_col = next((c for c in ["reason", "deny_reason", "gate_reason"] if c in dfp_norm.columns), "reason_norm")
                    unk_mask = (dfp_norm["reason_norm"] == dfp_norm[src_col]) & dfp_norm[src_col].astype(str).ne("")
                    unk = dfp_norm.loc[unk_mask, [src_col]].rename(columns={src_col: "reason"})
                    if isinstance(unk, pd.DataFrame) and not unk.empty:
                        unk_path = out_dir / "unknown_reasons.csv"
                        unk["count"] = 1
                        unk = unk.groupby("reason", as_index=False)["count"].sum().sort_values("count", ascending=False)
                        unk.to_csv(unk_path, index=False, encoding="utf-8")
            except Exception:
                pass
    if reasons.empty:
        pd.DataFrame(columns=["reason", "count"]).to_csv(reasons_path, index=False, encoding="utf-8")
    else:
        reasons.to_frame("count").to_csv(reasons_path, index=True, header=True, encoding="utf-8")

    cfg_candidates = [Path("config/config.yml"), Path("config/events.yml"), Path("config/reasons_map.yml")]
    cfg_hashes = {str(p): _sha256_of(p) for p in cfg_candidates if p.exists()}

    meta = {
        "generated_at": datetime.now().isoformat(),
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "rows": int(len(dfp)),
        "sources": [str(p) for p in (sources or [])],
        "git_commit": _git_commit_short(),
        "config_hashes": cfg_hashes,
        "artifacts": {
            "daily_csv": str(daily_path),
            "top_reasons_csv": str(reasons_path),
            **({"top_reasons_normalized_csv": str(norm_path)} if norm_path else {}),
            **({"unknown_reasons_csv": str(unk_path)} if unk_path else {}),
        },
        "notes": "Empty inputs produce empty CSVs with headers; safe No-Op by design.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_dir


def export_this_week(
    df: pd.DataFrame,
    *,
    out_root: Path = Path("reports"),
    sources: Optional[List[Path]] = None,
    reasons_map_path: Optional[Path] = None,
) -> Path:
    """今週（月曜〜日曜, ローカルカレンダー基準）のperiodicレポートを出力。"""
    today = datetime.now().date()
    start = today - timedelta(days=today.weekday())  # Monday
    end = start + timedelta(days=6)  # Sunday
    return export_periodic_report(
        df, start=start, end=end, out_root=out_root, sources=sources, reasons_map_path=reasons_map_path
    )


def export_last_week(
    df: pd.DataFrame,
    *,
    out_root: Path = Path("reports"),
    sources: Optional[List[Path]] = None,
    reasons_map_path: Optional[Path] = None,
) -> Path:
    """先週（月曜〜日曜）のperiodicレポートを出力。"""
    today = datetime.now().date()
    start = today - timedelta(days=today.weekday() + 7)
    end = start + timedelta(days=6)
    return export_periodic_report(
        df, start=start, end=end, out_root=out_root, sources=sources, reasons_map_path=reasons_map_path
    )

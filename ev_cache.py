"""Incremental / cached EV (expected pips) calculations.

Strategy (Phase 1):
  - Cache results of compute_expected_pips_table_for_levels keyed by a hash of
    (len(df), last_index, fwd_n, break_buffer, spread_pips, news_win, news_imp_min,
     apply_news_filter, touch_buffer, retest_wait_k)
  - If only len(df) increased by 1 (new bar) and parameters unchanged, perform
    an incremental update by re-running the full function for now (Phase 1 keeps it simple),
    but in future we can diff only last window.

Provides:
  get_ev_tables(df, levels, params_dict, compute_fn) -> (ev_table, ev_dir, meta)

meta dict includes timing info and cache hit flag.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Tuple
import time, hashlib
import pandas as pd

@dataclass
class EVCacheEntry:
    key: str
    len_df: int
    last_ts: Any
    ev_table: pd.DataFrame | None
    ev_dir: pd.DataFrame | None
    created: float
    params_sig: str

class EVCache:
    def __init__(self, max_entries: int = 32):
        self.store: dict[str, EVCacheEntry] = {}
        self.max_entries = max_entries

    def _make_key(self, sig: str) -> str:
        return hashlib.sha1(sig.encode('utf-8')).hexdigest()

    def _evict_if_needed(self):
        if len(self.store) <= self.max_entries:
            return
        # evict oldest
        oldest = sorted(self.store.values(), key=lambda e: e.created)[0]
        self.store.pop(oldest.key, None)

    def get(self, sig: str):
        k = self._make_key(sig)
        return self.store.get(k)

    def put(self, sig: str, len_df: int, last_ts, ev_table, ev_dir, params_sig: str):
        k = self._make_key(sig)
        self.store[k] = EVCacheEntry(k, len_df, last_ts, ev_table, ev_dir, time.time(), params_sig)
        self._evict_if_needed()

_global_ev_cache = EVCache()

def get_ev_tables(df: pd.DataFrame,
                  levels,
                  params: dict,
                  compute_fn: Callable[..., Tuple[pd.DataFrame|None, pd.DataFrame|None]],
                  *,
                  min_recompute_interval: float = 3.0,
                  max_bar_lookback_skip: int = 1):
    """Obtain EV tables with caching.

    params must include all function inputs (except df & levels) to build signature.
    """
    # Build signature
    last_ts = df.index[-1] if len(df) else None
    parts = [
        str(len(df)), str(last_ts)
    ] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    sig = '|'.join(parts)

    start = time.perf_counter()
    entry = _global_ev_cache.get(sig)
    # Case 1: perfect match (no new bars / no param change)
    if entry is not None and entry.len_df == len(df) and entry.last_ts == last_ts:
        return entry.ev_table, entry.ev_dir, {"cache_hit": True, "elapsed": 0.0, "skipped": False}

    # Case 2: small bar growth & short interval -> skip recompute, reuse previous if params identical except len/ts
    if entry is not None:
        time_since = time.perf_counter() - entry.created
        bar_diff = len(df) - entry.len_df
        if bar_diff <= max_bar_lookback_skip and time_since < min_recompute_interval:
            return entry.ev_table, entry.ev_dir, {"cache_hit": True, "elapsed": 0.0, "skipped": True, "reason": f"bar_diff={bar_diff},dt={time_since:.2f}s"}

    # Compute fresh
    ev_table, ev_dir = compute_fn(
        df,
        levels,
        params['fwd_n'],
        params['break_buffer'],
        params['spread_pips'],
        params['news_df'],
        params['news_win'],
        params['news_imp_min'],
        params['apply_news_filter'],
        params['touch_buffer'],
        params['retest_wait_k'],
    )
    elapsed = time.perf_counter() - start
    _global_ev_cache.put(sig, len(df), last_ts, ev_table, ev_dir, sig)
    return ev_table, ev_dir, {"cache_hit": False, "elapsed": elapsed, "skipped": False}

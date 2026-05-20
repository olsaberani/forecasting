#!/usr/bin/env python3
"""
collect_google_trends.py
========================
Download Google Trends weekly/monthly interest data for 6 unemployment-related
search terms across all 51 US states (50 states + DC).

Granularity note
----------------
pytrends returns MONTHLY data for time ranges > ~5 years.  Our TIMEFRAME
is 15 years, so each (state, term) pair yields ~180 monthly rows.  Since the
next pipeline step aggregates to monthly anyway, this is exactly what we need.
We record every row with its date as returned by Google ("week_start" column
is named for consistency even when the data is monthly).

Usage
-----
    uv run python src/collect_google_trends.py            # full run
    uv run python src/collect_google_trends.py --dry-run  # preview without API calls

Restart safety
--------------
Each (state, term) pair is cached as a separate CSV.  If you Ctrl+C and
re-run, already-cached pairs are skipped automatically.  Permanently-failed
pairs are written to data/raw/google_trends/_FAILED.txt and are NOT cached,
so they will be retried on the next run.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

# ── Configuration — edit these if needed ──────────────────────────────────────

TERMS: list[str] = [
    "unemployment",
    "file for unemployment",
    "unemployment benefits",
    "jobs hiring",
    "layoffs",
    "resume",
]

STATES: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

TIMEFRAME           = "2010-01-01 2024-12-31"
SLEEP_BASE          = 10    # seconds to sleep after every successful fetch
SLEEP_ON_ERROR_BASE = 2   # initial backoff (doubles each retry)
MAX_RETRIES         = 2     # retries per pair before writing to _FAILED.txt

# Paths are resolved relative to this file's parent (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR     = _PROJECT_ROOT / "data" / "raw" / "google_trends"
FAILED_FILE   = CACHE_DIR / "_FAILED.txt"

# ── Helpers ───────────────────────────────────────────────────────────────────

def term_slug(term: str) -> str:
    """'file for unemployment' → 'file_for_unemployment'"""
    return term.replace(" ", "_")


def cache_path(state_code: str, term: str) -> Path:
    return CACHE_DIR / f"{state_code}__{term_slug(term)}.csv"


def is_cached(state_code: str, term: str) -> bool:
    """Return True if a non-empty CSV already exists for this pair."""
    p = cache_path(state_code, term)
    return p.exists() and p.stat().st_size > 100  # >100 bytes rules out empty/header-only


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_client() -> TrendReq:
    """
    Create a fresh TrendReq.

    IMPORTANT: retries=0, backoff_factor=0 — this avoids a crash in pytrends
    4.9.x caused by urllib3 v2 removing the 'method_whitelist' parameter from
    Retry.__init__().  We implement our own retry/backoff logic below.
    """
    return TrendReq(
        hl="en-US",
        tz=360,
        timeout=(10, 60),
        retries=0,
        backoff_factor=0,
    )


def fetch_one(client: TrendReq, state_code: str, term: str) -> pd.DataFrame:
    """
    Fetch interest_over_time for one (state, term) pair.

    Returns a tidy DataFrame:
        week_start  |  term  |  interest  |  state_code

    Raises on any HTTP/parse error so the caller can do backoff + retry.
    """
    client.build_payload(
        kw_list=[term],
        timeframe=TIMEFRAME,
        geo=f"US-{state_code}",
    )
    raw = client.interest_over_time()

    if raw is None or raw.empty:
        # Legitimate: some state/term combos have zero recorded searches.
        return pd.DataFrame(columns=["week_start", "term", "interest", "state_code"])

    df = (
        raw.reset_index()
           .rename(columns={"date": "week_start", term: "interest"})
           [["week_start", "interest"]]
    )
    df["interest"]   = pd.to_numeric(df["interest"], errors="coerce")
    df["term"]       = term
    df["state_code"] = state_code

    return df[["week_start", "term", "interest", "state_code"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Google Trends data for unemployment terms × US states"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without calling the API",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build full pair list: state varies slowest so we exhaust all terms per
    # state before moving on — friendlier to Google's per-geo rate limits.
    all_pairs   = [(state, term) for state in STATES for term in TERMS]
    cached      = [p for p in all_pairs if is_cached(*p)]
    to_fetch    = [p for p in all_pairs if not is_cached(*p)]
    total       = len(all_pairs)

    print(f"{'='*60}")
    print(f"Google Trends collector — {now()}")
    print(f"{'='*60}")
    print(f"Total (state, term) pairs : {total}  ({len(STATES)} states × {len(TERMS)} terms)")
    print(f"Already cached            : {len(cached)}")
    print(f"To fetch                  : {len(to_fetch)}")
    if to_fetch:
        est_min = len(to_fetch) * SLEEP_BASE / 60
        print(f"Estimated time            : {est_min:.0f} min ({est_min/60:.1f} h) at {SLEEP_BASE}s/call")
    print(f"Cache directory           : {CACHE_DIR}")
    print()

    # ── Dry-run mode ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("-- DRY RUN: pairs that would be fetched --")
        for i, (s, t) in enumerate(to_fetch, 1):
            print(f"  [{i:3d}/{len(to_fetch):3d}] {s}__{term_slug(t)}")
        if not to_fetch:
            print("  (nothing — all pairs are cached)")
        return

    if not to_fetch:
        print("Nothing to fetch. All pairs are already cached.")
        return

    # ── Live fetch ────────────────────────────────────────────────────────────
    client = new_client()

    succeeded    = 0
    failed_pairs: list[tuple[str, str]] = []

    for idx, (state, term) in enumerate(to_fetch, 1):
        slug  = f"{state}__{term_slug(term)}"
        # Check again in case a previous run cached it mid-loop
        if is_cached(state, term):
            n = len(pd.read_csv(cache_path(state, term)))
            print(f"[{now()}] [{idx:3d}/{len(to_fetch):3d}] {slug:<35} OK (cached, {n} rows)")
            succeeded += 1
            continue

        last_exc = None

        for retry in range(MAX_RETRIES + 1):
            try:
                df = fetch_one(client, state, term)

                p = cache_path(state, term)
                df.to_csv(p, index=False)

                tag = f"OK ({len(df)} rows)" if not df.empty else "EMPTY (0 rows — saved placeholder)"
                print(f"[{now()}] [{idx:3d}/{len(to_fetch):3d}] {slug:<35} {tag}")

                succeeded += 1
                last_exc = None
                break  # success — exit retry loop

            except Exception as exc:
                last_exc = exc
                if retry < MAX_RETRIES:
                    wait = SLEEP_ON_ERROR_BASE * (2 ** retry)
                    print(
                        f"[{now()}] [{idx:3d}/{len(to_fetch):3d}] {slug:<35} "
                        f"FAILED ({str(exc):.80}), retry {retry+1}/{MAX_RETRIES} in {wait}s"
                    )
                    time.sleep(wait)
                    client = new_client()   # fresh cookie after each error
                else:
                    print(
                        f"[{now()}] [{idx:3d}/{len(to_fetch):3d}] {slug:<35} "
                        f"PERMANENTLY FAILED after {MAX_RETRIES} retries: {str(exc):.120}"
                    )

        if last_exc is not None:
            # Log permanent failure; do NOT cache so the next run retries it
            failed_pairs.append((state, term))
            with open(FAILED_FILE, "a") as fh:
                fh.write(f"{state}\t{term}\n")
        else:
            # Sleep only after success (backoff sleeps already counted for errors)
            if idx < len(to_fetch):
                time.sleep(SLEEP_BASE)

        # Progress heartbeat every 10 pairs
        if idx % 10 == 0:
            pct = idx / len(to_fetch) * 100
            print(
                f"\n  ── Progress: {idx}/{len(to_fetch)} ({pct:.0f}%)  "
                f"succeeded={succeeded}  failed={len(failed_pairs)} ──\n"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"COLLECTION COMPLETE — {now()}")
    print(f"  Succeeded : {succeeded} / {len(to_fetch)}")
    print(f"  Failed    : {len(failed_pairs)}")
    if failed_pairs:
        print(f"\n  Permanently failed pairs (also in {FAILED_FILE}):")
        for s, t in failed_pairs:
            print(f"    {s}__{term_slug(t)}")
        print("\n  Re-run the script to retry them (not cached, so will be attempted again).")
    print("=" * 60)


if __name__ == "__main__":
    main()

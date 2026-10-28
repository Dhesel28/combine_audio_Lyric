"""
Fetch all 2026 Chartmetric TikTok charts (sounds + tracks), one week at a time.
Deduplicates, merges with existing cm_training_songs.csv, fetches Genius lyrics
for any songs that don't have them yet.

Usage:
    python fetch_cm_2026.py              # fetch charts + lyrics
    python fetch_cm_2026.py --no-lyrics  # charts only, skip Genius
    python fetch_cm_2026.py --delay 1.0  # slower Genius calls if rate-limited
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from chartmetric_fetcher import fetch_tiktok_sounds, fetch_tiktok_tracks
from genius_fetcher import fetch_lyrics, _make_client
from config import GENIUS_TOKEN, CM_TRAINING_CSV

_DATA_DIR = Path(__file__).parent / "data"

_SUFFIX_RE = re.compile(
    r"\s*[-–]\s*"
    r"(sped[- ]?up|speed[- ]?up|slowed(?:\s*\+?\s*reverb)?|ultra\s*slowed|"
    r"slowed\s*version|nightcore|reverb only|lo[- ]?fi|phonk|drill|edit|"
    r"acoustic|live|extended|official)\s*$",
    re.IGNORECASE,
)
_FEAT_RE  = re.compile(r"\s*\(feat\..*?\)", re.IGNORECASE)
_PAREN_RE = re.compile(
    r"\s*\((official\s*(audio|video|lyric[s]?|music\s*video|mv)?|"
    r"audio|lyric[s]?|music\s*video|mv)\s*\)",
    re.IGNORECASE,
)


def clean_title(title: str) -> str:
    t = _SUFFIX_RE.sub("", title)
    t = _FEAT_RE.sub("", t)
    t = _PAREN_RE.sub("", t)
    return t.strip()


def _mondays_2026() -> list[str]:
    start = date(2026, 1, 5)   # first Monday of 2026
    today = date.today()
    weeks = []
    d = start
    while d <= today:
        weeks.append(str(d))
        d += timedelta(weeks=1)
    return weeks


def fetch_all_charts() -> pd.DataFrame:
    weeks = _mondays_2026()
    print(f"[Charts] Fetching {len(weeks)} weeks × 2 charts (sounds + tracks) ...")

    all_rows: list[pd.DataFrame] = []

    for i, week in enumerate(weeks, 1):
        print(f"\n  Week {i}/{len(weeks)} — {week}")
        sounds = fetch_tiktok_sounds(limit=100, date=week)
        tracks = fetch_tiktok_tracks(limit=100, date=week)

        for df in (sounds, tracks):
            if not df.empty:
                df["week"] = week
                all_rows.append(df)

        time.sleep(2)   # be polite to the API

    if not all_rows:
        print("[Charts] No data returned.")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\n[Charts] Total rows before dedup: {len(combined)}")

    # Deduplicate: keep row with highest video_count per (song, artist)
    combined["song_key"]   = combined["song"].str.strip().str.lower()
    combined["artist_key"] = combined["artist"].str.strip().str.lower()
    combined = (
        combined
        .sort_values("video_count", ascending=False)
        .drop_duplicates(subset=["song_key", "artist_key"])
        .drop(columns=["song_key", "artist_key"])
        .reset_index(drop=True)
    )
    print(f"[Charts] Unique songs after dedup: {len(combined)}")
    return combined


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new chart data with existing cm_training_songs.csv, preserving lyrics."""
    if CM_TRAINING_CSV.exists():
        existing = pd.read_csv(CM_TRAINING_CSV)
        print(f"[Merge] Existing: {len(existing)} songs ({existing['lyrics'].notna().sum()} with lyrics)")
    else:
        existing = pd.DataFrame(columns=["song", "artist", "lyrics"])

    # Build a lookup of existing lyrics by (song, artist)
    existing["song_key"]   = existing["song"].str.strip().str.lower()
    existing["artist_key"] = existing["artist"].str.strip().str.lower()
    lyrics_map = existing.dropna(subset=["lyrics"]).set_index(
        ["song_key", "artist_key"]
    )["lyrics"].to_dict()

    new_df["song_key"]   = new_df["song"].str.strip().str.lower()
    new_df["artist_key"] = new_df["artist"].str.strip().str.lower()

    # Carry over existing lyrics
    new_df["lyrics"] = new_df.apply(
        lambda r: lyrics_map.get((r["song_key"], r["artist_key"])), axis=1
    )
    new_df = new_df.drop(columns=["song_key", "artist_key"])

    # Add tier + train_weight columns expected by training pipeline
    if "tier" not in new_df.columns:
        new_df["tier"] = -1
    if "train_weight" not in new_df.columns:
        import numpy as np
        vc = new_df["video_count"].fillna(0).astype(float)
        new_df["train_weight"] = np.log1p(vc)

    carried = new_df["lyrics"].notna().sum()
    print(f"[Merge] New dataset: {len(new_df)} songs ({carried} lyrics carried over)")
    return new_df


def fetch_missing_lyrics(df: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    missing_mask = df["lyrics"].isna()
    missing = missing_mask.sum()
    print(f"\n[Genius] Fetching lyrics for {missing} songs ...")

    if missing == 0:
        return df

    if not GENIUS_TOKEN:
        print("  [!] GENIUS_ACCESS_TOKEN not set — skipping lyrics fetch.")
        return df

    genius = _make_client()
    found  = 0

    for idx, row in df[missing_mask].iterrows():
        artist = str(row["artist"]).strip()
        song   = clean_title(str(row["song"]).strip())
        print(f"  [{idx+1}/{len(df)}] {song} — {artist}")

        lyrics = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyrics if lyrics else ""
        if lyrics:
            found += 1

        time.sleep(delay)

        if found % 10 == 0 and found > 0:
            df.to_csv(CM_TRAINING_CSV, index=False)
            print(f"    → Progress saved ({found} new lyrics so far)")

    df.to_csv(CM_TRAINING_CSV, index=False)
    total_with = int((df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum())
    print(f"\n[Genius] Done. New lyrics: {found} | Total with lyrics: {total_with}/{len(df)}")
    return df


def run(fetch_lyrics_flag: bool = True, delay: float = 0.5):
    # 1. Fetch all 2026 weekly charts
    new_df = fetch_all_charts()
    if new_df.empty:
        print("No chart data fetched — exiting.")
        return

    # 2. Merge with existing data (carry over lyrics already fetched)
    df = merge_with_existing(new_df)

    # 3. Save merged dataset
    df.to_csv(CM_TRAINING_CSV, index=False)
    print(f"\n[Save] {len(df)} songs saved → {CM_TRAINING_CSV}")

    # 4. Fetch missing lyrics
    if fetch_lyrics_flag:
        df = fetch_missing_lyrics(df, delay=delay)
        df.to_csv(CM_TRAINING_CSV, index=False)

    # Summary
    with_lyrics = int((df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum())
    print(f"\n{'='*50}")
    print(f"  Total songs      : {len(df)}")
    print(f"  With lyrics      : {with_lyrics}")
    print(f"  Without lyrics   : {len(df) - with_lyrics}")
    print(f"  Saved to         : {CM_TRAINING_CSV}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lyrics", action="store_true", help="Skip Genius lyrics fetch")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between Genius calls")
    args = parser.parse_args()
    run(fetch_lyrics_flag=not args.no_lyrics, delay=args.delay)

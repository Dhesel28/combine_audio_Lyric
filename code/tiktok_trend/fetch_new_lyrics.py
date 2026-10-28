"""
Fetch Genius lyrics for the friend's TikTok dataset (tiktok_new_to_dhesel.csv).

Strips 'sped up', 'slowed', 'ultra slowed', etc. from titles before searching
so Genius can find the original song.

Saves incrementally to data/tiktok_new_to_dhesel_lyrics.csv.

Usage:
    python fetch_new_lyrics.py
    python fetch_new_lyrics.py --delay 1.0   # slower if rate-limited
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from genius_fetcher import fetch_lyrics, _make_client
from config import GENIUS_TOKEN

DATA_DIR   = Path(__file__).parent / "data"
INPUT_CSV  = DATA_DIR / "tiktok_new_to_dhesel.csv"
OUTPUT_CSV = Path("/Volumes/Dhesel Driv /Work") / "tiktok_new_to_dhesel_lyrics.csv"

# Strip TikTok remix/edit suffixes before Genius search
_SUFFIX_RE = re.compile(
    r"\s*[-–]\s*"
    r"(sped[- ]?up|speed[- ]?up|slowed(?:\s*\+?\s*reverb)?|ultra\s*slowed|"
    r"slowed\s*version|nightcore|reverb only|lo[- ]?fi|phonk|drill|edit|"
    r"acoustic|live|extended|official)\s*$",
    re.IGNORECASE,
)
_FEAT_RE   = re.compile(r"\s*\(feat\..*?\)", re.IGNORECASE)
_PAREN_RE  = re.compile(
    r"\s*\((official\s*(audio|video|lyric[s]?|music\s*video|mv)?|"
    r"audio|lyric[s]?|music\s*video|mv)\s*\)",
    re.IGNORECASE,
)


def clean_title(title: str) -> str:
    """Strip TikTok-specific suffixes so Genius can find the original song."""
    t = _SUFFIX_RE.sub("", title)
    t = _FEAT_RE.sub("", t)
    t = _PAREN_RE.sub("", t)
    return t.strip()


def run(delay: float = 0.5):
    if not GENIUS_TOKEN:
        print("ERROR: GENIUS_ACCESS_TOKEN not set — check your .env file.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Resume from a previous interrupted run
    if OUTPUT_CSV.exists():
        saved = pd.read_csv(OUTPUT_CSV)
        df = df.merge(
            saved[["artist", "title", "lyrics"]],
            on=["artist", "title"],
            how="left",
        )
        print(f"Resuming — loaded {OUTPUT_CSV.name}")
    else:
        df["lyrics"] = None

    df["search_title"] = df["title"].apply(clean_title)

    genius = _make_client()
    total  = len(df)
    found  = int((df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum())
    print(f"Total songs : {total}")
    print(f"Already done: {found}")
    print(f"Still needed: {total - found}\n")

    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics", "")).strip():
            continue

        artist       = str(row["artist"]).strip()
        search_title = str(row["search_title"]).strip()
        print(f"  [{idx+1}/{total}] {search_title} — {artist}")

        lyrics = fetch_lyrics(artist, search_title, genius)
        df.at[idx, "lyrics"] = lyrics if lyrics else ""
        if lyrics:
            found += 1

        time.sleep(delay)

        if (idx + 1) % 10 == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"    → Progress saved  ({found} found so far)")

    df.to_csv(OUTPUT_CSV, index=False)
    final = int((df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum())
    print(f"\nDone.  Lyrics found: {final}/{total}")
    print(f"Saved  → {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch lyrics for tiktok_new_to_dhesel dataset")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls (default 0.5)")
    args = parser.parse_args()
    run(delay=args.delay)

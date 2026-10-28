"""
Fetch Genius lyrics for the 121 Chartmetric songs missing lyrics in cm_training_songs.csv.
Saves progress incrementally every 10 songs.

Usage:
    python fetch_cm_lyrics.py
    python fetch_cm_lyrics.py --delay 1.0   # slower if rate-limited
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
from config import GENIUS_TOKEN, CM_TRAINING_CSV

# Strip TikTok remix/edit suffixes before Genius search
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


def run(delay: float = 0.5):
    if not GENIUS_TOKEN:
        print("ERROR: GENIUS_ACCESS_TOKEN not set — check your .env file.")
        sys.exit(1)

    df = pd.read_csv(CM_TRAINING_CSV)

    missing = df["lyrics"].isna().sum()
    already = df["lyrics"].notna().sum()
    total   = len(df)
    print(f"Total songs : {total}")
    print(f"Already done: {already}")
    print(f"Still needed: {missing}\n")

    genius = _make_client()
    found  = 0

    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics", "")).strip():
            continue

        artist = str(row["artist"]).strip()
        song   = clean_title(str(row["song"]).strip())
        print(f"  [{idx+1}/{total}] {song} — {artist}")

        lyrics = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyrics if lyrics else ""
        if lyrics:
            found += 1

        time.sleep(delay)

        if (idx + 1) % 10 == 0:
            df.to_csv(CM_TRAINING_CSV, index=False)
            print(f"    → Progress saved ({found} new lyrics found so far)")

    df.to_csv(CM_TRAINING_CSV, index=False)
    total_with_lyrics = int((df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum())
    print(f"\nDone. New lyrics fetched: {found}")
    print(f"Total songs with lyrics: {total_with_lyrics}/{total}")
    print(f"Saved → {CM_TRAINING_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()
    run(delay=args.delay)

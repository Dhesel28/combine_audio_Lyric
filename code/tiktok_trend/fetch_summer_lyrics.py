#!/usr/bin/env python3
"""
Fetch Genius lyrics for every song in the Summer Dataset.

  - Loads Summer Dataset (4,785 songs with tiers 0-4, no lyrics)
  - Pre-fills from existing sources to avoid unnecessary API calls
  - Fetches remaining via Genius API
  - Saves progress every 25 songs so you can safely stop and resume
  - Final output: summer_dataset_with_lyrics.csv on external drive

Usage:
  cd /Users/dhekha/Desktop/SUPA_Spring/tiktok_trend
  python fetch_summer_lyrics.py           # full run / resume
  python fetch_summer_lyrics.py --limit 100   # test first 100 missing songs
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

EXTERNAL_DIR = Path("/Volumes/Dhesel Driv /Work")
LOCAL_DIR    = _HERE / "data"
OUT_DIR      = EXTERNAL_DIR if EXTERNAL_DIR.exists() else LOCAL_DIR

OUTPUT_FILE  = OUT_DIR / "summer_dataset_with_lyrics.csv"
SAVE_EVERY   = 25   # save progress every N songs fetched


def _resolve(filename: str) -> Path | None:
    for base in [EXTERNAL_DIR, LOCAL_DIR]:
        p = base / filename
        if p.exists():
            return p
    return None


# ── Text normalisation (for matching existing lyrics) ────────────────────────

_FEAT_RE    = re.compile(r'\s*[\(\[]?\s*feat\..*', re.IGNORECASE)
_PAREN_RE   = re.compile(r'\([^)]*\)')
_BRACKET_RE = re.compile(r'\[[^\]]*\]')
_NONWORD_RE = re.compile(r'[^\w\s]')
_WS_RE      = re.compile(r'\s+')


def _norm(text: str) -> str:
    t = str(text).lower()
    t = _FEAT_RE.sub('', t)
    t = _PAREN_RE.sub(' ', t)
    t = _BRACKET_RE.sub(' ', t)
    t = _NONWORD_RE.sub(' ', t)
    t = _WS_RE.sub(' ', t).strip()
    return t


def _key(artist: str, song: str) -> tuple[str, str]:
    return (_norm(artist), _norm(song))


# ── Build lyrics lookup from all existing sources ────────────────────────────

def build_existing_lookup() -> dict[tuple[str, str], str]:
    """Load all available lyrics so we skip those songs in the API fetch."""
    lookup: dict[tuple[str, str], str] = {}

    sources = [
        ("cm_training_songs.csv",           "artist", "song",  "lyrics"),
        ("billboard_2026.csv",              "artist", "song",  "lyrics"),
        ("catalog_with_lyrics.csv",         "artist", "song",  "lyrics"),
        ("tiktok_new_to_dhesel_lyrics.csv", "artist", "title", "lyrics"),
    ]

    for filename, a_col, s_col, l_col in sources:
        path = _resolve(filename)
        if path is None:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not all(c in df.columns for c in [a_col, s_col, l_col]):
            continue
        added = 0
        for _, row in df.iterrows():
            lyr = str(row[l_col]).strip()
            if not lyr or lyr.lower() == 'nan' or len(lyr) < 50:
                continue
            k = _key(str(row[a_col]), str(row[s_col]))
            if k not in lookup:
                lookup[k] = lyr
                added += 1
        print(f"  {filename:<42}  +{added:>5,} lyrics  (total {len(lookup):,})")

    return lookup


# ── Genius lyrics cleaning ────────────────────────────────────────────────────

_EMBED_TRAILER = re.compile(r"\d+\s*Embed\s*$", re.MULTILINE)
_YOU_MIGHT     = re.compile(r"You might also like.*", re.DOTALL | re.IGNORECASE)


def _clean(raw: str) -> str:
    text = _YOU_MIGHT.sub("", raw)
    text = _EMBED_TRAILER.sub("", text)
    m = re.search(r"\[", text)
    if m:
        text = text[m.start():]
    return text.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='Only fetch this many missing songs (0 = all)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Summer Dataset — Genius Lyrics Fetcher")
    print(f"Output → {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    # ── Load Summer Dataset ───────────────────────────────────────────────
    summer_path = _resolve("Summer Dataset.csv")
    if summer_path is None:
        print("[ERROR] Summer Dataset.csv not found on drive or locally")
        sys.exit(1)

    summer = pd.read_csv(summer_path)
    print(f"Summer Dataset: {len(summer):,} songs")

    # ── Resume from previous run if output file exists ────────────────────
    if OUTPUT_FILE.exists():
        prev = pd.read_csv(OUTPUT_FILE, low_memory=False)
        # Merge previously fetched lyrics into summer by song_id
        if 'lyrics' in prev.columns and 'song_id' in prev.columns:
            prev_map = {
                row['song_id']: str(row['lyrics']).strip()
                for _, row in prev.iterrows()
                if pd.notna(row.get('lyrics')) and str(row.get('lyrics', '')).strip() not in ('', 'nan')
            }
            if 'lyrics' not in summer.columns:
                summer['lyrics'] = ''
            summer['lyrics'] = summer.apply(
                lambda r: prev_map.get(r['song_id'], r.get('lyrics', '') or ''),
                axis=1
            )
            already = (summer['lyrics'].notna() & (summer['lyrics'].str.strip() != '')).sum()
            print(f"Resumed: {already:,} songs already have lyrics from previous run")
    else:
        summer['lyrics'] = ''

    # ── Pre-fill from existing sources (avoids API calls) ────────────────
    print("\nPre-filling lyrics from existing sources...")
    existing = build_existing_lookup()

    prefilled = 0
    for idx, row in summer.iterrows():
        if str(row.get('lyrics', '')).strip() not in ('', 'nan'):
            continue  # already has lyrics
        k = _key(str(row['artist']), str(row['song']))
        if k in existing:
            summer.at[idx, 'lyrics'] = existing[k]
            prefilled += 1

    has_lyrics = (summer['lyrics'].notna() & (summer['lyrics'].str.strip() != '') & (summer['lyrics'] != 'nan')).sum()
    needs_fetch = len(summer) - has_lyrics
    print(f"\n  Pre-filled {prefilled:,} songs from existing sources")
    print(f"  Songs with lyrics:    {has_lyrics:,} / {len(summer):,}")
    print(f"  Songs needing Genius: {needs_fetch:,} / {len(summer):,}")

    if needs_fetch == 0:
        print("\nAll songs already have lyrics — nothing to fetch!")
        summer.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved → {OUTPUT_FILE}")
        return

    # ── Set up Genius client ──────────────────────────────────────────────
    from config import GENIUS_TOKEN
    if not GENIUS_TOKEN:
        print("[ERROR] GENIUS_ACCESS_TOKEN not set in .env")
        sys.exit(1)

    import lyricsgenius
    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Acoustic)"],
        verbose=False,
        timeout=12,
    )
    genius.remove_section_headers = False  # keep [Verse], [Chorus] labels

    # ── Fetch missing lyrics ──────────────────────────────────────────────
    missing_idx = summer[
        (summer['lyrics'].isna()) |
        (summer['lyrics'].str.strip() == '') |
        (summer['lyrics'] == 'nan')
    ].index.tolist()

    if args.limit > 0:
        missing_idx = missing_idx[:args.limit]
        print(f"\nLimited to first {len(missing_idx):,} missing songs")

    total_to_fetch = len(missing_idx)
    print(f"\nFetching {total_to_fetch:,} songs via Genius...")
    print("(saves every 25 songs — safe to Ctrl+C and resume)\n")

    fetched_ok   = 0
    fetched_fail = 0

    for i, idx in enumerate(missing_idx, 1):
        row    = summer.loc[idx]
        artist = str(row['artist']).strip()
        song   = str(row['song']).strip()

        pct = i / total_to_fetch * 100
        print(f"  [{i:>5}/{total_to_fetch}  {pct:>5.1f}%]  {artist} — {song}", end="  ", flush=True)

        try:
            result = genius.search_song(song, artist)
            if result and result.lyrics:
                lyrics = _clean(result.lyrics)
                summer.at[idx, 'lyrics'] = lyrics
                fetched_ok += 1
                print("✓")
            else:
                summer.at[idx, 'lyrics'] = ''
                fetched_fail += 1
                print("–")
        except Exception as e:
            summer.at[idx, 'lyrics'] = ''
            fetched_fail += 1
            print(f"err: {e}")

        time.sleep(0.2)  # respect Genius rate limit

        # Save progress checkpoint
        if i % SAVE_EVERY == 0:
            summer.to_csv(OUTPUT_FILE, index=False)
            total_with = (summer['lyrics'].notna() & (summer['lyrics'].str.strip() != '') & (summer['lyrics'] != 'nan')).sum()
            print(f"\n  -- checkpoint: {i}/{total_to_fetch} fetched | {total_with:,} total with lyrics | saved → {OUTPUT_FILE}\n")

    # Final save
    summer.to_csv(OUTPUT_FILE, index=False)

    total_with = (summer['lyrics'].notna() & (summer['lyrics'].str.strip() != '') & (summer['lyrics'] != 'nan')).sum()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Songs with lyrics: {total_with:,} / {len(summer):,} ({total_with/len(summer)*100:.1f}%)")
    print(f"  Genius fetched OK: {fetched_ok:,}")
    print(f"  Not found:         {fetched_fail:,}")
    print(f"  Saved → {OUTPUT_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

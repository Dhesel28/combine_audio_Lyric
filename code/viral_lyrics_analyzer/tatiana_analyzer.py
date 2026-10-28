"""
Tatiana Analyzer — Source C for training data.

Loads Tatiana_cyanite.csv (16 songs with Cyanite audio features),
fetches lyrics from Genius for each song, runs NLP section scoring,
and returns a per-section DataFrame for training + dashboard use.

Audio features merged per-section from Cyanite:
  cyanite_valence, cyanite_arousal, cyanite_tempo_bpm,
  cyanite_energy_level, cyanite_mood_tags, cyanite_genre_tags

Run: python tatiana_analyzer.py
"""

from __future__ import annotations

import sys
import time
import re
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import GENIUS_TOKEN, DATA_DIR
from nlp_viral_detector import (
    _parse_sections, _section_type, score_section, auto_label_sections,
)

FULL_LYRICS_CSV = Path(__file__).parent.parent / "Testing lyrics " / "full_tatiana_demaria_lyrics.csv"

TATIANA_CSV = Path(__file__).parent.parent / "Tatiana_cyanite.csv"

ENERGY_MAP = {"low": 0.25, "medium": 0.5, "high": 0.75, "variable": 0.5}


def _clean_title(title: str) -> str:
    """Strip surrounding quotes and 'Official Audio/Video' suffixes."""
    title = title.strip().strip("'\"")
    for suffix in ["Official Audio", "Official Video", "Official", "Lyric Video"]:
        title = re.sub(rf"\s*{suffix}\s*$", "", title, flags=re.IGNORECASE).strip()
    return title


def load_tatiana_csv() -> pd.DataFrame:
    """Load Tatiana_cyanite.csv and return cleaned DataFrame."""
    if not TATIANA_CSV.exists():
        raise FileNotFoundError(f"Cannot find {TATIANA_CSV}")
    df = pd.read_csv(TATIANA_CSV)
    df["song_title_clean"] = df["song_title"].apply(_clean_title)
    df["energy_numeric"] = df["cyanite_energy_level"].map(ENERGY_MAP).fillna(0.5)
    return df


def analyze_tatiana_songs(verbose: bool = True) -> pd.DataFrame:
    """
    For each Tatiana song:
      1. Fetch lyrics from Genius
      2. Parse into sections
      3. Run 12 NLP signals + viral score
      4. Merge Cyanite audio features (song-level replicated per section)
      5. Label: section with highest viral_score = is_viral=1

    Returns combined DataFrame (one row per section across all songs).
    """
    import lyricsgenius

    tatiana_df = load_tatiana_csv()

    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        verbose=False,
    )
    genius.remove_section_headers = False

    all_rows = []

    for _, song_row in tatiana_df.iterrows():
        title = song_row["song_title_clean"]
        artist = "Tatiana DeMaria"

        if verbose:
            print(f"  → {title}")

        try:
            song = genius.search_song(title, artist)
            if not song:
                song = genius.search_song(title, "")
            if not song:
                if verbose:
                    print(f"    [skip] Not found on Genius")
                continue
        except Exception as e:
            if verbose:
                print(f"    [skip] Genius error: {e}")
            continue

        raw_lyrics = song.lyrics
        sections = _parse_sections(raw_lyrics)
        if not sections:
            if verbose:
                print(f"    [skip] No sections parsed")
            continue

        labels = list(sections.keys())
        total = len(labels)
        section_rows = []

        for idx, label in enumerate(labels):
            lines = sections[label]
            stype = _section_type(label)
            scores = score_section(lines, idx, total, raw_lyrics)

            row = {
                "song":          song.title,
                "artist":        artist,
                "section_label": label,
                "section_type":  stype,
                "source":        "tatiana_cyanite",
                **scores,
                "lyrics_preview": " / ".join(lines[:3]),
                "full_lyrics":    "\n".join(lines),
                # Cyanite audio features (song-level, replicated per section)
                "cyanite_valence":     float(song_row.get("cyanite_valence", 0) or 0),
                "cyanite_arousal":     float(song_row.get("cyanite_arousal", 0) or 0),
                "cyanite_tempo_bpm":   float(song_row.get("cyanite_tempo_bpm", 120) or 120),
                "cyanite_energy":      float(song_row.get("energy_numeric", 0.5) or 0.5),
                "cyanite_mood_tags":   str(song_row.get("cyanite_mood_tags", "")),
                "cyanite_genre_tags":  str(song_row.get("cyanite_genre_tags", "")),
                "recommended_clip":    str(song_row.get("recommended_clip", "")),
                "clip_start_sec":      float(song_row.get("cyanite_best_clip_start_sec", 0) or 0),
                "clip_end_sec":        float(song_row.get("cyanite_best_clip_end_sec", 0) or 0),
                "valentine_fit_score": float(song_row.get("valentine_fit_score", 0) or 0),
            }
            section_rows.append(row)

        if not section_rows:
            continue

        # Label: section with highest viral_score = is_viral=1
        best_score = max(r["viral_score"] for r in section_rows)
        for r in section_rows:
            r["is_viral"] = 1 if r["viral_score"] == best_score else 0

        all_rows.extend(section_rows)

        if verbose:
            best = next(r for r in section_rows if r["is_viral"] == 1)
            print(f"    ✓ {total} sections | viral: {best['section_label']} ({best['viral_score']:.3f})")

        time.sleep(0.5)

    if not all_rows:
        print("  No Tatiana songs found on Genius.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    out = DATA_DIR / "tatiana_sections.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"\n  Saved {len(df)} sections → {out}")
    return df


def get_tatiana_dashboard_data() -> pd.DataFrame:
    """Load cached tatiana_sections.csv or compute fresh."""
    cache = DATA_DIR / "tatiana_sections.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print("Cache not found — analyzing Tatiana songs (requires Genius API)...")
    return analyze_tatiana_songs()


def analyze_full_lyrics_csv(verse_size: int = 8, verbose: bool = True) -> pd.DataFrame:
    """
    Process full_tatiana_demaria_lyrics.csv (6 songs, no section headers).

    Steps per song:
      1. auto_label_sections() — sliding-window chorus detector (verse_size lines per verse)
      2. _parse_sections() — split into {label: [lines]} dict
      3. score_section() — compute 12 NLP signals + viral_score
      4. Label: section with highest viral_score → is_viral=1

    Returns one row per section across all songs.
    Caches to data/tatiana_full_lyrics_sections.csv.
    """
    if not FULL_LYRICS_CSV.exists():
        if verbose:
            print(f"[!] Full lyrics CSV not found at {FULL_LYRICS_CSV}")
        return pd.DataFrame()

    raw_df = pd.read_csv(FULL_LYRICS_CSV)
    if verbose:
        print(f"  Loaded {len(raw_df)} songs from full_tatiana_demaria_lyrics.csv")

    all_rows: List[dict] = []

    for _, row in raw_df.iterrows():
        song_name = str(row["song"])
        artist = str(row["artist"])
        raw_lyrics = str(row["lyrics"])

        labeled = auto_label_sections(raw_lyrics, verse_size=verse_size)
        sections = _parse_sections(labeled)

        if not sections:
            if verbose:
                print(f"  [skip] {song_name} — no sections after labeling")
            continue

        labels = list(sections.keys())
        total = len(labels)
        section_rows: List[dict] = []

        for idx, label in enumerate(labels):
            lines = sections[label]
            stype = _section_type(label)
            scores = score_section(lines, idx, total, labeled)
            section_rows.append({
                "song":           song_name,
                "artist":         artist,
                "section_label":  label,
                "section_type":   stype,
                "source":         "tatiana_full_lyrics",
                "lyrics_preview": " / ".join(lines[:3]),
                "full_lyrics":    "\n".join(lines),
                **scores,
            })

        if not section_rows:
            continue

        best_score = max(r["viral_score"] for r in section_rows)
        for r in section_rows:
            r["is_viral"] = 1 if r["viral_score"] == best_score else 0

        all_rows.extend(section_rows)

        if verbose:
            best = next(r for r in section_rows if r["is_viral"] == 1)
            print(
                f"  ✓ {song_name} — {total} sections | "
                f"viral: {best['section_label']} ({best['viral_score']:.3f})"
            )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    out = DATA_DIR / "tatiana_full_lyrics_sections.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"  Saved {len(df)} sections → {out}")
    return df


def get_full_lyrics_dashboard_data(force: bool = False) -> pd.DataFrame:
    """Load cached tatiana_full_lyrics_sections.csv or compute fresh."""
    cache = DATA_DIR / "tatiana_full_lyrics_sections.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    print("Analyzing full Tatiana lyrics (no Genius API needed)...")
    return analyze_full_lyrics_csv()


if __name__ == "__main__":
    print("Analyzing Tatiana DeMaria songs from Tatiana_cyanite.csv...\n")
    df = analyze_tatiana_songs()
    if not df.empty:
        print(f"\n{'='*60}")
        print(f"Total: {len(df)} sections across {df['song'].nunique()} songs")
        viral = df[df["is_viral"] == 1][["song", "section_label", "viral_score"]]
        print("\nPredicted viral sections:")
        print(viral.to_string(index=False))

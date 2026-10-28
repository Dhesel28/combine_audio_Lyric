"""
Main pipeline: tie everything together.

Usage:
    python pipeline.py                    # analyze trending TikTok songs
    python pipeline.py --songs manual     # use a manual list of songs
    python pipeline.py --song "Espresso" "Sabrina Carpenter"  # single song
"""

import argparse
import pandas as pd
from pathlib import Path
from config import DATA_DIR
from tiktok_trends import get_trending_songs
from genius_analyzer import analyze_multiple_songs, analyze_song, parse_lyrics_into_sections
from youtube_shorts import cross_check_song
import lyricsgenius
from config import GENIUS_TOKEN
import time


# ── Curated fallback song list (in case TikTok API is down) ───────────────────
MANUAL_SONGS = [
    ("Espresso", "Sabrina Carpenter"),
    ("APT.", "ROSÉ"),
    ("Die With A Smile", "Lady Gaga"),
    ("Taste", "Sabrina Carpenter"),
    ("Luther", "Kendrick Lamar"),
    ("Beautiful Things", "Benson Boone"),
    ("Too Sweet", "Hozier"),
    ("Supernatural", "NewJeans"),
    ("Cruel Summer", "Taylor Swift"),
    ("Paint The Town Red", "Doja Cat"),
]


def load_tiktok_songs(count: int = 30) -> list[tuple[str, str]]:
    """Get trending songs from TikTok API."""
    print("\n[Step 1] Fetching TikTok trending songs...")
    df = get_trending_songs(count=count)

    if df.empty:
        print("  TikTok fetch failed — using curated fallback list")
        return MANUAL_SONGS

    songs = []
    skip_terms = {"original sound", "mashup", "remix", "tiktok mashup"}
    for _, row in df.head(20).iterrows():
        title = str(row.get("music_title", "")).strip()
        artist = str(row.get("music_author", "")).strip()
        if not title or not artist:
            continue
        if any(t in title.lower() for t in skip_terms):
            continue
        songs.append((title, artist))

    print(f"  Got {len(songs)} real songs from TikTok (filtered)")
    if len(songs) < 5:
        print("  Too few real songs found — supplementing with curated list")
        seen = {(t.lower(), a.lower()) for t, a in songs}
        for t, a in MANUAL_SONGS:
            if (t.lower(), a.lower()) not in seen:
                songs.append((t, a))

    return songs[:20]


def run_pipeline(
    songs: list[tuple[str, str]],
    cross_check_youtube: bool = True,
    save_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Full pipeline:
    1. Analyze each song's lyric sections (Genius)
    2. Cross-check with YouTube Shorts
    3. Merge results into one DataFrame
    """
    print(f"\n[Step 2] Analyzing {len(songs)} songs on Genius...")
    genius_df = analyze_multiple_songs(songs, save=True)

    if genius_df.empty:
        print("No Genius data. Check your GENIUS_ACCESS_TOKEN.")
        return pd.DataFrame()

    # Build section dict for YouTube cross-check
    if cross_check_youtube:
        print("\n[Step 3] Cross-checking with YouTube Shorts...")
        genius_client = lyricsgenius.Genius(GENIUS_TOKEN, verbose=False)
        yt_all = []

        for title, artist in songs:
            song = genius_client.search_song(title, artist)
            if not song:
                continue
            sections = parse_lyrics_into_sections(song.lyrics)
            yt_df = cross_check_song(title, artist, sections, max_results=30)
            if not yt_df.empty:
                yt_all.append(yt_df)
            time.sleep(0.5)

        if yt_all:
            yt_combined = pd.concat(yt_all, ignore_index=True)
            # Merge with Genius results
            merged = genius_df.merge(
                yt_combined[["song", "section_label", "yt_shorts_matches",
                             "unique_videos_matched", "top_matched_line"]],
                on=["song", "section_label"],
                how="left",
            )
            merged["yt_shorts_matches"] = merged["yt_shorts_matches"].fillna(0).astype(int)
            merged["unique_videos_matched"] = merged["unique_videos_matched"].fillna(0).astype(int)

            # Final cross-platform score
            # Normalize YT matches
            max_yt = merged["yt_shorts_matches"].max()
            if max_yt > 0:
                merged["yt_score"] = (merged["yt_shorts_matches"] / max_yt).round(3)
            else:
                merged["yt_score"] = 0.0

            merged["cross_platform_score"] = (
                merged["combined_viral_score"] * 0.6 + merged["yt_score"] * 0.4
            ).round(3)

            out = save_dir / "cross_platform_analysis.csv"
            merged.to_csv(out, index=False)
            print(f"\nSaved cross-platform analysis to {out}")
            return merged
        else:
            print("  No YouTube Shorts data (check API key or quota)")

    return genius_df


def print_top_hooks(df: pd.DataFrame, top_n: int = 10):
    """Print the top viral lyric sections found."""
    score_col = "cross_platform_score" if "cross_platform_score" in df.columns else "combined_viral_score"
    top = df.nlargest(top_n, score_col)

    print(f"\n{'='*70}")
    print(f"TOP {top_n} VIRAL LYRIC SECTIONS ({score_col})")
    print('='*70)

    for _, row in top.iterrows():
        print(f"\n[{row['rank']}] {row['song']} — {row['section_label']} ({row['section_type']})")
        print(f"    Score: {row[score_col]:.3f} | Genius annot: {int(row.get('genius_annotation_count', 0))} | YT matches: {int(row.get('yt_shorts_matches', 0))}")
        print(f"    Preview: {row['lyrics_preview'][:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", choices=["tiktok", "manual"], default="tiktok",
                        help="Where to get the song list from")
    parser.add_argument("--song", nargs=2, metavar=("TITLE", "ARTIST"),
                        help="Analyze a single song: --song 'Espresso' 'Sabrina Carpenter'")
    parser.add_argument("--no-youtube", action="store_true",
                        help="Skip YouTube Shorts cross-check")
    args = parser.parse_args()

    if args.song:
        # Single song mode
        df = analyze_song(args.song[0], args.song[1])
        if df is not None:
            print(df[["section_label", "section_type", "combined_viral_score",
                       "genius_annotation_count", "lyrics_preview"]].to_string(index=False))
    else:
        if args.songs == "tiktok":
            songs = load_tiktok_songs(count=30)
        else:
            songs = MANUAL_SONGS
            print(f"Using {len(songs)} curated songs")

        df = run_pipeline(songs, cross_check_youtube=not args.no_youtube)
        if not df.empty:
            print_top_hooks(df, top_n=15)

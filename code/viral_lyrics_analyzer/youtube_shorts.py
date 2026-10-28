"""
YouTube Shorts cross-check:
- Search YouTube Shorts for a given song
- Extract which lyric phrases appear in video titles/descriptions/captions
- Compare with Genius trending sections to see if same portions go viral

Requires: YOUTUBE_API_KEY in .env
Get one free at: console.cloud.google.com → Enable "YouTube Data API v3"
"""

import re
import time
import requests
import pandas as pd
from collections import Counter
from config import YOUTUBE_API_KEY, DATA_DIR

YOUTUBE_BASE = "https://www.googleapis.com/youtube/v3"


def _has_yt_key() -> bool:
    return bool(YOUTUBE_API_KEY) and YOUTUBE_API_KEY != "your_youtube_key_here"


def search_shorts_for_song(song_title: str, artist: str, max_results: int = 50) -> list[dict]:
    """
    Search YouTube Shorts for videos using a specific song.
    Returns list of video metadata dicts.
    """
    if not _has_yt_key():
        print("  [!] No YouTube API key. Add YOUTUBE_API_KEY to .env")
        print("  Get it free: console.cloud.google.com → YouTube Data API v3")
        return []

    query = f"{song_title} {artist} shorts"
    print(f"  [YouTube] Searching: '{query}'")

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoDuration": "short",  # Shorts are under 60s
        "maxResults": min(max_results, 50),
        "order": "viewCount",
        "key": YOUTUBE_API_KEY,
    }

    try:
        resp = requests.get(f"{YOUTUBE_BASE}/search", params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        print(f"  [!] YouTube search error: {e}")
        return []

    videos = []
    for item in items:
        snippet = item.get("snippet", {})
        videos.append({
            "video_id": item.get("id", {}).get("videoId", ""),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
        })

    print(f"  Found {len(videos)} Shorts")
    return videos


def get_video_captions(video_id: str) -> str:
    """
    Attempt to fetch auto-generated captions for a YouTube video.
    Uses the YouTube captions API (requires OAuth for full access).
    Falls back to title + description only.
    """
    # Full caption access requires OAuth which is complex.
    # We use the title + description as a proxy.
    return ""


def find_lyric_matches_in_videos(
    videos: list[dict],
    lyric_sections: dict[str, list[str]],
) -> pd.DataFrame:
    """
    For each lyric section, count how many YouTube Shorts titles/descriptions
    contain lines from that section.

    lyric_sections: {section_label: [list of lyric lines]}
    Returns DataFrame with match counts per section.
    """
    rows = []

    for section_label, lines in lyric_sections.items():
        total_matches = 0
        matched_videos = []
        matched_lines = Counter()

        for line in lines:
            line_clean = line.strip().lower()
            if len(line_clean) < 8:  # skip very short lines
                continue

            # Check each video title + description for this lyric line
            for video in videos:
                combined_text = (
                    video["title"] + " " + video["description"]
                ).lower()
                if line_clean in combined_text:
                    total_matches += 1
                    matched_videos.append(video["video_id"])
                    matched_lines[line] += 1

        rows.append({
            "section_label": section_label,
            "yt_shorts_matches": total_matches,
            "unique_videos_matched": len(set(matched_videos)),
            "top_matched_line": matched_lines.most_common(1)[0][0] if matched_lines else "",
            "top_matched_line_count": matched_lines.most_common(1)[0][1] if matched_lines else 0,
        })

    return pd.DataFrame(rows)


def cross_check_song(
    song_title: str,
    artist: str,
    lyric_sections: dict[str, list[str]],
    max_results: int = 50,
) -> pd.DataFrame:
    """
    Full cross-check for one song:
    1. Search YouTube Shorts for videos using this song
    2. Find which lyric sections appear in video titles/descriptions
    Returns DataFrame with YouTube match data per section.
    """
    videos = search_shorts_for_song(song_title, artist, max_results)
    if not videos:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "section_label", "yt_shorts_matches",
            "unique_videos_matched", "top_matched_line", "top_matched_line_count"
        ])

    df = find_lyric_matches_in_videos(videos, lyric_sections)
    df["song"] = song_title
    df["artist"] = artist
    df["yt_videos_searched"] = len(videos)
    return df


if __name__ == "__main__":
    # Test (requires YouTube API key)
    from genius_analyzer import parse_lyrics_into_sections
    import lyricsgenius
    from config import GENIUS_TOKEN

    genius = lyricsgenius.Genius(GENIUS_TOKEN, verbose=False)
    song = genius.search_song("Espresso", "Sabrina Carpenter")
    if song:
        sections = parse_lyrics_into_sections(song.lyrics)
        df = cross_check_song("Espresso", "Sabrina Carpenter", sections)
        print(df.to_string(index=False))

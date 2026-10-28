"""
Fetch trending TikTok songs and sounds using RapidAPI (tiktok-api23).
Saves results to data/tiktok_trending.csv
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from config import RAPIDAPI_KEY, TIKTOK_API_HOST, DATA_DIR


HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": TIKTOK_API_HOST,
}

BASE_URL = f"https://{TIKTOK_API_HOST}"


def get_trending_videos(count: int = 30) -> list[dict]:
    """
    Fetch trending TikTok videos via search for trending keywords.
    Uses /api/search/general which returns items with music metadata.
    """
    print(f"[TikTok] Fetching ~{count} trending videos via search...")

    # Search multiple trending keywords to build a broad pool
    keywords = ["trending song 2025", "viral song", "trending music", "tiktok hit 2025"]
    all_items = []

    for kw in keywords:
        if len(all_items) >= count:
            break
        params = {"keyword": kw, "count": min(20, count), "cursor": 0}
        try:
            resp = requests.get(
                f"{BASE_URL}/api/search/general",
                headers=HEADERS, params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            all_items.extend(items)
        except Exception as e:
            print(f"  [!] Search error for '{kw}': {e}")
            continue
        time.sleep(0.3)

    print(f"  Got {len(all_items)} raw items")

    results = []
    for entry in all_items:
        try:
            v = entry.get("item", entry)  # items are nested under "item" key
            music = v.get("music", {})
            stats = v.get("stats", {})
            author = v.get("author", {})

            results.append({
                "video_id": v.get("id", ""),
                "description": v.get("desc", ""),
                "author": author.get("uniqueId", ""),
                "play_count": stats.get("playCount", 0),
                "like_count": stats.get("diggCount", 0),
                "comment_count": stats.get("commentCount", 0),
                "share_count": stats.get("shareCount", 0),
                "music_id": music.get("id", ""),
                "music_title": music.get("title", ""),
                "music_author": music.get("authorName", ""),
                "music_duration": music.get("duration", 0),
            })
        except Exception:
            continue

    return results


def search_trending_sounds(query: str = "top songs 2025", count: int = 20) -> list[dict]:
    """Search TikTok for sounds/music by keyword."""
    print(f"[TikTok] Searching sounds: '{query}'...")

    endpoint = f"{BASE_URL}/api/search/general"
    params = {"keyword": query, "count": count, "type": 1}  # type 1 = music

    try:
        resp = requests.get(endpoint, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] Search error: {e}")
        return []

    items = data.get("data", [])
    sounds = []
    for item in items:
        if item.get("type") == 1:  # music items
            music = item.get("music_info", {})
            sounds.append({
                "music_id": music.get("id", ""),
                "music_title": music.get("title", ""),
                "music_author": music.get("author", ""),
                "video_count": item.get("common", {}).get("video_num", 0),
            })
    return sounds


def get_music_info(music_id: str) -> dict:
    """Get details for a specific TikTok sound/music."""
    endpoint = f"{BASE_URL}/api/music/info"
    params = {"musicId": music_id}

    try:
        resp = requests.get(endpoint, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [!] Music info error for {music_id}: {e}")
        return {}


def get_trending_songs(count: int = 50) -> pd.DataFrame:
    """
    Main function: get trending songs from TikTok.
    Returns a DataFrame of unique songs with engagement stats.
    """
    videos = get_trending_videos(count=count)

    if not videos:
        print("  [!] No videos fetched — check your RapidAPI key/quota")
        return pd.DataFrame()

    df = pd.DataFrame(videos)

    # Deduplicate by music — keep highest-play version
    if "music_title" in df.columns and not df.empty:
        # Filter out "original sound" and generic/foreign entries
        original_sound_pattern = r'original\s*sound|оригинальный звук|suara asli|som original'
        df = df[
            df["music_title"].notna()
            & (df["music_title"] != "")
            & ~df["music_title"].str.lower().str.contains(original_sound_pattern, regex=True, na=False)
            & (df["music_title"].str.len() > 3)
        ]
        song_stats = (
            df.groupby(["music_title", "music_author"])
            .agg(
                total_plays=("play_count", "sum"),
                total_likes=("like_count", "sum"),
                video_count=("video_id", "count"),
                music_id=("music_id", "first"),
            )
            .reset_index()
            .sort_values("total_plays", ascending=False)
        )
        return song_stats

    return df


if __name__ == "__main__":
    df = get_trending_songs(count=50)
    if not df.empty:
        out = DATA_DIR / "tiktok_trending.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} trending songs to {out}")
        print(df.head(15).to_string(index=False))
    else:
        print("No data returned.")

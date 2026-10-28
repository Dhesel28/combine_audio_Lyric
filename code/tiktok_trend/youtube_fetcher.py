"""
Fetch trending music from YouTube Data API v3.
Parses artist/song from video titles, enriches with Genius lyrics
and virality analysis. Outputs same format as tiktok_fetcher.
"""

from __future__ import annotations

import re
import time
import requests
import pandas as pd
from config import YOUTUBE_API_KEY, TIKTOK_TRENDING_CSV, DATA_DIR

YOUTUBE_BASE = "https://www.googleapis.com/youtube/v3"

# Patterns to extract "Artist - Song" from YouTube titles
_TITLE_PATTERNS = [
    # "Artist - Song Title (Official ...)"
    re.compile(r'^(.+?)\s*[-–—]\s*"?(.+?)"?\s*[\(\[]', re.I),
    # "Artist - Song Title"
    re.compile(r'^(.+?)\s*[-–—]\s*(.+)$', re.I),
]

_NOISE = re.compile(
    r'\b(official|music|video|lyric|audio|ft\.?|feat\.?|prod\.?|'
    r'explicit|version|visualizer|clip|4k|hd|vevo|topic)\b.*$',
    re.I,
)


def _parse_title(raw_title: str, channel: str) -> tuple[str, str]:
    """Return (artist, song) from a YouTube video title."""
    title = raw_title.strip()
    for pat in _TITLE_PATTERNS:
        m = pat.match(title)
        if m:
            artist = _NOISE.sub("", m.group(1)).strip().strip('"\'')
            song   = _NOISE.sub("", m.group(2)).strip().strip('"\'')
            if artist and song and len(artist) > 1 and len(song) > 1:
                return artist, song
    # Fallback: use channel name as artist, cleaned title as song
    clean = _NOISE.sub("", title).strip()
    artist = re.sub(r'\s*-\s*Topic$', '', channel, flags=re.I).strip()
    return artist, clean


_REGIONS = ["US", "GB", "CA", "AU", "IN", "BR", "DE", "FR", "MX", "KR"]


def _fetch_region(region: str, max_per_region: int = 50) -> list[dict]:
    """Fetch trending music videos for one region."""
    items: list[dict] = []
    page_token: str | None = None

    while len(items) < max_per_region:
        params: dict = {
            "part":            "snippet,statistics",
            "chart":           "mostPopular",
            "videoCategoryId": "10",
            "regionCode":      region,
            "maxResults":      min(50, max_per_region - len(items)),
            "key":             YOUTUBE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(f"{YOUTUBE_BASE}/videos", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [!] YouTube API error ({region}): {e}")
            break

        batch = data.get("items", [])
        # tag region on each item for dedup tracking
        for item in batch:
            item["_region"] = region
        items.extend(batch)

        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)

    return items


def _fetch_trending_pages(max_results: int = 200) -> list[dict]:
    """Pull trending music from multiple regions, deduplicate by video ID."""
    seen_ids: set[str] = set()
    all_items: list[dict] = []

    per_region = max(50, max_results // len(_REGIONS))

    for region in _REGIONS:
        if len(all_items) >= max_results:
            break
        batch = _fetch_region(region, max_per_region=per_region)
        new = 0
        for item in batch:
            vid_id = item.get("id", "")
            if vid_id and vid_id not in seen_ids:
                seen_ids.add(vid_id)
                all_items.append(item)
                new += 1
        print(f"  Region {region}: +{new} new (total: {len(all_items)})")
        time.sleep(0.5)

    return all_items


def _parse_videos(raw_items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in raw_items:
        try:
            snippet    = item.get("snippet", {})
            stats      = item.get("statistics", {})
            raw_title  = snippet.get("title", "")
            channel    = snippet.get("channelTitle", "")
            video_id   = item.get("id", "")

            artist, song = _parse_title(raw_title, channel)

            if not song or len(song) <= 2:
                continue

            rows.append({
                "song":          song,
                "artist":        artist,
                "music_id":      video_id,
                "total_plays":   int(stats.get("viewCount",    0)),
                "total_likes":   int(stats.get("likeCount",    0)),
                "total_comments":int(stats.get("commentCount", 0)),
                "total_shares":  0,
                "video_count":   1,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("total_plays", ascending=False)
    # Drop obvious duplicates by (song, artist)
    df = df.drop_duplicates(subset=["song", "artist"]).reset_index(drop=True)
    return df


def _enrich_lyrics(df: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    try:
        from genius_fetcher import fetch_lyrics, _make_client
        genius = _make_client()
    except Exception as e:
        print(f"  [!] Genius init failed: {e}")
        df["lyrics"] = None
        return df

    if "lyrics" not in df.columns:
        df["lyrics"] = None

    total = len(df)
    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics", "")).strip():
            continue
        artist, song = str(row["artist"]), str(row["song"])
        print(f"  [Genius] [{idx+1}/{total}] {song} — {artist}")
        lyr = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyr or ""
        time.sleep(delay)

    found = (df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum()
    print(f"  [Genius] Lyrics fetched for {found}/{total} songs")
    return df


def _enrich_virality(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from snippet_detector import virality_analysis
    except Exception:
        for col in ("virality_score", "virality_label", "best_snippet", "hook_line"):
            df[col] = None
        return df

    scores, labels, snippets, hooks = [], [], [], []
    for _, row in df.iterrows():
        lyr = str(row.get("lyrics", "")).strip()
        score, label, snippet, hook = None, None, None, None
        if lyr:
            try:
                va      = virality_analysis(lyr)
                score   = va["overall_virality"]
                label   = va["virality_label"]
                vm      = va.get("viral_moment", {})
                clip    = vm.get("viral_clip", [])
                vline   = vm.get("viral_line") or {}
                snippet = "\n".join(s["text"] for s in clip[:4])
                hook    = vline.get("text")
            except Exception:
                pass
        scores.append(score)
        labels.append(label)
        snippets.append(snippet)
        hooks.append(hook)

    df["virality_score"] = scores
    df["virality_label"] = labels
    df["best_snippet"]   = snippets
    df["hook_line"]      = hooks
    return df


def get_trending_songs(max_results: int = 200, fetch_lyrics: bool = True) -> pd.DataFrame:
    """
    Fetch top trending music from YouTube, enrich with Genius lyrics
    and virality analysis. Saves to tiktok_trending.csv on the external drive.
    """
    print(f"[YouTube] Fetching up to {max_results} trending music videos ...")
    raw = _fetch_trending_pages(max_results=max_results)
    print(f"  Raw videos collected: {len(raw)}")

    df = _parse_videos(raw)
    if df.empty:
        print("  [!] No valid songs parsed — check YouTube API key/quota")
        return df

    print(f"  Unique songs parsed: {len(df)}")
    df["source"] = "youtube_trending"
    df["tier"]   = -1

    # Carry over any existing lyrics to avoid re-fetching
    if TIKTOK_TRENDING_CSV.exists():
        existing = pd.read_csv(TIKTOK_TRENDING_CSV)
        if "lyrics" in existing.columns:
            lyr_map = (
                existing.dropna(subset=["lyrics"])
                .set_index(["song", "artist"])["lyrics"]
                .to_dict()
            )
            df["lyrics"] = df.apply(
                lambda r: lyr_map.get((r["song"], r["artist"])), axis=1
            )
        else:
            df["lyrics"] = None
    else:
        df["lyrics"] = None

    if fetch_lyrics:
        missing = df["lyrics"].isna().sum()
        print(f"[YouTube] Fetching lyrics for {missing} songs ...")
        df = _enrich_lyrics(df)

    print("[YouTube] Running virality analysis ...")
    df = _enrich_virality(df)

    df.to_csv(TIKTOK_TRENDING_CSV, index=False)
    print(f"[YouTube] Saved {len(df)} songs → {TIKTOK_TRENDING_CSV}")
    return df


if __name__ == "__main__":
    df = get_trending_songs()
    print(df[["song", "artist", "total_plays", "virality_label"]].head(20).to_string(index=False))

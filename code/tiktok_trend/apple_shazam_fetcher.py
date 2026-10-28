"""
Fetch real TikTok-adjacent hits from two free, no-auth sources:

1. Apple Music Top 100 (US) — reflects what's trending NOW
2. Shazam US Top 200 (via Kworb) — Shazam is the #1 TikTok song-discovery app

Both charts are heavily driven by TikTok virality and update weekly.
Results are merged, deduplicated, enriched with Genius lyrics + virality,
and saved to tiktok_trending.csv on the external drive.
"""

from __future__ import annotations

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from config import DATA_DIR, TIKTOK_TRENDING_CSV

APPLE_URL  = "https://rss.applemarketingtools.com/api/v2/us/music/most-played/100/songs.json"
SHAZAM_URL = "https://kworb.net/charts/shazam/us.html"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

_FEAT_RE = re.compile(
    r'\s*[\(\[](feat\.?|ft\.?|with|x|&)[^\)\]]*[\)\]]', re.I
)
_NOISE_RE = re.compile(
    r'\s*(official|music|video|lyric|audio|explicit|version|remaster|\(.*\)|\[.*\]).*$',
    re.I
)


def _clean(text: str) -> str:
    text = _FEAT_RE.sub("", text)
    text = _NOISE_RE.sub("", text)
    return text.strip()


# ── Apple Music Top 100 ────────────────────────────────────────────────────────

def fetch_apple_music() -> pd.DataFrame:
    """Fetch Apple Music US Top 100 songs."""
    print("[Apple Music] Fetching Top 100 US ...")
    try:
        resp = requests.get(APPLE_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [!] Error: {e}")
        return pd.DataFrame()

    results = data.get("feed", {}).get("results", [])
    rows = []
    for i, item in enumerate(results, 1):
        song   = _clean(item.get("name", ""))
        artist = _clean(item.get("artistName", ""))
        if song and artist:
            rows.append({
                "rank":   i,
                "song":   song,
                "artist": artist,
                "source": "apple_music_top100",
            })

    df = pd.DataFrame(rows)
    print(f"  Got {len(df)} songs from Apple Music")
    return df


# ── Shazam US Top 200 ──────────────────────────────────────────────────────────

def fetch_shazam() -> pd.DataFrame:
    """Scrape Shazam US Top 200 from Kworb."""
    print("[Shazam] Fetching US Top 200 ...")
    try:
        resp = requests.get(SHAZAM_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [!] Error: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows_data = []

    for row in soup.select("table tr"):
        cols = [td.get_text(strip=True) for td in row.select("td")]
        if len(cols) < 3:
            continue
        rank_str  = cols[0]
        entry     = cols[2] if len(cols) > 2 else cols[-1]

        try:
            rank = int(rank_str)
        except ValueError:
            continue

        # Entry format: "Artist - Song" or "Artist - Song (feat. ...)"
        if " - " in entry:
            parts  = entry.split(" - ", 1)
            artist = _clean(parts[0].strip())
            song   = _clean(parts[1].strip())
        else:
            artist = ""
            song   = _clean(entry)

        if song and len(song) > 1:
            rows_data.append({
                "rank":   rank,
                "song":   song,
                "artist": artist,
                "source": "shazam_us_top200",
            })

    df = pd.DataFrame(rows_data)
    print(f"  Got {len(df)} songs from Shazam")
    return df


# ── Merge + dedup ──────────────────────────────────────────────────────────────

def merge_charts(apple_df: pd.DataFrame, shazam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine both charts. Songs on both get a 'both_charts' source tag
    and their best (lowest) rank.
    """
    all_df = pd.concat([apple_df, shazam_df], ignore_index=True)

    # Score: lower rank = better. Combine by best rank per song+artist
    all_df["rank"] = pd.to_numeric(all_df["rank"], errors="coerce").fillna(999)
    all_df["song_key"]   = all_df["song"].str.lower().str.strip()
    all_df["artist_key"] = all_df["artist"].str.lower().str.strip()

    # Tag songs appearing on both
    apple_keys  = set(zip(apple_df["song"].str.lower(),  apple_df["artist"].str.lower()))
    shazam_keys = set(zip(shazam_df["song"].str.lower(), shazam_df["artist"].str.lower()))
    both_keys   = apple_keys & shazam_keys

    def tag(row):
        key = (row["song_key"], row["artist_key"])
        if key in both_keys:
            return "apple_and_shazam"
        return row["source"]

    all_df["source"] = all_df.apply(tag, axis=1)

    # Keep best rank per song
    all_df = (
        all_df.sort_values("rank")
        .drop_duplicates(subset=["song_key", "artist_key"])
        .drop(columns=["song_key", "artist_key"])
        .reset_index(drop=True)
    )

    print(f"\n[Merge] {len(all_df)} unique songs total")
    print(f"  On both charts : {(all_df['source'] == 'apple_and_shazam').sum()}")
    print(f"  Apple only     : {(all_df['source'] == 'apple_music_top100').sum()}")
    print(f"  Shazam only    : {(all_df['source'] == 'shazam_us_top200').sum()}")
    return all_df


# ── Lyrics enrichment ──────────────────────────────────────────────────────────

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

    # Carry over any lyrics already in the existing CSV
    if TIKTOK_TRENDING_CSV.exists():
        existing = pd.read_csv(TIKTOK_TRENDING_CSV)
        if "lyrics" in existing.columns:
            lyr_map = (
                existing.dropna(subset=["lyrics"])
                .assign(
                    song_key   = lambda x: x["song"].str.lower().str.strip(),
                    artist_key = lambda x: x["artist"].str.lower().str.strip(),
                )
                .set_index(["song_key", "artist_key"])["lyrics"]
                .to_dict()
            )
            df["lyrics"] = df.apply(
                lambda r: lyr_map.get(
                    (r["song"].lower().strip(), r["artist"].lower().strip())
                ), axis=1
            )

    total   = len(df)
    missing = df["lyrics"].isna().sum()
    print(f"\n[Lyrics] Fetching {missing}/{total} songs ...")

    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics", "")).strip():
            continue
        artist, song = str(row["artist"]), str(row["song"])
        print(f"  [Genius] [{idx+1}/{total}] {song} — {artist}")
        lyr = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyr or ""
        time.sleep(delay)

    found = (df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum()
    print(f"  Lyrics found: {found}/{total}")
    return df


# ── Virality enrichment ────────────────────────────────────────────────────────

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


# ── Public API ─────────────────────────────────────────────────────────────────

def get_tiktok_hits(fetch_lyrics: bool = True) -> pd.DataFrame:
    """
    Fetch real TikTok-driven hits from Apple Music + Shazam,
    enrich with lyrics and virality, save to tiktok_trending.csv.
    """
    apple_df  = fetch_apple_music()
    shazam_df = fetch_shazam()

    if apple_df.empty and shazam_df.empty:
        print("[!] Both sources failed.")
        return pd.DataFrame()

    df = merge_charts(apple_df, shazam_df)
    df["tier"]           = -1
    df["total_plays"]    = 0
    df["total_likes"]    = 0
    df["total_comments"] = 0
    df["total_shares"]   = 0
    df["music_id"]       = ""
    df["video_count"]    = 1

    if fetch_lyrics:
        df = _enrich_lyrics(df)

    print("\n[Virality] Running analysis ...")
    df = _enrich_virality(df)

    df.to_csv(TIKTOK_TRENDING_CSV, index=False)
    print(f"\n✓ Saved {len(df)} songs → {TIKTOK_TRENDING_CSV}")
    return df


if __name__ == "__main__":
    df = get_tiktok_hits()
    cols = ["rank", "song", "artist", "source", "virality_label"]
    print(df[cols].head(30).to_string(index=False))

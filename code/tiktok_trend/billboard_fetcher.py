"""
Scrape all 2026 Billboard Hot 100 weekly charts.
Deduplicates songs, fetches Genius lyrics, computes virality features,
and saves to external drive as billboard_2026.csv.
"""

from __future__ import annotations

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, timedelta
from pathlib import Path
from config import DATA_DIR, TIKTOK_TRENDING_CSV, BILLBOARD_CSV
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── Date generation ────────────────────────────────────────────────────────────

def _chart_dates_2026() -> list[str]:
    """Return all Saturday chart dates in 2026 up to today."""
    start = date(2026, 1, 3)
    today = date.today()
    dates = []
    d = start
    while d <= today:
        dates.append(str(d))
        d += timedelta(weeks=1)
    return dates


# ── Scraper ────────────────────────────────────────────────────────────────────

def _scrape_hot100(chart_date: str) -> list[dict]:
    """Scrape one week of Hot 100. Returns list of {rank, song, artist, chart_date}."""
    url = f"https://www.billboard.com/charts/hot-100/{chart_date}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [!] Failed to fetch {chart_date}: {e}")
        return []

    soup    = BeautifulSoup(resp.text, "html.parser")
    titles  = soup.select("li.o-chart-results-list__item h3#title-of-a-story")
    artists = soup.select("li.o-chart-results-list__item span.c-label.a-no-trucate")

    rows = []
    for rank, (t, a) in enumerate(zip(titles, artists), start=1):
        song   = t.get_text(strip=True)
        artist = a.get_text(strip=True)
        if song and artist:
            rows.append({
                "rank":        rank,
                "song":        song,
                "artist":      artist,
                "chart_date":  chart_date,
                "peak_rank":   rank,   # will be updated during dedup
            })
    return rows


def fetch_all_2026(delay: float = 1.5) -> pd.DataFrame:
    """
    Scrape all 2026 Hot 100 weeks. Returns deduplicated DataFrame
    with peak_rank = best (lowest) rank seen across all weeks.
    """
    dates = _chart_dates_2026()
    print(f"[Billboard] Scraping {len(dates)} weeks of Hot 100 ...")

    all_rows: list[dict] = []
    for i, d in enumerate(dates, 1):
        rows = _scrape_hot100(d)
        all_rows.extend(rows)
        print(f"  [{i}/{len(dates)}] {d}: {len(rows)} songs")
        time.sleep(delay)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Deduplicate: keep best (lowest) rank per (song, artist)
    df["peak_rank"] = df.groupby(["song", "artist"])["rank"].transform("min")
    df = (
        df.sort_values("peak_rank")
        .drop_duplicates(subset=["song", "artist"])
        .reset_index(drop=True)
    )
    df = df.drop(columns=["rank"])
    print(f"[Billboard] {len(df)} unique songs after dedup")
    return df


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

    total = len(df)
    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics", "")).strip():
            continue
        artist, song = str(row["artist"]), str(row["song"])
        print(f"  [Genius] [{idx+1}/{total}] {song} — {artist}")
        lyr = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyr or ""
        time.sleep(delay)

        # Save progress every 20 songs
        if (idx + 1) % 20 == 0:
            df.to_csv(BILLBOARD_CSV, index=False)
            print(f"    → Progress saved ({idx+1}/{total})")

    found = (df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum()
    print(f"  [Genius] Lyrics fetched for {found}/{total} Billboard songs")
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

def get_billboard_2026(fetch_lyrics: bool = True) -> pd.DataFrame:
    """
    Fetch all 2026 Billboard Hot 100, enrich with lyrics + virality.
    Saves to billboard_2026.csv on the external drive.
    """
    # Load existing progress if available
    if BILLBOARD_CSV.exists():
        print(f"[Billboard] Resuming from existing {BILLBOARD_CSV.name} ...")
        df = pd.read_csv(BILLBOARD_CSV)
    else:
        df = fetch_all_2026()
        if df.empty:
            return df
        df["source"] = "billboard_hot100"
        df["tier"]   = -1

    if fetch_lyrics:
        missing = df["lyrics"].isna().sum() if "lyrics" in df.columns else len(df)
        print(f"[Billboard] Fetching lyrics for {missing} songs ...")
        df = _enrich_lyrics(df)

    print("[Billboard] Running virality analysis ...")
    df = _enrich_virality(df)

    df.to_csv(BILLBOARD_CSV, index=False)
    print(f"[Billboard] Saved {len(df)} songs → {BILLBOARD_CSV}")
    return df


if __name__ == "__main__":
    df = get_billboard_2026()
    print(df[["song", "artist", "peak_rank", "virality_label"]].head(30).to_string(index=False))

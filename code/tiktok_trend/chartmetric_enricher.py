"""
Chartmetric Training Enricher
==============================
Converts Chartmetric trending data (cm_tiktok_sounds.csv / cm_tiktok_tracks.csv)
into catalog-format rows that train.py can directly consume.

Key role in fixing the algorithm
---------------------------------
Chartmetric's `video_count` tells us HOW MANY TikTok videos actually used each sound.
This is ground-truth TikTok virality — far more reliable than:
  • tier 3/4 (just big chart hits, not necessarily TikTok-viral)
  • RapidAPI search scrape (keyword-biased, quota-limited)

We use video_count to compute a per-song training weight so the model
pays proportionally more attention to high-confidence viral examples.

Weight formula  (used in train.py):
    base    = 2.0                            (all CM songs = confirmed TikTok)
    bonus   = log1p(video_count / 50_000)    (scales from 0 for unknown →
                                              ~3.0 for 10M-video songs)
    weight  = base + bonus

Usage
-----
    from chartmetric_enricher import build_cm_training_songs, load_cm_training_songs

    df = build_cm_training_songs()    # fetch lyrics for CM trending songs
    df = load_cm_training_songs()     # load from disk if already built
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd

from config import CM_TRAINING_CSV
_DATA_DIR = Path(__file__).parent / "data"


# ── Load raw Chartmetric data ─────────────────────────────────────────────────

def load_cm_trending() -> pd.DataFrame:
    """
    Merge cm_tiktok_sounds.csv + cm_tiktok_tracks.csv.
    Returns DataFrame with: song, artist, video_count, source.
    Returns empty DataFrame if neither file exists (run Chartmetric refresh first).
    """
    frames = []
    for fname in ("cm_tiktok_sounds.csv", "cm_tiktok_tracks.csv"):
        path = _DATA_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)

    if not frames:
        print("[CM Enricher] No Chartmetric CSVs found — run Chartmetric Refresh in the app first.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with empty song/artist
    combined = combined[
        combined["song"].notna() & (combined["song"].astype(str).str.strip() != "")
        & combined["artist"].notna() & (combined["artist"].astype(str).str.strip() != "")
    ].copy()

    combined["video_count"] = (
        pd.to_numeric(combined.get("video_count", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Keep the row with the highest video_count per (song, artist)
    combined = (
        combined.sort_values("video_count", ascending=False)
        .drop_duplicates(subset=["song", "artist"])
        .reset_index(drop=True)
    )

    print(f"[CM Enricher] {len(combined)} unique CM trending songs loaded.")
    return combined


# ── Weight calculation ────────────────────────────────────────────────────────

def compute_cm_weight(video_count: int) -> float:
    """
    Training weight for a Chartmetric-sourced song.
    Base 2.0 (confirmed TikTok viral) + log bonus proportional to usage volume.

    Examples:
        0 videos     →  2.0
        50K videos   →  2.69  (log1p(1) ≈ 0.69)
        500K videos  →  4.30  (log1p(10) ≈ 2.30)
        5M  videos   →  6.61  (log1p(100) ≈ 4.61)
    """
    return 2.0 + math.log1p(max(video_count, 0) / 50_000)


# ── Build training-ready catalog ──────────────────────────────────────────────

def build_cm_training_songs(
    fetch_lyrics: bool = True,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    1. Load CM trending songs.
    2. Fetch Genius lyrics for songs that don't have them yet.
    3. Save to data/cm_training_songs.csv.
    4. Return DataFrame with columns:
           artist, song, tier, source, lyrics, video_count, train_weight

    tier = -1 throughout (TikTok viral ground truth).
    """
    trending = load_cm_trending()
    if trending.empty:
        return pd.DataFrame()

    catalog = trending[["artist", "song", "video_count", "source"]].copy()
    catalog["tier"]         = -1
    catalog["train_weight"] = catalog["video_count"].apply(compute_cm_weight)

    # Carry over lyrics fetched in a previous run
    catalog["lyrics"] = None
    if CM_TRAINING_CSV.exists():
        existing = pd.read_csv(CM_TRAINING_CSV)
        if "lyrics" in existing.columns:
            lyr_map = (
                existing.dropna(subset=["lyrics"])
                .assign(
                    _sk=lambda x: x["song"].str.lower().str.strip(),
                    _ak=lambda x: x["artist"].str.lower().str.strip(),
                )
                .set_index(["_sk", "_ak"])["lyrics"]
                .to_dict()
            )
            catalog["lyrics"] = catalog.apply(
                lambda r: lyr_map.get(
                    (r["song"].lower().strip(), r["artist"].lower().strip())
                ),
                axis=1,
            )

    if fetch_lyrics:
        missing_mask = (
            catalog["lyrics"].isna()
            | (catalog["lyrics"].astype(str).str.strip() == "")
        )
        total_missing = int(missing_mask.sum())
        print(f"[CM Enricher] Fetching lyrics for {total_missing}/{len(catalog)} songs …")

        if total_missing > 0:
            try:
                from genius_fetcher import fetch_lyrics, _make_client
                genius = _make_client()
            except Exception as e:
                print(f"  [!] Genius init failed: {e}")
                catalog.to_csv(CM_TRAINING_CSV, index=False)
                return catalog

            done = 0
            for idx, row in catalog[missing_mask].iterrows():
                done += 1
                artist = str(row["artist"]).strip()
                song   = str(row["song"]).strip()
                print(f"  [{done}/{total_missing}] {song} — {artist}")
                lyr = fetch_lyrics(artist, song, genius)
                catalog.at[idx, "lyrics"] = lyr or ""
                time.sleep(delay)

    found = (
        catalog["lyrics"].notna()
        & (catalog["lyrics"].astype(str).str.strip() != "")
    ).sum()
    print(f"[CM Enricher] Lyrics found: {found}/{len(catalog)}")

    catalog.to_csv(CM_TRAINING_CSV, index=False)
    print(f"[CM Enricher] Saved → {CM_TRAINING_CSV}")
    return catalog


def load_cm_training_songs() -> pd.DataFrame:
    """
    Load from disk if cm_training_songs.csv exists.
    Call build_cm_training_songs() first to populate it.
    """
    if CM_TRAINING_CSV.exists():
        df = pd.read_csv(CM_TRAINING_CSV)
        # Recompute weights in case video_count was updated
        if "video_count" in df.columns:
            df["train_weight"] = df["video_count"].apply(compute_cm_weight)
        else:
            df["train_weight"] = 2.0
        return df
    return pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_cm_training_songs()
    if not df.empty:
        print(f"\n{len(df)} CM training songs ready.")
        cols = ["song", "artist", "video_count", "train_weight", "tier"]
        have_lyr = df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")
        print(df[cols][have_lyr].head(20).to_string(index=False))

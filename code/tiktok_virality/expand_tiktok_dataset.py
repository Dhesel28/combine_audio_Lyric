"""
Expand TikTok Virality Dataset
===============================
Combines 5 data sources to build a much larger dataset:
  1. tiktok.csv          — 6,746 songs with audio features (TikTok playlists)
  2. TikTok_songs_2020   —   292 songs with audio features (TikTok 2020)
  3. playlist_2010to2023 — 2,400 songs with audio features (Spotify top hits)
  4. spotify-2023.csv    —   953 songs with audio features (top 2023 hits)
  5. spotify_2024        — 4,600 songs with TikTok View metrics (for labels)

Output:
  data/raw/tiktok_expanded.csv  — unified dataset for training
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_CSV  = RAW_DIR / "tiktok_expanded.csv"

AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ── 1. Load and clean existing tiktok.csv ─────────────────────────────────────
def load_tiktok_csv() -> pd.DataFrame:
    path = RAW_DIR / "tiktok_trending_tracks" / "tiktok.csv"
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="track_id")
    df["source"] = "tiktok_trending"
    df["_key"] = df["track_name"].apply(norm) + "|" + df["artist_name"].apply(norm)
    print(f"  tiktok.csv          : {len(df):,} tracks")
    return df


# ── 2. Load TikTok 2020 ───────────────────────────────────────────────────────
def load_tiktok_2020() -> pd.DataFrame:
    path = RAW_DIR / "tiktok_popular_2020" / "TikTok_songs_2020.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"track_pop": "popularity", "duration_ms": "duration"})
    df["genre"]  = "TIKTOK_2020"
    df["source"] = "tiktok_2020"
    df["_key"]   = df["track_name"].apply(norm) + "|" + df["artist_name"].apply(norm)
    # Ensure audio cols present
    for c in AUDIO_COLS:
        if c not in df.columns:
            df[c] = np.nan
    print(f"  TikTok_songs_2020   : {len(df):,} tracks")
    return df


# ── 3. Load Spotify Top Hits 2010-2023 ────────────────────────────────────────
def load_top_hits() -> pd.DataFrame:
    path = RAW_DIR / "spotify_top_hits" / "playlist_2010to2023.csv"
    df = pd.read_csv(path, encoding="latin1")
    df = df.rename(columns={"track_popularity": "popularity"})
    # Drop if already in tiktok (will dedup later)
    df["genre"]     = "SPOTIFY_TOP_HIT"
    df["source"]    = "top_hits_2010_2023"
    df["_key"]      = df["track_name"].apply(norm) + "|" + df["artist_name"].apply(norm)
    df["release_date"] = df["year"].astype(str)
    for c in AUDIO_COLS:
        if c not in df.columns:
            df[c] = np.nan
    print(f"  top_hits 2010-2023  : {len(df):,} tracks")
    return df


# ── 4. Load Spotify 2023 (top chart songs) ────────────────────────────────────
def load_spotify_2023() -> pd.DataFrame:
    path = RAW_DIR / "spotify_2023" / "spotify-2023.csv"
    df   = pd.read_csv(path, encoding="latin1")

    # Rename to standard columns
    df = df.rename(columns={
        "track_name":       "track_name",
        "artist(s)_name":  "artist_name",
        "released_year":   "release_year",
    })

    # Audio features are on 0-100 scale → convert to 0-1
    col_map = {
        "danceability_%": "danceability",
        "valence_%":      "valence",
        "energy_%":       "energy",
        "acousticness_%": "acousticness",
        "instrumentalness_%": "instrumentalness",
        "liveness_%":     "liveness",
        "speechiness_%":  "speechiness",
        "bpm":            "tempo",
    }
    for old, new in col_map.items():
        if old in df.columns:
            val = pd.to_numeric(df[old], errors="coerce")
            if new != "tempo":  # tempo stays as-is
                val = val / 100.0
            df[new] = val

    if "key" not in df.columns:
        df["key"] = np.nan
    if "loudness" not in df.columns:
        df["loudness"] = np.nan
    if "mode" not in df.columns:
        df["mode"] = df.get("mode", np.nan)

    # Streams as popularity proxy (log-scaled)
    streams = pd.to_numeric(df.get("streams", pd.Series(dtype=float)), errors="coerce")
    if streams.notna().sum() > 0:
        df["popularity"] = (np.log1p(streams) / np.log1p(streams.max()) * 100).round(1)
    else:
        df["popularity"] = 50.0

    df["release_date"] = df.get("release_year", 2023).astype(str)
    df["genre"]        = "SPOTIFY_2023"
    df["source"]       = "spotify_2023"
    df["_key"]         = df["track_name"].apply(norm) + "|" + df["artist_name"].apply(norm)

    for c in AUDIO_COLS:
        if c not in df.columns:
            df[c] = np.nan

    print(f"  spotify-2023        : {len(df):,} tracks")
    return df


# ── 5. Load 2024 dataset for TikTok virality labels ──────────────────────────
def load_spotify_2024_labels() -> pd.DataFrame:
    """Returns a dict: _key → tiktok_views for label improvement."""
    path = RAW_DIR / "spotify_2024" / "Most Streamed Spotify Songs 2024.csv"
    df   = pd.read_csv(path, encoding="latin1")

    df["_key"] = df["Track"].apply(norm) + "|" + df["Artist"].apply(norm)

    def clean_num(col):
        return pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        )

    df["tiktok_views"] = clean_num("TikTok Views")
    df["tiktok_posts"] = clean_num("TikTok Posts")
    df["sp_popularity"] = pd.to_numeric(df.get("Spotify Popularity"), errors="coerce")

    print(f"  spotify_2024        : {len(df):,} tracks  "
          f"({df['tiktok_views'].notna().sum():,} have TikTok Views)")
    return df[["_key", "tiktok_views", "tiktok_posts", "sp_popularity"]].dropna(subset=["_key"])


# ── Merge and expand ──────────────────────────────────────────────────────────
def build_expanded() -> pd.DataFrame:
    print("=" * 60)
    print("Loading data sources...")
    print("-" * 60)

    tiktok   = load_tiktok_csv()
    t2020    = load_tiktok_2020()
    top_hits = load_top_hits()
    sp_2023  = load_spotify_2023()
    sp_2024  = load_spotify_2024_labels()

    # Stack all sources, unifying columns
    common = ["track_name", "artist_name", "release_date", "popularity",
              "genre", "source", "_key"] + AUDIO_COLS

    def safe_select(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Select only columns that exist; fill missing with NaN."""
        out = {}
        for c in cols:
            out[c] = df[c] if c in df.columns else np.nan
        return pd.DataFrame(out, index=df.index)

    frames = [safe_select(df, common)
              for df in [tiktok, t2020, top_hits, sp_2023]]

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nBefore dedup : {len(merged):,} rows")

    # Deduplicate by _key (keep first occurrence, prefer tiktok_trending)
    merged = merged.drop_duplicates(subset="_key", keep="first")
    print(f"After dedup  : {len(merged):,} unique songs")

    # Attach TikTok labels from 2024 dataset
    sp2024_map = sp_2024.set_index("_key")[["tiktok_views", "tiktok_posts", "sp_popularity"]]
    merged = merged.join(sp2024_map, on="_key", how="left")

    tiktok_matched = merged["tiktok_views"].notna().sum()
    print(f"Songs with TikTok Views from 2024 data: {tiktok_matched:,}")

    return merged


# ── Create viral labels ────────────────────────────────────────────────────────
def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Viral = 1 if:
      - TikTok Views available AND in top 33% of all songs with TikTok data
      - OR TikTok Views not available AND popularity >= 70
    Not viral = 0 if:
      - TikTok Views available AND in bottom 50% of TikTok data
      - OR TikTok Views not available AND popularity < 50
    Drop: ambiguous middle.
    """
    df = df.copy()

    # TikTok-based threshold (top 33% = viral)
    tv = df["tiktok_views"].dropna()
    tiktok_viral_threshold   = tv.quantile(0.67) if len(tv) > 0 else np.inf
    tiktok_notviral_threshold = tv.quantile(0.50) if len(tv) > 0 else 0

    pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(50)
    tv  = df["tiktok_views"]

    has_tt = tv.notna()

    # Viral conditions
    viral_tt  = has_tt & (tv >= tiktok_viral_threshold)
    viral_pop = (~has_tt) & (pop >= 70)
    # Not viral conditions
    nv_tt  = has_tt & (tv < tiktok_notviral_threshold)
    nv_pop = (~has_tt) & (pop < 50)

    df["viral"] = np.select(
        [viral_tt | viral_pop, nv_tt | nv_pop],
        [1,                    0],
        default=-1,
    )

    print(f"\nLabel distribution:")
    print(f"  Viral (1)    : {(df['viral']==1).sum():,}")
    print(f"  Not viral (0): {(df['viral']==0).sum():,}")
    print(f"  Ambiguous    : {(df['viral']==-1).sum():,}  (dropped for training)")
    print(f"  TikTok threshold — viral: >{tiktok_viral_threshold:,.0f}  not-viral: <{tiktok_notviral_threshold:,.0f} views")

    return df


# ── Audit audio feature coverage ─────────────────────────────────────────────
def audit(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("AUDIO FEATURE COVERAGE")
    print("=" * 60)
    for col in AUDIO_COLS:
        n = df[col].notna().sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct / 5)
        print(f"  {col:<20} {n:>5,}/{len(df):,}  ({pct:5.1f}%)  {bar}")

    print(f"\nSOURCE BREAKDOWN")
    print(df.groupby("source").size().rename("count").to_string())

    labeled = df[df["viral"] != -1]
    print(f"\nTraining-ready rows (labeled): {len(labeled):,}")
    audio_ok = labeled[AUDIO_COLS].notna().all(axis=1).sum()
    print(f"With all audio features      : {audio_ok:,}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = build_expanded()
    df = assign_labels(df)
    df.to_csv(OUT_CSV, index=False)
    audit(df)

    print("\n" + "=" * 60)
    print(f"Saved → {OUT_CSV}")
    print(f"Total songs : {len(df):,}  (was 6,746)")
    print(f"Labeled     : {(df['viral'] != -1).sum():,}")
    print("=" * 60)
    print("\nNext: run train_model.py to retrain on expanded dataset.")


if __name__ == "__main__":
    main()

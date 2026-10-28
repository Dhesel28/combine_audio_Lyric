"""
Build training dataset for the viral section ML model.

Sources:
  A. TikTok viral songs (15 songs) — fetched live from Genius,
     label = highest text_virality_score section
  B. Tatiana_cyanite.csv (16 songs) — Cyanite audio features + Genius lyrics,
     label = highest NLP viral score section
  C. data/cross_platform_analysis.csv (10 annotated songs) —
     label = section with highest combined_viral_score

Output: data/training_data.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure we can import siblings
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from nlp_viral_detector import (
    _parse_sections, _section_type, score_section as _nlp_score, WEIGHTS,
)

NLP_FEATURE_NAMES: list = list(WEIGHTS.keys())

ROOT = Path(__file__).parent.parent   # SUPA_Spring/

# ── TikTok viral songs for Source C ───────────────────────────────────────────
TIKTOK_TRAINING_SONGS = [
    ("Espresso",           "Sabrina Carpenter"),
    ("APT.",               "ROSÉ"),
    ("Die With A Smile",   "Lady Gaga"),
    ("Taste",              "Sabrina Carpenter"),
    ("Beautiful Things",   "Benson Boone"),
    ("Too Sweet",          "Hozier"),
    ("Cruel Summer",       "Taylor Swift"),
    ("Paint The Town Red", "Doja Cat"),
    ("Luther",             "Kendrick Lamar"),
    ("Supernatural",       "NewJeans"),
    ("Birds of a Feather", "Billie Eilish"),
    ("Levitating",         "Dua Lipa"),
    ("As It Was",          "Harry Styles"),
    ("Blinding Lights",    "The Weeknd"),
    ("Stay",               "The Kid LAROI"),
]



# ── Source B: Cross-platform annotated songs (10 songs) ───────────────────────

def _build_source_b() -> list[dict]:
    """Load cross_platform_analysis.csv and re-extract features with labels."""
    csv_path = DATA_DIR / "cross_platform_analysis.csv"
    if not csv_path.exists():
        print(f"[!] cross_platform_analysis.csv not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    print(f"  Source B: {len(df)} sections across {df['song'].nunique()} songs")

    rows = []

    for song_name, group in df.groupby("song"):
        group = group.reset_index(drop=True)
        n = len(group)
        artist = group["artist"].iloc[0] if "artist" in group.columns else "Unknown"

        # Label: section with highest combined_viral_score
        score_col = "combined_viral_score" if "combined_viral_score" in group.columns else "text_virality_score"
        best_score = group[score_col].max()

        for pos, (_, row) in enumerate(group.iterrows()):
            label = row["section_label"]
            stype = row["section_type"] if "section_type" in row else _section_type(label)
            full_lyrics = row.get("full_lyrics", row.get("lyrics_preview", ""))
            if pd.isna(full_lyrics):
                full_lyrics = ""

            lines = [l.strip() for l in str(full_lyrics).split("\n") if l.strip()]
            if not lines:
                preview = str(row.get("lyrics_preview", ""))
                lines = [l.strip() for l in preview.split("/") if l.strip()]
            if not lines:
                continue

            # NLP features — pass section text as raw_lyrics (no full song available)
            section_raw = "\n".join(lines)
            scores = _nlp_score(lines, pos, n, section_raw)
            nlp_feats = {sig: scores[sig] for sig in NLP_FEATURE_NAMES}

            rows.append({
                "song": song_name,
                "artist": artist,
                "section_label": label,
                "section_type": stype,
                "text_virality_score": scores["viral_score"],
                **nlp_feats,
                "is_viral": 1 if row[score_col] == best_score else 0,
                "source": "annotated",
            })

    print(f"  Source B: {len(rows)} sections")
    return rows


# ── Source C: TikTok viral songs ──────────────────────────────────────────────

def _build_source_c() -> list:
    """
    Source C: 15 popular TikTok viral songs fetched live from Genius.
    Label = section with highest text_virality_score.
    """
    import lyricsgenius
    import time
    from config import GENIUS_TOKEN

    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        verbose=False,
    )
    genius.remove_section_headers = False

    rows = []
    print(f"  Fetching {len(TIKTOK_TRAINING_SONGS)} TikTok songs from Genius...")

    for title, artist in TIKTOK_TRAINING_SONGS:
        try:
            song = genius.search_song(title, artist)
            if not song:
                print(f"    [skip] {title} — not found")
                continue

            sections = _parse_sections(song.lyrics)
            if not sections:
                continue

            labels = list(sections.keys())
            n = len(labels)
            section_scores = []

            for idx, label in enumerate(labels):
                lines = sections[label]
                stype = _section_type(label)
                scores = _nlp_score(lines, idx, n, song.lyrics)
                nlp_feats = {sig: scores[sig] for sig in NLP_FEATURE_NAMES}
                section_scores.append({
                    "song":               song.title,
                    "artist":             artist,
                    "section_label":      label,
                    "section_type":       stype,
                    "text_virality_score": scores["viral_score"],
                    "source":             "tiktok_viral",
                    **nlp_feats,
                })

            if not section_scores:
                continue

            best = max(s["text_virality_score"] for s in section_scores)
            for s in section_scores:
                s["is_viral"] = 1 if s["text_virality_score"] == best else 0
                rows.append(s)

            print(f"    ✓ {title} ({artist}) — {n} sections")
            time.sleep(0.5)

        except Exception as e:
            print(f"    [error] {title}: {e}")
            continue

    print(f"  Source C (TikTok): {len(rows)} sections from {len([r for r in rows if r['is_viral']==1])} songs")
    return rows


# ── Source D: Tatiana DeMaria (Cyanite) ───────────────────────────────────────

def _build_source_d() -> list:
    """
    Source D: Tatiana_cyanite.csv — 16 songs with Cyanite audio features.
    Lyrics fetched from Genius, NLP scored, label = highest viral score.
    """
    try:
        from tatiana_analyzer import analyze_tatiana_songs
        tatiana_df = analyze_tatiana_songs(verbose=True)
        if tatiana_df.empty:
            return []

        rows = []
        for _, row in tatiana_df.iterrows():
            stype = str(row.get("section_type", "Other"))
            # tatiana_df already has the 12 NLP signal columns from score_section()
            nlp_feats = {sig: float(row.get(sig, 0.0)) for sig in NLP_FEATURE_NAMES}
            rows.append({
                "song":               str(row.get("song", "")),
                "artist":             "Tatiana DeMaria",
                "section_label":      str(row.get("section_label", "")),
                "section_type":       stype,
                "text_virality_score": float(row.get("viral_score", 0)),
                "source":             "tatiana_cyanite",
                **nlp_feats,
                "is_viral": int(row.get("is_viral", 0)),
            })

        print(f"  Source D (Tatiana): {len(rows)} sections from {tatiana_df['song'].nunique()} songs")
        return rows

    except Exception as e:
        print(f"  [!] Source D (Tatiana) error: {e}")
        return []


# ── Source E: Full Tatiana DeMaria lyrics (no Genius API) ─────────────────────

def _build_source_e() -> list:
    """
    Source E: full_tatiana_demaria_lyrics.csv — 6 songs with raw lyrics.
    Sections auto-labeled via sliding-window repetition detector.
    Label = section with highest text_virality_score.
    """
    try:
        from tatiana_analyzer import analyze_full_lyrics_csv
        df = analyze_full_lyrics_csv(verbose=True)
        if df.empty:
            return []

        rows = []
        for _, row in df.iterrows():
            nlp_feats = {sig: float(row.get(sig, 0.0)) for sig in NLP_FEATURE_NAMES}
            rows.append({
                "song":               str(row.get("song", "")),
                "artist":             "Tatiana DeMaria",
                "section_label":      str(row.get("section_label", "")),
                "section_type":       str(row.get("section_type", "Other")),
                "text_virality_score": float(row.get("viral_score", 0.0)),
                "source":             "tatiana_full_lyrics",
                **nlp_feats,
                "is_viral":           int(row.get("is_viral", 0)),
            })

        print(f"  Source E (Tatiana full lyrics): {len(rows)} sections")
        return rows

    except Exception as e:
        print(f"  [!] Source E (Tatiana full lyrics) error: {e}")
        return []


# ── Main ───────────────────────────────────────────────────────────────────────

def build() -> pd.DataFrame:
    print("Building training data...")

    print("\n[Source A] TikTok viral songs (live Genius fetch)")
    rows_a = _build_source_c()

    print("\n[Source B] Tatiana DeMaria — Cyanite audio + Genius lyrics")
    rows_b = _build_source_d()

    print("\n[Source C] Cross-platform annotated songs")
    rows_c = _build_source_b()

    print("\n[Source E] Tatiana DeMaria — full lyrics (no Genius API)")
    rows_e = _build_source_e()

    all_rows = rows_a + rows_b + rows_c + rows_e
    if not all_rows:
        print("[!] No training data generated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Ensure all 12 NLP feature columns are present (fill zeros for any missing)
    for feat in NLP_FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0.0

    # Reorder: metadata cols + NLP features + label
    meta_cols = ["song", "artist", "section_label", "section_type",
                 "text_virality_score", "source"]
    available_meta = [c for c in meta_cols if c in df.columns]
    df = df[available_meta + NLP_FEATURE_NAMES + ["is_viral"]]

    n_songs = df["song"].nunique()
    n_positives = df["is_viral"].sum()
    n_sections = len(df)

    print(f"\n{'='*50}")
    print(f"Training data: {n_sections} sections, {n_songs} songs, {n_positives} positives")
    print(f"  Positive rate: {n_positives/n_sections:.1%}")
    if "source" in df.columns:
        src_counts = df.groupby("source").size()
        for src, cnt in src_counts.items():
            print(f"  [{src}]: {cnt} sections")
    print(f"{'='*50}")

    out = DATA_DIR / "training_data.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")

    return df


if __name__ == "__main__":
    df = build()
    if not df.empty:
        print("\nSample:")
        print(df[["song", "section_label", "section_type", "is_viral", "source"]].head(10).to_string(index=False))

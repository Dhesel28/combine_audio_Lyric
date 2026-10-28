"""
Lyric-based Song Segmentation Model
=====================================
Clusters TikTok songs into meaningful groups using K-means on lyrical features.

Features (18 dimensions):
  Structural   : avg_energy, avg_positive, avg_negative, avg_brevity,
                 repetition_rate, max_repeats_norm, hook_line_ratio,
                 rhyme_density, vocab_richness, line_count_norm, duration_norm
  Thematic     : theme_love_romance, theme_heartbreak, theme_confidence,
                 theme_party_energy, theme_nostalgia, theme_struggle_grind,
                 theme_spiritual

Input  : data/tiktok_new_to_dhesel_lyrics.csv  (artist, title, lyrics, duration_sec)
Output : data/tiktok_new_segments.csv           (+ cluster_id, cluster_name, all features)
         models/segmentation_model.pkl          (KMeans + scaler for future prediction)

Usage:
    python segmentation_model.py               # auto-select k via silhouette score
    python segmentation_model.py --k 5         # force 5 clusters
    python segmentation_model.py --input path/to/file.csv
    python segmentation_model.py --predict song_title artist  # assign a new song
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
HD_DIR     = Path("/Volumes/Dhesel Driv /Work")
LYRICS_CSV = HD_DIR / "tiktok_new_to_dhesel_lyrics.csv"
OUTPUT_CSV = HD_DIR / "tiktok_new_segments.csv"
MODEL_PATH = MODELS_DIR / "segmentation_model.pkl"

# ── Word sets (mirrors existing codebase) ─────────────────────────────────────

ENERGY_WORDS = {
    "fire","lit","hype","wild","crazy","turnt","vibe","bounce","drop","bang",
    "banger","loud","scream","jump","move","go","fast","hard","rage","beast",
    "king","queen","boss","flex","drip","swag","pop","rock","hit","rise",
    "shine","burn","blaze","fly","rush","push","break","unstoppable","epic",
}
POSITIVE_WORDS = {
    "love","happy","joy","smile","beautiful","wonderful","amazing","great",
    "good","best","perfect","bright","warm","sweet","dream","hope","win",
    "glory","bless","grace","light","gold","free","peace","together",
    "forever","always","paradise","heaven",
}
NEGATIVE_WORDS = {
    "hate","pain","cry","tears","hurt","break","lost","alone","dark","cold",
    "dead","fall","fail","lie","cheat","betray","gone","miss","sad","empty",
    "numb","sorrow","grief","regret","fear","wrong","bad","worst","hell",
    "die","kill","blood",
}

THEME_CLUSTERS: dict[str, set[str]] = {
    "love_romance":   {"love","heart","kiss","baby","darling","forever","together",
                       "hold","feel","need","want","close","touch","arms","smile","eyes","beautiful"},
    "heartbreak":     {"break","cry","tears","goodbye","miss","lost","alone","hurt","pain",
                       "leave","gone","never","regret","mistake","cold","empty","hollow"},
    "confidence":     {"boss","king","queen","top","crown","flex","drip","shine","win","best",
                       "rich","money","clout","power","run","number","only"},
    "party_energy":   {"party","dance","night","vibe","fire","lit","move","bounce","jump",
                       "turn","hype","loud","crowd","floor","song","beat","bass"},
    "nostalgia":      {"remember","childhood","memory","back","yesterday","old","time","past",
                       "used","once","when","still","home","young","before"},
    "struggle_grind": {"grind","hustle","work","fight","rise","push","overcome","hard","real",
                       "sweat","earn","built","made","prove","never","quit"},
    "spiritual":      {"god","pray","faith","bless","soul","grace","heaven","holy","spirit",
                       "lord","amen","mercy","divine","saved"},
}

FEATURE_NAMES = [
    "avg_energy", "avg_positive", "avg_negative", "avg_brevity",
    "repetition_rate", "max_repeats_norm", "hook_line_ratio",
    "rhyme_density", "vocab_richness", "line_count_norm", "duration_norm",
    "theme_love_romance", "theme_heartbreak", "theme_confidence",
    "theme_party_energy", "theme_nostalgia", "theme_struggle_grind",
    "theme_spiritual",
]

_SECTION_RE   = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)
_HOOK_SECS    = {"chorus", "hook", "bridge", "refrain", "pre-chorus", "prechorus", "drop"}


# ── Feature extraction ─────────────────────────────────────────────────────────

def _last3(line: str) -> str:
    words = line.strip().split()
    return words[-1].lower()[-3:] if words else ""


def _rhymes(a: str, b: str) -> bool:
    ea, eb = _last3(a), _last3(b)
    return len(ea) >= 2 and ea == eb


def _parse_lines(lyrics: str) -> list[dict]:
    current_sec = "intro"
    entries: list[dict] = []
    for raw in lyrics.split("\n"):
        s = raw.strip()
        if not s:
            continue
        m = _SECTION_RE.match(s)
        if m:
            current_sec = re.sub(r'\s*\d+$', '', m.group(1).lower().strip())
        else:
            entries.append({"section": current_sec, "text": s})
    return entries


def extract_features(lyrics: str, duration_sec: float = 0.0) -> dict | None:
    """
    Extract 18-dimensional feature vector from raw lyrics string.
    Returns None if lyrics are too short or malformed.
    """
    if not lyrics or not isinstance(lyrics, str) or len(lyrics.strip()) < 30:
        return None

    entries = _parse_lines(lyrics)
    if len(entries) < 5:
        return None

    texts  = [e["text"] for e in entries]
    total  = len(texts)
    counts = Counter(t.lower().strip() for t in texts)
    max_r  = max(counts.values())

    energy_v, pos_v, neg_v, brev_v = [], [], [], []
    for t in texts:
        words = re.sub(r"[^\w\s]", " ", t.lower()).split()
        wc    = max(len(words), 1)
        ws    = set(words)
        energy_v.append(min(len(ws & ENERGY_WORDS)   / wc * 5, 1.0))
        pos_v.append(   min(len(ws & POSITIVE_WORDS) / wc * 5, 1.0))
        neg_v.append(   min(len(ws & NEGATIVE_WORDS) / wc * 5, 1.0))
        n = len(words)
        if 4 <= n <= 9:
            brev_v.append(1.0)
        elif n < 4:
            brev_v.append(0.4 + n * 0.1)
        else:
            brev_v.append(max(0.0, 1.0 - (n - 9) * 0.07))

    hook_lines  = sum(1 for e in entries if e["section"] in _HOOK_SECS)
    repeated    = sum(1 for t in texts if counts[t.lower().strip()] > 1)
    rhyme_pairs = sum(1 for i in range(total - 1) if _rhymes(texts[i], texts[i + 1]))
    all_words   = re.sub(r"[^\w\s]", " ", lyrics.lower()).split()
    vocab_rich  = len(set(all_words)) / max(len(all_words), 1)

    feats: dict[str, float] = {
        "avg_energy":       float(np.mean(energy_v)),
        "avg_positive":     float(np.mean(pos_v)),
        "avg_negative":     float(np.mean(neg_v)),
        "avg_brevity":      float(np.mean(brev_v)),
        "repetition_rate":  repeated / total,
        "max_repeats_norm": min(max_r / 10.0, 1.0),
        "hook_line_ratio":  hook_lines / total,
        "rhyme_density":    rhyme_pairs / max(total - 1, 1),
        "vocab_richness":   vocab_rich,
        "line_count_norm":  min(total / 80.0, 1.0),
        "duration_norm":    min(duration_sec / 300.0, 1.0) if duration_sec > 0 else 0.5,
    }

    words_set = set(all_words)
    for theme, theme_words in THEME_CLUSTERS.items():
        feats[f"theme_{theme}"] = len(words_set & theme_words) / len(theme_words)

    return feats


# ── Cluster naming ─────────────────────────────────────────────────────────────

def _name_cluster(center: pd.Series) -> str:
    """Assign a human-readable label based on dominant features of a cluster center."""
    themes    = {k: float(center[f"theme_{k}"]) for k in THEME_CLUSTERS}
    top_theme = max(themes, key=themes.get)
    energy    = float(center["avg_energy"])
    positive  = float(center["avg_positive"])
    negative  = float(center["avg_negative"])
    hook      = float(center["hook_line_ratio"])
    rep       = float(center["repetition_rate"])

    if top_theme == "party_energy" or (energy > 0.04 and hook > 0.25):
        return "High-Energy / Party"
    elif top_theme == "heartbreak" or (negative > 0.05 and positive < 0.04):
        return "Emotional / Heartbreak"
    elif top_theme == "love_romance" or (positive > 0.06 and negative < 0.04):
        return "Love & Romance"
    elif top_theme == "confidence":
        return "Confidence / Flex"
    elif top_theme == "nostalgia":
        return "Nostalgic / Story-driven"
    elif top_theme == "struggle_grind":
        return "Hustle & Grind"
    elif top_theme == "spiritual":
        return "Inspirational / Spiritual"
    elif rep > 0.5 and hook > 0.30:
        return "Hook-Heavy / Repetitive"
    else:
        return "Mixed / General"


def _dedupe_names(raw: list[str]) -> dict[int, str]:
    """Ensure every cluster has a unique label."""
    counts: Counter = Counter(raw)
    seen:   dict[str, int] = {}
    result: dict[int, str] = {}
    for i, name in enumerate(raw):
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            result[i] = f"{name} {seen[name]}"
        else:
            result[i] = name
    return result


# ── Predict helper (for new songs after training) ─────────────────────────────

def predict_cluster(lyrics: str, duration_sec: float = 0.0) -> dict | None:
    """
    Assign a pre-trained cluster to a new song's lyrics.
    Loads saved model from MODEL_PATH.
    """
    if not MODEL_PATH.exists():
        print(f"No saved model at {MODEL_PATH}. Run segmentation_model.py first.")
        return None

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    feats = extract_features(lyrics, duration_sec)
    if feats is None:
        return None

    X = bundle["scaler"].transform([[ feats[n] for n in bundle["feature_names"] ]])
    cid = int(bundle["kmeans"].predict(X)[0])
    return {
        "cluster_id":   cid,
        "cluster_name": bundle["cluster_names"][cid],
        "features":     feats,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(lyrics_csv: Path = LYRICS_CSV, n_clusters: int | None = None) -> pd.DataFrame:
    if not lyrics_csv.exists():
        print(f"Lyrics file not found: {lyrics_csv}")
        print("Run: python fetch_new_lyrics.py")
        sys.exit(1)

    df = pd.read_csv(lyrics_csv)
    has_lyrics = df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")
    df_lyr     = df[has_lyrics].copy().reset_index(drop=True)
    print(f"Songs with lyrics : {len(df_lyr)} / {len(df)}\n")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("Extracting features...")
    records:   list[dict] = []
    valid_idx: list[int]  = []
    for i, row in df_lyr.iterrows():
        dur   = float(row.get("duration_sec") or 0)
        feats = extract_features(str(row["lyrics"]), dur)
        if feats:
            records.append(feats)
            valid_idx.append(i)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df_lyr)} processed")

    df_valid = df_lyr.iloc[valid_idx].copy().reset_index(drop=True)
    X_df     = pd.DataFrame(records, columns=FEATURE_NAMES)
    print(f"Features ready: {len(records)} songs\n")

    if len(records) < 4:
        print("Not enough songs with lyrics to cluster (need at least 4).")
        sys.exit(1)

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_df.values)

    # ── Find optimal k ────────────────────────────────────────────────────────
    if n_clusters is None:
        max_k      = min(9, len(X) // 10 + 1)
        best_k     = 4
        best_score = -1.0
        print(f"Finding optimal k (2..{max_k - 1})...")
        for k in range(2, max_k):
            km_    = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_ = km_.fit_predict(X)
            score_  = silhouette_score(X, labels_)
            marker  = " ←" if score_ > best_score else ""
            print(f"  k={k}  silhouette={score_:.4f}{marker}")
            if score_ > best_score:
                best_score = score_
                best_k     = k
        n_clusters = best_k
        print(f"\nBest k = {n_clusters}  (silhouette = {best_score:.4f})\n")

    # ── Final clustering ──────────────────────────────────────────────────────
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    centers_raw = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=FEATURE_NAMES,
    )
    raw_names   = [_name_cluster(centers_raw.iloc[i]) for i in range(n_clusters)]
    clust_names = _dedupe_names(raw_names)

    # ── Attach results ────────────────────────────────────────────────────────
    df_valid["cluster_id"]   = labels
    df_valid["cluster_name"] = [clust_names[l] for l in labels]
    for col in FEATURE_NAMES:
        df_valid[col] = [r[col] for r in records]

    # ── Save ──────────────────────────────────────────────────────────────────
    df_valid.to_csv(OUTPUT_CSV, index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "kmeans":        km,
            "scaler":        scaler,
            "cluster_names": clust_names,
            "feature_names": FEATURE_NAMES,
        }, f)

    print(f"Segments → {OUTPUT_CSV}")
    print(f"Model    → {MODEL_PATH}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 62)
    print("SEGMENT SUMMARY")
    print("=" * 62)
    for cid in range(n_clusters):
        subset    = df_valid[df_valid["cluster_id"] == cid]
        c         = centers_raw.iloc[cid]
        top_theme = max(THEME_CLUSTERS, key=lambda k: float(c[f"theme_{k}"]))
        print(f"\n[{clust_names[cid]}]  —  {len(subset)} songs")
        print(f"  energy={c['avg_energy']:.3f}  "
              f"positive={c['avg_positive']:.3f}  "
              f"negative={c['avg_negative']:.3f}")
        print(f"  hook_ratio={c['hook_line_ratio']:.3f}  "
              f"repetition={c['repetition_rate']:.3f}  "
              f"top_theme={top_theme}")
        examples = subset[["title", "artist"]].head(5)
        for _, r in examples.iterrows():
            print(f"    - {r['title']}  ({r['artist']})")
    print()

    return df_valid


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cluster TikTok songs by lyrical features")
    p.add_argument("--k",     type=int,  default=None,       help="Force number of clusters (default: auto)")
    p.add_argument("--input", type=Path, default=LYRICS_CSV,  help="CSV with a 'lyrics' column")
    args = p.parse_args()
    run(lyrics_csv=args.input, n_clusters=args.k)

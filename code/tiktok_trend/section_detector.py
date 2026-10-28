"""
Section Detector — Supervised ML Model
=======================================
Learns what Chorus, Verse, Bridge, Intro, and Outro blocks look like
from 500+ Genius-labeled songs, then predicts section labels for any
new lyrics — even when no [Chorus]/[Verse] markers exist.

Key insight: the model learns BLOCK-LEVEL features, not song context.
The most powerful signal is block repetition — the chorus is almost
always the block that repeats most in a song.

Features per block (12 dimensions):
  block_repeat_count    how many times this exact block appears in the song
  similarity_score      max similarity to any other block (chorus repeats = high)
  position_start        where block starts in song (0=beginning, 1=end)
  position_mid          position of block centre
  block_length          number of lines
  avg_line_length       average words per line
  unique_line_ratio     unique lines / total lines (low = very repetitive)
  rhyme_density         fraction of adjacent pairs that rhyme
  avg_energy            energy word density
  avg_positive          positive sentiment density
  avg_negative          negative sentiment density
  avg_brevity           average brevity score (peak at 4-9 words)

Labels (simplified to 5 classes):
  chorus      [Chorus], [Hook], [Refrain], [Drop]
  verse       [Verse]
  bridge      [Bridge], [Breakdown], [Interlude]
  intro       [Intro], [Pre-Chorus], [Pre-Hook]
  outro       [Outro], [Coda], [Tag]

Usage:
    python section_detector.py --train          # train and evaluate
    python section_detector.py --predict        # predict on unlabeled songs
    python section_detector.py --song "Title" --artist "Artist"
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
HD_DIR     = Path("/Volumes/Dhesel Driv /Work")

LYRICS_CSV  = HD_DIR / "tiktok_new_to_dhesel_lyrics.csv"
CATALOG_CSV = DATA_DIR / "catalog_with_lyrics.csv"
MODEL_PATH  = MODELS_DIR / "section_detector.pkl"
OUTPUT_CSV  = HD_DIR / "tiktok_section_predictions.csv"

# ── Word sets ──────────────────────────────────────────────────────────────────
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

# ── Label normalisation ────────────────────────────────────────────────────────
_LABEL_MAP = {
    "chorus":     "chorus",
    "hook":       "chorus",
    "refrain":    "chorus",
    "drop":       "chorus",
    "verse":      "verse",
    "bridge":     "bridge",
    "breakdown":  "bridge",
    "interlude":  "bridge",
    "intro":      "intro",
    "pre-chorus": "intro",
    "prechorus":  "intro",
    "pre-hook":   "intro",
    "outro":      "outro",
    "coda":       "outro",
    "tag":        "outro",
    "outro/outro":"outro",
}

_SECTION_RE = re.compile(r'^\[([^\]]+)\]$', re.IGNORECASE)
FEATURE_NAMES = [
    "block_repeat_count", "similarity_score",
    "position_start", "position_mid",
    "block_length", "avg_line_length", "unique_line_ratio",
    "rhyme_density", "avg_energy", "avg_positive", "avg_negative", "avg_brevity",
]


def _norm_label(raw: str) -> str | None:
    base = re.sub(r'\s*\d+$', '', raw.lower().strip())
    # strip "artist name:" prefix  e.g. "intro: drake" → "intro"
    base = re.split(r'\s*:', base)[0].strip()
    return _LABEL_MAP.get(base)


def _last3(line: str) -> str:
    words = line.strip().split()
    return words[-1].lower()[-3:] if words else ""


def _rhymes(a: str, b: str) -> bool:
    ea, eb = _last3(a), _last3(b)
    return len(ea) >= 2 and ea == eb


def _block_sig(lines: list[str]) -> str:
    """Normalised fingerprint of a block for repetition detection."""
    return " | ".join(re.sub(r"[^\w\s]", "", l.lower()).strip() for l in lines)


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_block_features(
    lines:       list[str],
    block_idx:   int,
    all_blocks:  list[list[str]],
    total_lines: int,
    start_line:  int,
) -> dict:
    """
    Compute 12 features for a single block relative to its song context.

    Parameters
    ----------
    lines       : lines belonging to this block
    block_idx   : index of this block among all_blocks
    all_blocks  : every block in the song (for repetition + similarity)
    total_lines : total lyric line count in the song
    start_line  : first line index of this block (0-indexed)
    """
    n_lines = len(lines)
    mid_line = start_line + n_lines / 2

    # ── Repetition: how many blocks in this song are identical ────────────────
    sig  = _block_sig(lines)
    sigs = [_block_sig(b) for b in all_blocks]
    block_repeat = sigs.count(sig)                        # incl. itself

    # ── Similarity: max Jaccard with any other block ──────────────────────────
    sims = [_jaccard(sig, s) for i, s in enumerate(sigs) if i != block_idx]
    sim_score = max(sims) if sims else 0.0

    # ── Line-level metrics ────────────────────────────────────────────────────
    lengths, energy_v, pos_v, neg_v, brev_v = [], [], [], [], []
    for t in lines:
        words = re.sub(r"[^\w\s]", " ", t.lower()).split()
        wc    = max(len(words), 1)
        ws    = set(words)
        lengths.append(wc)
        energy_v.append(min(len(ws & ENERGY_WORDS)   / wc * 5, 1.0))
        pos_v.append(   min(len(ws & POSITIVE_WORDS) / wc * 5, 1.0))
        neg_v.append(   min(len(ws & NEGATIVE_WORDS) / wc * 5, 1.0))
        n = len(words)
        if 4 <= n <= 9:    brev_v.append(1.0)
        elif n < 4:        brev_v.append(0.4 + n * 0.1)
        else:              brev_v.append(max(0.0, 1.0 - (n - 9) * 0.07))

    rhyme_pairs  = sum(1 for i in range(n_lines - 1) if _rhymes(lines[i], lines[i+1]))
    unique_ratio = len(set(l.lower().strip() for l in lines)) / n_lines

    return {
        "block_repeat_count": float(block_repeat),
        "similarity_score":   float(sim_score),
        "position_start":     start_line / max(total_lines - 1, 1),
        "position_mid":       mid_line   / max(total_lines - 1, 1),
        "block_length":       float(n_lines),
        "avg_line_length":    float(np.mean(lengths)),
        "unique_line_ratio":  float(unique_ratio),
        "rhyme_density":      rhyme_pairs / max(n_lines - 1, 1),
        "avg_energy":         float(np.mean(energy_v)),
        "avg_positive":       float(np.mean(pos_v)),
        "avg_negative":       float(np.mean(neg_v)),
        "avg_brevity":        float(np.mean(brev_v)),
    }


# ── Lyrics → labeled blocks ────────────────────────────────────────────────────

def parse_labeled_song(lyrics: str) -> list[dict]:
    """
    Parse a Genius-labeled song into blocks.
    Returns list of {label, lines, features} — one dict per labeled block.
    Only returns blocks whose label maps to a known class.
    Returns [] if the song has no usable section markers.
    """
    current_label = None
    current_lines: list[str] = []
    raw_blocks: list[tuple[str, list[str]]] = []   # (label, lines)

    for raw in lyrics.split("\n"):
        s = raw.strip()
        m = _SECTION_RE.match(s)
        if m:
            if current_label and current_lines:
                raw_blocks.append((current_label, current_lines))
            raw_label = m.group(1)
            mapped    = _norm_label(raw_label)
            current_label = mapped   # None if unknown
            current_lines = []
        else:
            if s and current_label:
                current_lines.append(s)

    if current_label and current_lines:
        raw_blocks.append((current_label, current_lines))

    # Filter to known labels only
    blocks = [(lbl, lines) for lbl, lines in raw_blocks if lbl is not None]
    if not blocks:
        return []

    # Compute context
    all_block_lines = [b[1] for b in raw_blocks]   # use ALL blocks for context
    total_lines     = sum(len(b) for b in all_block_lines)
    start           = 0
    result          = []

    for lbl, lines in raw_blocks:
        if lbl is not None:
            feats = extract_block_features(lines, len(result), all_block_lines, total_lines, start)
            result.append({"label": lbl, "lines": lines, "features": feats})
        start += len(lines)

    return result


def parse_unlabeled_song(lyrics: str) -> list[dict]:
    """
    Parse a song WITHOUT section markers into blocks by splitting on blank lines.
    Returns list of {lines, features} — one dict per detected block.
    """
    # Split into paragraphs (blank line = block boundary)
    # Strip section marker lines so the model works on pure lyric content
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for raw in lyrics.split("\n"):
        s = raw.strip()
        if not s:
            if current:
                paragraphs.append(current)
                current = []
        elif _SECTION_RE.match(s):
            # marker line — treat as a block boundary, don't include the text
            if current:
                paragraphs.append(current)
                current = []
        else:
            current.append(s)
    if current:
        paragraphs.append(current)

    if not paragraphs:
        return []

    total_lines = sum(len(p) for p in paragraphs)
    start       = 0
    result      = []
    for i, lines in enumerate(paragraphs):
        feats = extract_block_features(lines, i, paragraphs, total_lines, start)
        result.append({"lines": lines, "features": feats})
        start += len(lines)

    return result


# ── Training data builder ──────────────────────────────────────────────────────

def build_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all labeled lyrics, parse into blocks, return (X, y, label_names).
    """
    all_records: list[dict] = []

    def load_csv(path: Path):
        if not path.exists():
            return
        df = pd.read_csv(path)
        col = "lyrics"
        if col not in df.columns:
            return
        for _, row in df.iterrows():
            lyr = str(row.get(col, "") or "")
            if not lyr.strip():
                continue
            blocks = parse_labeled_song(lyr)
            all_records.extend(blocks)

    load_csv(LYRICS_CSV)
    load_csv(CATALOG_CSV)

    if not all_records:
        raise ValueError("No labeled training blocks found.")

    X = np.array([[r["features"][f] for f in FEATURE_NAMES] for r in all_records])
    y = np.array([r["label"] for r in all_records])

    counts = Counter(y)
    print(f"Training blocks: {len(y)}")
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:12s}  {cnt:4d} blocks")

    return X, y, list(counts.keys())


# ── Train ──────────────────────────────────────────────────────────────────────

def train(save: bool = True) -> dict:
    print("Building training data...\n")
    X, y, labels = build_training_data()

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )

    # 5-fold stratified cross-validation
    print("\nCross-validation (5-fold stratified)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_sc, y_enc, cv=cv, scoring="accuracy")
    print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train on full data
    clf.fit(X_sc, y_enc)

    # Per-class report on training data (indicative)
    y_pred = clf.predict(X_sc)
    print("\nClassification report (training set):")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    # Feature importances
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:22s}  {imp:.3f}  {bar}")

    bundle = {
        "clf":           clf,
        "scaler":        scaler,
        "label_encoder": le,
        "feature_names": FEATURE_NAMES,
    }

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\nModel saved → {MODEL_PATH}")

    return bundle


# ── Predict ────────────────────────────────────────────────────────────────────

def predict_sections(lyrics: str, bundle: dict | None = None) -> list[dict]:
    """
    Predict section labels for raw lyrics (with or without markers).
    Returns list of {predicted_label, confidence, lines}.
    """
    if bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No model at {MODEL_PATH}. Run: python section_detector.py --train")
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

    clf  = bundle["clf"]
    sc   = bundle["scaler"]
    le   = bundle["label_encoder"]

    blocks = parse_unlabeled_song(lyrics)
    if not blocks:
        return []

    X = np.array([[b["features"][f] for f in FEATURE_NAMES] for b in blocks])
    X_sc  = sc.transform(X)
    probs = clf.predict_proba(X_sc)
    preds = clf.predict(X_sc)

    results = []
    for i, block in enumerate(blocks):
        label      = le.inverse_transform([preds[i]])[0]
        confidence = float(probs[i].max())
        results.append({
            "predicted_label": label,
            "confidence":      round(confidence, 3),
            "lines":           block["lines"],
            "features":        block["features"],
        })

    return results


def format_prediction(results: list[dict], song: str = "", artist: str = "") -> str:
    """Pretty-print predicted sections."""
    out = []
    if song:
        out += ["=" * 56, f"  {song}" + (f"  —  {artist}" if artist else ""), "=" * 56, ""]

    for i, r in enumerate(results, 1):
        label = r["predicted_label"].upper()
        conf  = r["confidence"]
        rep   = int(r["features"]["block_repeat_count"])
        bar   = "█" * int(conf * 16) + "░" * (16 - int(conf * 16))
        out.append(f"  Block {i:02d}  [{label:10s}]  [{bar}]  {conf:.0%}  (repeats {rep}x)")
        for line in r["lines"][:3]:
            out.append(f"           {line}")
        if len(r["lines"]) > 3:
            out.append(f"           ... ({len(r['lines'])} lines total)")
        out.append("")

    return "\n".join(out)


# ── Batch prediction on unlabeled songs ───────────────────────────────────────

def predict_batch(bundle: dict | None = None) -> pd.DataFrame:
    """Predict sections for every song in the lyrics CSV."""
    df = pd.read_csv(LYRICS_CSV)
    has = df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip().ne(""))
    df_lyr = df[has].copy().reset_index(drop=True)

    if bundle is None:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

    rows = []
    for i, row in df_lyr.iterrows():
        results = predict_sections(str(row["lyrics"]), bundle)
        if not results:
            continue

        # Find the block predicted as chorus with highest confidence
        chorus_blocks = [r for r in results if r["predicted_label"] == "chorus"]
        best_chorus   = max(chorus_blocks, key=lambda r: r["confidence"]) if chorus_blocks else None

        section_summary = "  |  ".join(
            f"{r['predicted_label']}({r['confidence']:.0%})" for r in results
        )
        rows.append({
            "artist":          row.get("artist", ""),
            "title":           row.get("title", ""),
            "n_blocks":        len(results),
            "has_chorus":      bool(chorus_blocks),
            "chorus_conf":     best_chorus["confidence"]  if best_chorus else 0.0,
            "chorus_line":     best_chorus["lines"][0]    if best_chorus else "",
            "sections":        section_summary,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df_lyr)} done")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBatch predictions → {OUTPUT_CSV}")
    return out_df


# ── CLI ────────────────────────────────────────────────────────────────────────

def _find_lyrics(title: str, artist: str) -> str | None:
    if not LYRICS_CSV.exists():
        return None
    df = pd.read_csv(LYRICS_CSV)
    mask = (
        df["title"].str.lower().str.strip().str.contains(title.lower(), na=False)
        & df["artist"].str.lower().str.strip().str.contains(artist.lower(), na=False)
    )
    hits = df[mask & df["lyrics"].notna() & df["lyrics"].astype(str).str.strip().ne("")]
    return str(hits.iloc[0]["lyrics"]) if not hits.empty else None


def main():
    p = argparse.ArgumentParser(description="Section detector — train and predict")
    p.add_argument("--train",   action="store_true", help="Train model on labeled songs")
    p.add_argument("--predict", action="store_true", help="Batch predict on all unlabeled songs")
    p.add_argument("--song",    type=str, default="", help="Song title to analyze")
    p.add_argument("--artist",  type=str, default="", help="Artist name")
    args = p.parse_args()

    if args.train:
        train()
        return

    if args.predict:
        if not MODEL_PATH.exists():
            print("Model not found — training first...")
            train()
        predict_batch()
        return

    if args.song:
        if not MODEL_PATH.exists():
            print("Model not found — training first...")
            train()
        lyrics = _find_lyrics(args.song, args.artist)
        if not lyrics:
            print(f"Song '{args.song}' not found in dataset.")
            sys.exit(1)
        results = predict_sections(lyrics)
        print(format_prediction(results, args.song, args.artist))
        return

    p.print_help()


if __name__ == "__main__":
    main()

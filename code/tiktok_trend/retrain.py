#!/usr/bin/env python3
"""
Retrain TikTok Viral Models with Full Dataset
===============================================
Assembles a much larger, better-balanced training set than the original
and retrains all three models.

Training sources:
  summer_dataset_with_lyrics.csv  — 4,576 songs, tiers 0-4
      tier 0/1  → non-viral  (weight 1.0)
      tier 2    → skipped (ambiguous)
      tier 3/4  → viral      (weight 0.8)

  billboard_flops.csv             — 538 Billboard hits, NOT on TikTok
      label: non-viral            (weight 1.5 — hard negatives)

  cm_training_songs.csv           — Chartmetric TikTok trending
      tier -1   → viral           (weight 2.0 + video_count boost)

Train/test split: 80/20 stratified
Models saved to:
  local:    tiktok_trend/models/
  drive:    /Volumes/Dhesel Driv /Work/models/

Usage:
  cd /Users/dhekha/Desktop/SUPA_Spring/tiktok_trend
  python retrain.py          # retrain all 3 models
  python retrain.py --eval   # retrain + full evaluation metrics
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

EXTERNAL_DIR = Path("/Volumes/Dhesel Driv /Work")
LOCAL_DIR    = _HERE / "data"
MODELS_DIR   = _HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)
EXT_MODELS   = EXTERNAL_DIR / "models"


def _resolve(filename: str) -> Path | None:
    for base in [EXTERNAL_DIR, LOCAL_DIR]:
        p = base / filename
        if p.exists():
            return p
    return None


# ── Sample weight policy ─────────────────────────────────────────────────────

def _song_weight(tier, video_count=0) -> float:
    try:
        t = int(float(tier))
    except (ValueError, TypeError):
        return 1.0
    if t == -1:
        vc = max(int(video_count) if str(video_count) not in ('nan', '', 'None') else 0, 0)
        return 2.0 + math.log1p(vc / 50_000)
    if t in (3, 4):
        return 0.8
    return 1.0  # tier 0/1


def _billboard_weight() -> float:
    return 1.5  # hard negatives — confirmed not TikTok viral despite good lyrics


# ── Load and assemble training data ──────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """
    Load all sources, assign labels and weights, combine into one DataFrame.
    Returns df with columns: artist, song, tier, lyrics, sample_weight
    """
    frames = []

    # ── 1. Summer Dataset with lyrics ────────────────────────────────────
    path = _resolve("summer_dataset_with_lyrics.csv")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df['lyrics'].notna() & (df['lyrics'].astype(str).str.strip() != '') & (df['lyrics'] != 'nan')]
        # Skip tier 2 (ambiguous)
        df = df[df['tier'].isin([0, 1, 3, 4])].copy()
        df['sample_weight'] = df['tier'].apply(lambda t: _song_weight(t))
        df['source_set'] = 'summer'
        frames.append(df[['artist', 'song', 'tier', 'lyrics', 'sample_weight', 'source_set']])
        print(f"  Summer Dataset:       {len(df):>5,} songs  "
              f"(viral: {df['tier'].isin([3,4]).sum():,}  non-viral: {df['tier'].isin([0,1]).sum():,})")
    else:
        print("  [warn] summer_dataset_with_lyrics.csv not found")

    # ── 2. Billboard TikTok flops (hard negatives) ────────────────────────
    # Use billboard_2026.csv (has lyrics) and filter out songs that ARE on TikTok
    bb_path = _resolve("billboard_2026.csv")
    if bb_path:
        bb = pd.read_csv(bb_path, low_memory=False)
        bb = bb.drop_duplicates(subset=['song', 'artist'], keep='first')
        bb = bb[bb['lyrics'].notna() & (bb['lyrics'].astype(str).str.strip() != '') & (bb['lyrics'] != 'nan')]

        # Build TikTok fingerprint to exclude songs that went viral on TikTok
        import re as _re
        def _norm(t):
            t = str(t).lower()
            t = _re.sub(r'\s*[\(\[]?\s*feat\..*', '', t, flags=_re.IGNORECASE)
            t = _re.sub(r'[^\w\s]', ' ', t)
            return _re.sub(r'\s+', ' ', t).strip()

        tiktok_keys: set[tuple] = set()
        for fname, ac, sc in [("cm_tiktok_tracks.csv","artist","song"),
                               ("cm_tiktok_sounds.csv","artist","song"),
                               ("cm_training_songs.csv","artist","song"),
                               ("tiktok_trending.csv","artist","song"),
                               ("tiktok_new_to_dhesel_lyrics.csv","artist","title")]:
            p = _resolve(fname)
            if p is None:
                continue
            try:
                tmp = pd.read_csv(p, low_memory=False)
                for _, r in tmp.iterrows():
                    tiktok_keys.add((_norm(str(r.get(ac,''))), _norm(str(r.get(sc,'')))))
            except Exception:
                pass

        bb['_key'] = bb.apply(lambda r: (_norm(str(r['artist'])), _norm(str(r['song']))), axis=1)
        flops = bb[~bb['_key'].isin(tiktok_keys)].drop(columns=['_key'])

        flops = flops.copy()
        flops['tier']          = 0
        flops['sample_weight'] = _billboard_weight()
        flops['source_set']    = 'billboard_flops'
        frames.append(flops[['artist', 'song', 'tier', 'lyrics', 'sample_weight', 'source_set']])
        print(f"  Billboard flops:      {len(flops):>5,} songs  (all non-viral, weight {_billboard_weight()})")
    else:
        print("  [warn] billboard_2026.csv not found")

    # ── 3. Chartmetric TikTok trending (strong positives) ─────────────────
    path = _resolve("cm_training_songs.csv")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df['lyrics'].notna() & (df['lyrics'].astype(str).str.strip() != '') & (df['lyrics'] != 'nan')]
        df['tier'] = -1
        df['sample_weight'] = df.apply(
            lambda r: _song_weight(-1, r.get('video_count', 0)), axis=1
        )
        df['source_set'] = 'chartmetric'
        frames.append(df[['artist', 'song', 'tier', 'lyrics', 'sample_weight', 'source_set']])
        print(f"  Chartmetric TikTok:   {len(df):>5,} songs  (all viral, avg weight {df['sample_weight'].mean():.2f})")
    else:
        print("  [warn] cm_training_songs.csv not found")

    if not frames:
        raise RuntimeError("No training data found — check file paths")

    combined = pd.concat(frames, ignore_index=True)

    # Drop duplicates (prefer Chartmetric version over Summer for same song)
    combined = combined.drop_duplicates(
        subset=['artist', 'song'], keep='first'
    ).reset_index(drop=True)

    # Binary viral label
    combined['viral'] = combined['tier'].apply(
        lambda t: 1 if int(float(t)) in (-1, 3, 4) else 0
    )

    print(f"\n  Combined total:       {len(combined):>5,} songs")
    print(f"  Viral  (label=1):     {combined['viral'].sum():>5,}")
    print(f"  Non-viral (label=0):  {(combined['viral']==0).sum():>5,}")
    print(f"  Ratio:                {combined['viral'].mean():.2%} viral\n")

    return combined


# ── Train/test split ──────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['viral'], random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_song_model(clf, scaler, test_df: pd.DataFrame, detector) -> None:
    from song_viral_model import build_song_training_data

    print("\n  Evaluating on held-out test set...")
    X_test, y_test, _ = build_song_training_data(test_df, detector)

    if len(X_test) == 0:
        print("  [warn] No test samples — skipping evaluation")
        return

    X_sc   = scaler.transform(X_test)
    y_pred = clf.predict(X_sc)
    y_prob = clf.predict_proba(X_sc)[:, 1]

    print(classification_report(y_test, y_pred,
          target_names=["Non-Viral", "Viral"], digits=3))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  AUC-ROC: {auc:.4f}")
    except Exception:
        pass

    # Show score distribution
    viral_probs    = y_prob[y_test == 1]
    nonviral_probs = y_prob[y_test == 0]
    print(f"\n  Avg score — true viral:     {viral_probs.mean():.3f}  (median {np.median(viral_probs):.3f})")
    print(f"  Avg score — true non-viral: {nonviral_probs.mean():.3f}  (median {np.median(nonviral_probs):.3f})")
    print(f"  Separation gap:             {viral_probs.mean() - nonviral_probs.mean():+.3f}")


# ── Save models to both local and external drive ──────────────────────────────

def _save_pkl(obj: dict, filename: str) -> None:
    local = MODELS_DIR / filename
    with open(local, "wb") as f:
        pickle.dump(obj, f, protocol=5)
    print(f"    Local  → {local}")

    if EXTERNAL_DIR.exists():
        EXT_MODELS.mkdir(exist_ok=True)
        ext = EXT_MODELS / filename
        with open(ext, "wb") as f:
            pickle.dump(obj, f, protocol=5)
        print(f"    Drive  → {ext}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='Print full evaluation metrics on held-out test set')
    parser.add_argument('--skip-similarity', action='store_true',
                        help='Skip retraining the similarity model (fastest)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("TikTok Viral Model — Retraining")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Step 1/4  Loading training data...")
    combined = load_training_data()
    train_df, test_df = split_data(combined, test_size=0.2)
    print(f"  Train: {len(train_df):,} songs  |  Test: {len(test_df):,} songs")

    # ── Import training functions from train.py ───────────────────────────
    from train import (
        build_training_data,
        train_viral_line_model,
        train_similarity_model,
        FEATURE_NAMES,
    )
    from song_viral_model import (
        build_song_training_data,
        train_song_model,
        SONG_FEATURE_NAMES,
    )

    # ════════════════════════════════════════════════════════════════════════
    # MODEL 1 — Viral Line Classifier
    # ════════════════════════════════════════════════════════════════════════
    print("\nStep 2/4  Training Model 1 — Viral Line Classifier...")
    X, y, weights = build_training_data(train_df)
    print(f"  Lines: {len(X):,}  |  Viral: {y.sum():,} ({y.mean():.1%})  |  Non-viral: {(y==0).sum():,}")
    print(f"  Weight range: {weights.min():.2f} – {weights.max():.2f}  (mean {weights.mean():.2f})")

    clf1, scaler1 = train_viral_line_model(X, y, evaluate=args.eval, sample_weight=weights)
    print("  Done.")

    if args.eval:
        fi = sorted(zip(FEATURE_NAMES, clf1.feature_importances_), key=lambda x: x[1], reverse=True)
        print("\n  Feature importances:")
        for name, imp in fi:
            print(f"    {name:<22} {imp:.4f}  {'█' * int(imp * 40)}")

    _save_pkl({"clf": clf1, "scaler": scaler1}, "viral_line_model.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # MODEL 2 — Song Similarity
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_similarity:
        print("\nStep 3/4  Training Model 2 — Song Similarity (TF-IDF)...")
        sim_bundle = train_similarity_model(combined)  # use full dataset for similarity
        print(f"  Catalog: {sim_bundle['tfidf_matrix'].shape[0]:,} songs  "
              f"|  Vocab: {sim_bundle['tfidf_matrix'].shape[1]:,} terms")

        save_bundle = {
            "vectorizer":   sim_bundle["vectorizer"],
            "nn_model":     sim_bundle["nn_model"],
            "catalog_df":   sim_bundle["catalog_df"].drop(columns=["lyrics"], errors="ignore"),
            "tfidf_matrix": sim_bundle["tfidf_matrix"],
            "token_sets":   sim_bundle["token_sets"],
        }
        _save_pkl(save_bundle, "similarity_model.pkl")
    else:
        print("\nStep 3/4  [skipped] Similarity model")

    # ════════════════════════════════════════════════════════════════════════
    # MODEL 3 — Song-Level Virality Classifier
    # ════════════════════════════════════════════════════════════════════════
    print("\nStep 4/4  Training Model 3 — Song Virality Classifier...")
    print("  (Loading fresh Model 1 from disk to compute line features...)")

    # Reload Model 1 from disk so ViralLineDetector picks up the new weights
    from nlp_models import ViralLineDetector
    ViralLineDetector._instance = None   # reset singleton
    detector = ViralLineDetector.get()

    X_song, y_song, w_song = build_song_training_data(train_df, detector)
    print(f"  Songs: {len(X_song):,}  |  Viral: {y_song.sum():,} ({y_song.mean():.1%})  "
          f"|  Non-viral: {(y_song==0).sum():,}")
    print(f"  Weight range: {w_song.min():.2f} – {w_song.max():.2f}  (mean {w_song.mean():.2f})")

    clf3, scaler3 = train_song_model(X_song, y_song, evaluate=args.eval, sample_weight=w_song)

    if args.eval:
        evaluate_song_model(clf3, scaler3, test_df, detector)

        fi = sorted(zip(SONG_FEATURE_NAMES, clf3.feature_importances_), key=lambda x: x[1], reverse=True)
        print("\n  Song feature importances:")
        for name, imp in fi:
            print(f"    {name:<25} {imp:.4f}  {'█' * int(imp * 40)}")

    _save_pkl({"clf": clf3, "scaler": scaler3}, "song_viral_model.pkl")

    print(f"\n{'='*60}")
    print("Retraining complete. Models saved:")
    for f in ["viral_line_model.pkl", "song_viral_model.pkl", "similarity_model.pkl"]:
        local = MODELS_DIR / f
        flag  = "✓" if local.exists() else "✗"
        print(f"  {flag}  {local}")
    print(f"{'='*60}\n")
    print("Next: run  python model_tester.py  to re-evaluate with new models.\n")


if __name__ == "__main__":
    main()

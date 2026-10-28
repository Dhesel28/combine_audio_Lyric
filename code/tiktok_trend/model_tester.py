#!/usr/bin/env python3
"""
TikTok Viral Model Testing Suite
==================================
Tests the trained models against two evaluation sets:

  Set A — Summer Dataset
      Labeled tiers 0-4 (0/1 = non-viral, 3/4 = viral, 2 = ambiguous/skipped)
      Lyrics matched from all available sources on your laptop and external drive.
      Ground truth: the tier labels already in the dataset.

  Set B — Billboard TikTok Flops (Non-hit Set 2)
      Songs that charted on Billboard 2026 Hot 100
      BUT do NOT appear in ANY Chartmetric/TikTok dataset.
      These are traditional mainstream hits that never went viral on TikTok.
      Ground truth: true label = 0 (not TikTok viral)

Both sets are kept separate — not combined.

Outputs (saved to external drive or local data/ as fallback):
  summer_evaluation.csv      — per-song predictions vs. true Summer tier
  billboard_flops.csv        — per-song predictions on Billboard non-TikTok songs
  evaluation_report.txt      — full classification metrics for both sets
  evaluation_metrics.json    — metrics as structured JSON

Usage:
  cd /Users/dhekha/Desktop/SUPA_Spring/tiktok_trend
  python model_tester.py                    # full run
  python model_tester.py --limit 100        # test run (100 songs per set)
  python model_tester.py --fetch-missing    # fetch missing lyrics via Genius
  python model_tester.py --skip-billboard   # only run Summer Dataset
  python model_tester.py --skip-summer      # only run Billboard flops
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

EXTERNAL_DIR = Path("/Volumes/Dhesel Driv /Work")
LOCAL_DIR    = _HERE / "data"

# Save outputs to external drive if mounted, otherwise save locally
OUT_DIR = EXTERNAL_DIR if EXTERNAL_DIR.exists() else LOCAL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve(filename: str) -> Path | None:
    """Return the first location where this file actually exists."""
    for base in [EXTERNAL_DIR, LOCAL_DIR]:
        p = base / filename
        if p.exists():
            return p
    return None


# ── Tier definitions (mirrors train.py / song_viral_model.py) ────────────────
VIRAL_TIERS    = {-1, 3, 4}   # TikTok trending / major / super hits
NONVIRAL_TIERS = {0, 1}       # non-hits / minor hits
THRESHOLD      = 0.5           # viral_prob >= 0.5 → predicted viral


# ── Text normalisation for fuzzy artist/song matching ────────────────────────

_FEAT_RE    = re.compile(r'\s*[\(\[]?\s*feat\..*', re.IGNORECASE)
_PAREN_RE   = re.compile(r'\([^)]*\)')
_BRACKET_RE = re.compile(r'\[[^\]]*\]')
_NONWORD_RE = re.compile(r'[^\w\s]')
_WS_RE      = re.compile(r'\s+')
_SUFFIXES   = re.compile(
    r'\s*[-–]\s*(official\s*(video|audio|lyric|mv)|lyrics?|audio|video|remaster.*|live.*|'
    r'sped\s*up|slowed|remix|edit).*$', re.IGNORECASE
)


def _norm(text: str) -> str:
    """Normalise a song or artist name for fuzzy matching."""
    t = str(text).lower()
    t = _FEAT_RE.sub('', t)
    t = _PAREN_RE.sub(' ', t)
    t = _BRACKET_RE.sub(' ', t)
    t = _SUFFIXES.sub('', t)
    t = _NONWORD_RE.sub(' ', t)
    t = _WS_RE.sub(' ', t).strip()
    return t


def _key(artist: str, song: str) -> tuple[str, str]:
    return (_norm(artist), _norm(song))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LYRICS LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

def build_lyrics_lookup() -> dict[tuple[str, str], str]:
    """
    Build (norm_artist, norm_song) → lyrics dict from every available source.

    Sources checked (external drive first, then local):
      cm_training_songs.csv          — Chartmetric training set (~995 with lyrics)
      billboard_2026.csv             — Billboard Hot 100 2026 (~590 with lyrics)
      catalog_with_lyrics.csv        — main catalog (~15k songs)
      tiktok_new_to_dhesel_lyrics.csv — curated TikTok songs with lyrics
    """
    lookup: dict[tuple[str, str], str] = {}

    sources = [
        ("cm_training_songs.csv",           "artist", "song",  "lyrics"),
        ("billboard_2026.csv",              "artist", "song",  "lyrics"),
        ("catalog_with_lyrics.csv",         "artist", "song",  "lyrics"),
        ("tiktok_new_to_dhesel_lyrics.csv", "artist", "title", "lyrics"),
    ]

    for filename, a_col, s_col, l_col in sources:
        path = _resolve(filename)
        if path is None:
            print(f"  [skip] {filename} not found")
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  [warn] Could not read {filename}: {e}")
            continue

        if not all(c in df.columns for c in [a_col, s_col, l_col]):
            missing = [c for c in [a_col, s_col, l_col] if c not in df.columns]
            print(f"  [warn] {filename} missing columns: {missing}")
            continue

        before = len(lookup)
        for _, row in df.iterrows():
            lyrics = str(row[l_col]).strip()
            if not lyrics or lyrics.lower() in ('nan', '') or len(lyrics) < 50:
                continue
            k = _key(str(row[a_col]), str(row[s_col]))
            if k not in lookup:
                lookup[k] = lyrics

        added = len(lookup) - before
        print(f"  {filename:<40}  +{added:>5,} songs  (total {len(lookup):,})")

    print(f"\n  Lyrics lookup complete: {len(lookup):,} unique songs\n")
    return lookup


def _find_lyrics(
    artist: str,
    song: str,
    lookup: dict[tuple[str, str], str],
) -> str:
    """
    Try several matching strategies to find lyrics for artist/song.
    Returns empty string if not found.
    """
    # Exact normalised match
    k = _key(artist, song)
    if k in lookup:
        return lookup[k]

    # Song-only fallback (artist might be spelled differently)
    norm_song = _norm(song)
    for (ka, ks), lyr in lookup.items():
        if ks == norm_song:
            return lyr

    # Partial: song starts-with match (handles "song name (radio edit)" etc.)
    for (ka, ks), lyr in lookup.items():
        if norm_song and ks.startswith(norm_song[:min(len(norm_song), 10)]):
            if _norm(artist) and ka.startswith(_norm(artist)[:min(len(_norm(artist)), 6)]):
                return lyr

    return ''


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], str],
    fetch_missing: bool = False,
    set_label: str = "",
) -> pd.DataFrame:
    """
    For each row in df, find lyrics, then run ViralLineDetector +
    SongViralityPredictor.

    Columns added:
      lyrics_found, viral_prob, viral_label, viral_line,
      viral_section, viral_repeats, top_line_prob, avg_line_prob
    """
    from nlp_models import ViralLineDetector, SongViralityPredictor

    print(f"  Loading models...")
    try:
        detector  = ViralLineDetector.get()
        predictor = SongViralityPredictor.get()
    except FileNotFoundError as e:
        print(f"  [ERROR] Model file missing: {e}")
        print("  Run:  cd tiktok_trend && python train.py")
        sys.exit(1)

    results    = []
    no_lyrics  = 0
    errors     = 0
    total      = len(df)

    prefix = f"[{set_label}] " if set_label else ""
    print(f"  {prefix}Running inference on {total:,} songs...")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 100 == 0 or i == total:
            pct = i / total * 100
            found_pct = (i - no_lyrics) / i * 100
            print(f"    {i:>5}/{total}  ({pct:.0f}%)  lyrics found: {found_pct:.0f}%")

        artist = str(row.get('artist', '')).strip()
        song   = str(row.get('song', row.get('title', ''))).strip()
        tier   = row.get('tier', None)
        source = str(row.get('source', ''))

        # ── Find lyrics ──────────────────────────────────────────────────
        lyrics = _find_lyrics(artist, song, lookup)

        # Optionally fetch via Genius if still missing
        if not lyrics and fetch_missing:
            try:
                from genius_fetcher import GeniusFetcher
                fetcher = GeniusFetcher()
                fetched = fetcher.fetch(artist, song)
                if fetched and len(fetched) >= 50:
                    lyrics = fetched
                    lookup[_key(artist, song)] = lyrics
                time.sleep(0.3)  # rate limit
            except Exception:
                pass

        rec = {
            'artist':       artist,
            'song':         song,
            'tier':         tier,
            'source':       source,
            'lyrics_found': bool(lyrics),
        }

        if not lyrics:
            no_lyrics += 1
            rec.update({
                'viral_prob':    None,
                'viral_label':   'no_lyrics',
                'viral_line':    '',
                'viral_section': '',
                'viral_repeats': 0,
                'top_line_prob': None,
                'avg_line_prob': None,
            })
        else:
            try:
                line_res = detector.predict(lyrics)
                song_res = predictor.predict(lyrics, line_result=line_res)
                vl       = line_res.get('viral_line') or {}
                all_l    = line_res.get('all_lines', [])
                avg_p    = (sum(l['viral_prob'] for l in all_l) / len(all_l)
                            if all_l else None)
                rec.update({
                    'viral_prob':    song_res['viral_prob'],
                    'viral_label':   song_res['label'],
                    'viral_line':    vl.get('text', ''),
                    'viral_section': vl.get('section', ''),
                    'viral_repeats': vl.get('repeats', 0),
                    'top_line_prob': vl.get('viral_prob', None),
                    'avg_line_prob': round(avg_p, 4) if avg_p is not None else None,
                })
            except Exception as e:
                errors += 1
                rec.update({
                    'viral_prob':    None,
                    'viral_label':   f'error',
                    'viral_line':    '',
                    'viral_section': '',
                    'viral_repeats': 0,
                    'top_line_prob': None,
                    'avg_line_prob': None,
                })

        results.append(rec)

    scored   = total - no_lyrics - errors
    print(f"\n  {prefix}Done.")
    print(f"    Songs scored:       {scored:>5,} / {total:,}")
    print(f"    No lyrics found:    {no_lyrics:>5,} / {total:,}")
    print(f"    Errors:             {errors:>5,} / {total:,}")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_true_label(tier) -> int | None:
    """Convert Summer Dataset tier to binary TikTok-viral label."""
    try:
        t = int(float(tier))
    except (ValueError, TypeError):
        return None
    if t in VIRAL_TIERS:    return 1
    if t in NONVIRAL_TIERS: return 0
    return None  # tier 2 = ambiguous, skip


def evaluate_set(df: pd.DataFrame, set_name: str) -> dict:
    """
    Compute classification metrics.
    Uses rows where viral_prob is valid AND tier maps to a clear label.
    """
    sub = df[df['viral_prob'].notna()].copy()
    sub['true_label'] = sub['tier'].apply(get_true_label)
    sub = sub[sub['true_label'].notna()].copy()
    sub['pred_label'] = (sub['viral_prob'] >= THRESHOLD).astype(int)

    n = len(sub)
    if n < 5:
        print(f"  [warn] {set_name}: only {n} evaluable rows — skipping metrics")
        return {'set_name': set_name, 'n_songs': n, 'error': 'too_few_samples'}

    y_true = sub['true_label'].astype(int).values
    y_pred = sub['pred_label'].values
    y_prob = sub['viral_prob'].astype(float).values

    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Non-Viral", "Viral"],
        digits=3,
        zero_division=0,
    )

    try:
        from sklearn.metrics import roc_auc_score
        auc = round(roc_auc_score(y_true, y_prob), 4)
    except Exception:
        auc = 'N/A'

    metrics: dict = {
        'set_name':            set_name,
        'n_scored':            n,
        'n_viral_true':        int(y_true.sum()),
        'n_nonviral_true':     int((1 - y_true).sum()),
        'n_predicted_viral':   int(y_pred.sum()),
        'accuracy':            round(acc, 4),
        'f1_weighted':         round(f1_w, 4),
        'f1_macro':            round(f1_m, 4),
        'precision_weighted':  round(prec, 4),
        'recall_weighted':     round(rec, 4),
        'auc_roc':             auc,
        'avg_viral_prob':      round(float(y_prob.mean()), 4),
        'confusion_matrix':    cm.tolist(),
        'classification_report': report,
    }

    # Per-tier breakdown (for Summer Dataset)
    tier_breakdown = {}
    for tier_val in sorted(sub['tier'].dropna().unique()):
        t_sub = sub[sub['tier'] == tier_val]
        cnt   = len(t_sub)
        if cnt == 0:
            continue
        pred_v = (t_sub['viral_prob'] >= THRESHOLD).sum()
        tier_breakdown[int(tier_val)] = {
            'count':            cnt,
            'predicted_viral':  int(pred_v),
            'viral_rate':       round(pred_v / cnt, 3),
            'avg_viral_prob':   round(float(t_sub['viral_prob'].mean()), 3),
            'median_viral_prob':round(float(t_sub['viral_prob'].median()), 3),
        }

    metrics['tier_breakdown'] = tier_breakdown
    return metrics


def evaluate_billboard_flops(df: pd.DataFrame) -> dict:
    """
    For Billboard flops, all songs are true non-viral (TikTok label = 0).
    Measures: what fraction does the model correctly predict as non-viral?
    Also reports average viral score — useful for cross-set comparison.
    """
    valid = df[df['viral_prob'].notna()].copy()
    n = len(valid)
    if n < 5:
        return {'set_name': 'Set B: Billboard TikTok Flops', 'n_songs': n,
                'error': 'too_few_samples'}

    y_prob   = valid['viral_prob'].astype(float).values
    y_pred   = (y_prob >= THRESHOLD).astype(int)
    pred_nv  = int((y_pred == 0).sum())
    pred_v   = int((y_pred == 1).sum())

    return {
        'set_name':                 'Set B: Billboard TikTok Flops',
        'n_songs':                  n,
        'n_viral_true':             0,           # ground truth: 0 are TikTok viral
        'n_nonviral_true':          n,
        'n_predicted_nonviral':     pred_nv,
        'n_predicted_viral':        pred_v,
        'correct_nonviral_rate':    round(pred_nv / n, 4),   # higher is better
        'false_viral_rate':         round(pred_v / n, 4),    # lower is better
        'avg_viral_prob':           round(float(y_prob.mean()), 4),
        'median_viral_prob':        round(float(np.median(y_prob)), 4),
        'pct_below_threshold':      round(pred_nv / n * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BILLBOARD FLOPS BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_billboard_flops() -> pd.DataFrame:
    """
    Returns a DataFrame of Billboard 2026 Hot 100 songs that do NOT appear
    in any Chartmetric/TikTok dataset.

    'Not on TikTok' is determined by checking:
      cm_tiktok_tracks.csv
      cm_tiktok_sounds.csv
      cm_training_songs.csv
      tiktok_trending.csv
      tiktok_new_to_dhesel_lyrics.csv

    These songs are Billboard hits that did not go viral on TikTok.
    """
    # ── Load Billboard ────────────────────────────────────────────────────
    bb_path = _resolve("billboard_2026.csv")
    if bb_path is None:
        print("  [warn] billboard_2026.csv not found on external drive or locally")
        return pd.DataFrame()

    bb = pd.read_csv(bb_path, low_memory=False)
    print(f"  Billboard 2026 loaded: {len(bb):,} rows")

    # Deduplicate by song+artist — keep first occurrence (earliest chart date)
    bb = bb.drop_duplicates(subset=['song', 'artist'], keep='first').copy()
    print(f"  After dedup: {len(bb):,} unique songs")

    # ── Build TikTok fingerprint ──────────────────────────────────────────
    tiktok_keys: set[tuple[str, str]] = set()

    tiktok_sources = [
        ("cm_tiktok_tracks.csv",            "artist", "song"),
        ("cm_tiktok_sounds.csv",            "artist", "song"),
        ("cm_training_songs.csv",           "artist", "song"),
        ("tiktok_trending.csv",             "artist", "song"),
        ("tiktok_new_to_dhesel_lyrics.csv", "artist", "title"),
    ]

    for filename, a_col, s_col in tiktok_sources:
        path = _resolve(filename)
        if path is None:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if a_col not in df.columns or s_col not in df.columns:
            continue
        before = len(tiktok_keys)
        for _, row in df.iterrows():
            tiktok_keys.add(_key(str(row[a_col]), str(row[s_col])))
        print(f"  {filename:<40}  +{len(tiktok_keys)-before:>4,} TikTok keys")

    print(f"  Total TikTok fingerprints: {len(tiktok_keys):,}")

    # ── Separate flops from TikTok-viral ──────────────────────────────────
    flop_rows  = []
    viral_rows = []
    for _, row in bb.iterrows():
        k = _key(str(row['artist']), str(row['song']))
        if k in tiktok_keys:
            viral_rows.append(row)
        else:
            flop_rows.append(row)

    print(f"\n  Billboard songs that ARE on TikTok:     {len(viral_rows):>4,}")
    print(f"  Billboard songs NOT on TikTok (flops):  {len(flop_rows):>4,}")

    flops = pd.DataFrame(flop_rows).copy()
    flops['tier']                = 0    # true TikTok-viral label = non-viral
    flops['billboard_hit']       = True # they ARE Billboard hits
    flops['tiktok_viral']        = False
    flops['source']              = 'billboard_nonviral_tiktok'
    return flops


# ═══════════════════════════════════════════════════════════════════════════════
# 5. REPORT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(val: float, width: int = 30) -> str:
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def format_report(
    metrics_a: dict,
    metrics_b: dict,
    summer_df: pd.DataFrame | None = None,
    flops_df:  pd.DataFrame | None = None,
) -> str:
    lines = [
        "=" * 72,
        "  TikTok Viral Model — Evaluation Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 72,
        "",
        "OVERVIEW",
        "  Two evaluation sets, kept separate (not combined):",
        "    Set A — Summer Dataset:    labeled tiers 0-4",
        "    Set B — Billboard Flops:   songs on Billboard but never TikTok-viral",
        "",
        "  Model threshold:   viral_prob >= 0.50  →  predicted viral",
        "",
    ]

    # ── Set A ─────────────────────────────────────────────────────────────
    if metrics_a and 'error' not in metrics_a:
        m = metrics_a
        lines += [
            "─" * 72,
            "SET A — SUMMER DATASET",
            f"  Songs evaluated:         {m['n_scored']:,}",
            f"  True viral (tier 3/4):   {m['n_viral_true']:,}",
            f"  True non-viral (0/1):    {m['n_nonviral_true']:,}",
            "",
            f"  Accuracy:                {m['accuracy']:.1%}  {_bar(m['accuracy'])}",
            f"  F1 (weighted):           {m['f1_weighted']:.3f}  {_bar(m['f1_weighted'])}",
            f"  F1 (macro):              {m['f1_macro']:.3f}  {_bar(m['f1_macro'])}",
            f"  Precision (weighted):    {m['precision_weighted']:.3f}",
            f"  Recall (weighted):       {m['recall_weighted']:.3f}",
            f"  AUC-ROC:                 {m['auc_roc']}",
            f"  Avg predicted viral prob:{m['avg_viral_prob']:.3f}",
            "",
            "  Classification Report:",
        ]
        for ln in m['classification_report'].split('\n'):
            lines.append(f"    {ln}")

        if m.get('tier_breakdown'):
            lines += ["", "  Tier-level breakdown (Set A):"]
            lines.append(f"  {'Tier':<6} {'Count':>6} {'Pred Viral':>10} {'Viral Rate':>11} {'Avg Prob':>9} {'Median':>8}")
            lines.append(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*11} {'-'*9} {'-'*8}")
            for t, tb in sorted(m['tier_breakdown'].items()):
                label = "← hits" if t in (3, 4) else ("← non-hits" if t in (0, 1) else "← ambiguous")
                lines.append(
                    f"  {t:<6} {tb['count']:>6,} {tb['predicted_viral']:>10,} "
                    f"{tb['viral_rate']:>10.1%} {tb['avg_viral_prob']:>9.3f} "
                    f"{tb['median_viral_prob']:>8.3f}  {label}"
                )
        lines.append("")

    # ── Set B ─────────────────────────────────────────────────────────────
    if metrics_b and 'error' not in metrics_b:
        m = metrics_b
        lines += [
            "─" * 72,
            "SET B — BILLBOARD TIKTOK FLOPS (Non-hit Set 2)",
            "  These songs charted on Billboard 2026 Hot 100",
            "  but do NOT appear in any Chartmetric / TikTok dataset.",
            "  Ground truth: ALL are TikTok non-viral (true label = 0).",
            "",
            f"  Songs evaluated:                 {m['n_songs']:,}",
            f"  Correctly predicted non-viral:   {m['n_predicted_nonviral']:,} ({m['correct_nonviral_rate']:.1%})",
            f"  Incorrectly flagged as viral:     {m['n_predicted_viral']:,} ({m['false_viral_rate']:.1%})",
            f"  Avg viral score:                 {m['avg_viral_prob']:.3f}",
            f"  Median viral score:              {m['median_viral_prob']:.3f}",
            "",
            "  Interpretation:",
            "    A high false-viral rate here means the model is picking up",
            "    mainstream-lyric patterns (Billboard-worthy) as TikTok-viral.",
            "    A low false-viral rate means the model correctly distinguishes",
            "    between chart success and TikTok virality — which is the goal.",
            "",
        ]

    # ── Cross-set comparison ──────────────────────────────────────────────
    if (metrics_a and 'avg_viral_prob' in metrics_a and
            metrics_b and 'avg_viral_prob' in metrics_b):
        diff = metrics_a['avg_viral_prob'] - metrics_b['avg_viral_prob']
        lines += [
            "─" * 72,
            "CROSS-SET COMPARISON",
            f"  Set A avg viral prob:  {metrics_a['avg_viral_prob']:.3f}",
            f"  Set B avg viral prob:  {metrics_b['avg_viral_prob']:.3f}",
            f"  Difference (A - B):   {diff:+.3f}",
            "",
            ("  → The model scores Summer hits higher than Billboard flops." if diff > 0 else
             "  → The model scores Billboard flops similarly to Summer songs (check threshold)."),
            "",
        ]

    lines.append("=" * 72)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test TikTok viral models on Summer Dataset and Billboard flops"
    )
    parser.add_argument('--fetch-missing', action='store_true',
                        help="Fetch missing lyrics via Genius API (slow)")
    parser.add_argument('--limit', type=int, default=0,
                        help="Max songs per set — 0 means no limit (use for quick tests)")
    parser.add_argument('--skip-summer',    action='store_true', help="Skip Set A")
    parser.add_argument('--skip-billboard', action='store_true', help="Skip Set B")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("TikTok Viral Model Tester")
    print(f"Output directory: {OUT_DIR}")
    print(f"External drive:   {'mounted ✓' if EXTERNAL_DIR.exists() else 'not found — using local'}")
    print(f"{'='*60}\n")

    # ── Build shared lyrics lookup ────────────────────────────────────────
    print("Step 1/4  Building lyrics lookup from all available sources...")
    lookup = build_lyrics_lookup()

    metrics_a:   dict         = {}
    metrics_b:   dict         = {}
    summer_df:   pd.DataFrame = pd.DataFrame()
    flops_df:    pd.DataFrame = pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════
    # SET A — SUMMER DATASET
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_summer:
        print("Step 2/4  Loading Summer Dataset with lyrics...")
        # Prefer the fully-enriched file; fall back to bare Summer Dataset
        summer_path = _resolve("summer_dataset_with_lyrics.csv") or _resolve("Summer Dataset.csv")
        if summer_path is None:
            print("  [warn] Summer Dataset not found — skipping Set A")
        else:
            summer = pd.read_csv(summer_path, low_memory=False)
            has_lyrics = 'lyrics' in summer.columns
            if has_lyrics:
                with_lyr = (summer['lyrics'].notna() & (summer['lyrics'].str.strip() != '') & (summer['lyrics'] != 'nan')).sum()
                print(f"  {len(summer):,} songs loaded — {with_lyr:,} have lyrics ({with_lyr/len(summer)*100:.1f}%)")
            tier_counts = summer['tier'].value_counts().sort_index().to_dict()
            print(f"  Tier distribution: {tier_counts}")

            if args.limit > 0:
                n_per_tier = max(args.limit // 5, 5)
                summer = pd.concat([
                    g.sample(min(len(g), n_per_tier), random_state=42)
                    for _, g in summer.groupby('tier')
                ]).reset_index(drop=True)
                print(f"  Limited to {len(summer):,} songs ({n_per_tier} per tier)")

            # If lyrics are embedded in the file, inject them into the lookup
            if has_lyrics:
                injected = 0
                for _, row in summer.iterrows():
                    lyr = str(row.get('lyrics', '')).strip()
                    if lyr and lyr != 'nan' and len(lyr) >= 50:
                        k = _key(str(row.get('artist', '')), str(row.get('song', '')))
                        if k not in lookup:
                            lookup[k] = lyr
                            injected += 1
                print(f"  Injected {injected:,} new lyrics into lookup from enriched file")

            print()
            summer_df = run_inference(summer, lookup, args.fetch_missing, set_label="Set A")

            # Save to drive
            out = OUT_DIR / "summer_evaluation.csv"
            summer_df.to_csv(out, index=False)
            print(f"\n  Saved → {out}")

            # Evaluate
            print("\n  Computing Set A metrics...")
            metrics_a = evaluate_set(summer_df, "Set A: Summer Dataset")
            if 'accuracy' in metrics_a:
                print(f"  Accuracy:    {metrics_a['accuracy']:.1%}")
                print(f"  F1 weighted: {metrics_a['f1_weighted']:.3f}")
                print(f"  AUC-ROC:     {metrics_a['auc_roc']}")
    else:
        print("Step 2/4  [skipped] Summer Dataset")

    # ══════════════════════════════════════════════════════════════════════
    # SET B — BILLBOARD TIKTOK FLOPS
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_billboard:
        print("\nStep 3/4  Building Billboard TikTok Flops (Non-hit Set 2)...")
        flops_raw = build_billboard_flops()

        if flops_raw.empty:
            print("  [warn] No Billboard flops found — skipping Set B")
        else:
            if args.limit > 0:
                flops_raw = flops_raw.sample(
                    min(len(flops_raw), args.limit), random_state=42
                )
                print(f"  Limited to {len(flops_raw):,} songs")

            print()
            flops_df = run_inference(flops_raw, lookup, args.fetch_missing, set_label="Set B")

            # Ground truth: all flops are TikTok non-viral
            flops_df['tier']         = 0
            flops_df['billboard_hit'] = True

            # Save to drive
            out = OUT_DIR / "billboard_flops.csv"
            flops_df.to_csv(out, index=False)
            print(f"\n  Saved → {out}")

            # Evaluate
            print("\n  Computing Set B metrics...")
            metrics_b = evaluate_billboard_flops(flops_df)
            if 'correct_nonviral_rate' in metrics_b:
                print(f"  Correctly predicted non-viral: {metrics_b['correct_nonviral_rate']:.1%}")
                print(f"  Falsely flagged as viral:      {metrics_b['false_viral_rate']:.1%}")
                print(f"  Avg viral score:               {metrics_b['avg_viral_prob']:.3f}")
    else:
        print("Step 3/4  [skipped] Billboard flops")

    # ── Save report and metrics ───────────────────────────────────────────
    print("\nStep 4/4  Saving evaluation report and metrics...")

    _s = summer_df if not summer_df.empty else None
    _f = flops_df  if not flops_df.empty  else None
    report = format_report(metrics_a, metrics_b, _s, _f)
    print("\n" + report)

    rpt_path = OUT_DIR / "evaluation_report.txt"
    rpt_path.write_text(report, encoding='utf-8')
    print(f"  Report    → {rpt_path}")

    json_path = OUT_DIR / "evaluation_metrics.json"
    json_path.write_text(
        json.dumps(
            {
                'generated':       datetime.now().isoformat(),
                'threshold':       THRESHOLD,
                'set_a_summer':    {k: v for k, v in metrics_a.items()
                                    if k != 'classification_report'},
                'set_b_billboard': metrics_b,
            },
            indent=2, default=str
        ),
        encoding='utf-8',
    )
    print(f"  Metrics   → {json_path}")

    # Also save a compact summary CSV
    summary_rows = []
    if not summer_df.empty:
        for _, r in summer_df[summer_df['viral_prob'].notna()].iterrows():
            summary_rows.append({
                'set': 'A_summer',
                'artist': r['artist'],
                'song': r['song'],
                'tier': r['tier'],
                'true_viral': get_true_label(r['tier']),
                'viral_prob': r['viral_prob'],
                'predicted_viral': 1 if r['viral_prob'] >= THRESHOLD else 0,
                'viral_line': r['viral_line'],
            })
    if not flops_df.empty:
        for _, r in flops_df[flops_df['viral_prob'].notna()].iterrows():
            summary_rows.append({
                'set': 'B_billboard_flop',
                'artist': r['artist'],
                'song': r['song'],
                'tier': 0,
                'true_viral': 0,
                'viral_prob': r['viral_prob'],
                'predicted_viral': 1 if r['viral_prob'] >= THRESHOLD else 0,
                'viral_line': r['viral_line'],
            })

    if summary_rows:
        summary_path = OUT_DIR / "evaluation_all_songs.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"  All songs → {summary_path}")

    print(f"\n{'='*60}")
    print("All done. Files saved to:")
    for f in ['summer_evaluation.csv', 'billboard_flops.csv',
              'evaluation_report.txt', 'evaluation_metrics.json',
              'evaluation_all_songs.csv']:
        p = OUT_DIR / f
        exists = "✓" if p.exists() else "✗"
        print(f"  {exists}  {p}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

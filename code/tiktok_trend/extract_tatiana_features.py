"""
Extract All 51 Lyric Features for Tatiana DeMaria Songs
========================================================
Reads data/lyrics/full_tatiana_demaria_lyrics.csv, computes the same
51 lf_ features used by the combined TikTok Virality model, and saves:

    data/lyrics/tatiana_lyric_features.csv

Feature groups (51 total):
  lf_   (11)  basic song-level
  lf2_  (17)  section-level signals
  lf3_  (12)  Tier 1 NLP
  lf_emb (11)  MiniLM sentence embeddings + chorus/verse contrast

Note on lf_emb_0..9:
  Training used PCA fit on the full training set (~2,700 songs).
  That PCA transform is not saved, so PCA here is re-fit on Tatiana's
  6 songs only — values are not directly comparable to training data.
  Use these columns for intra-Tatiana comparison only.

Usage:
    cd SUPA_Spring
    python code/tiktok_trend/extract_tatiana_features.py
"""
from __future__ import annotations

import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent          # code/tiktok_trend
_CODE  = _HERE.parent                             # code/
_ROOT  = _CODE.parent                             # SUPA_Spring

INPUT_CSV  = _ROOT / "full_tatiana_demaria_lyrics.csv"
OUTPUT_CSV = _ROOT / "data" / "lyrics" / "tatiana_lyric_features.csv"

# Add lyric_analyzer package to path
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_HERE))

from lyric_analyzer.analyzer import LyricAnalyzer, _extract_song_features
from lyric_analyzer.signals import (
    sig_hook_vocabulary, sig_cross_song_repetition, sig_internal_repetition,
    sig_sentiment_extremity, sig_end_rhyme_density, sig_line_conciseness,
    sig_syllable_consistency, sig_phonetic_openness, sig_pronoun_relatability,
    sig_internal_rhyme, sig_energy_markers,
    line_sig_repetition, line_sig_energy, line_sig_rhyme,
    line_sig_brevity, line_sig_section_bonus, line_sig_emotional_punch,
    _count_syllables,
)
from lyric_analyzer.config import HOOK_SECTION_KEYS, LINE_WEIGHTS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader    = SentimentIntensityAnalyzer()
_ANALYZER = LyricAnalyzer()
_SECTION_RE = re.compile(r'\[([^\]]+)\]', re.I)

# ── Imperative (action verb) starters ────────────────────────────────────────
_ACTION_VERBS = {
    "go","come","look","run","feel","take","make","give","get","put","call",
    "dance","move","shake","turn","stop","start","listen","hold","let","say",
    "tell","bring","leave","stay","fall","rise","break","show","find","open",
    "love","hate","cry","laugh","smile","sing","play","push","pull","jump","fly",
    "reach","walk","close","keep","throw","catch","hit","drop","lift","pull",
}

# ── Explicit / slang word set ─────────────────────────────────────────────────
_EXPLICIT = {
    "fuck","shit","bitch","ass","damn","dick","pussy","cock","nigga","nigger",
    "hoe","slut","whore","bastard","bullshit","fucking","bitches","asses",
    "drunk","wasted","high","stoned","motherfucker","crap",
}

# ── Function words for lexical density ───────────────────────────────────────
_FUNCTION_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","it","its","this","that",
    "these","those","as","if","so","i","me","my","we","our","you","your",
    "he","she","they","him","her","them","his","hers","their","gonna","gotta",
    "wanna","let","would","could","should","just","now","when","where","what",
    "how","yeah","oh","ooh","na","la","da","hey","uh","mm","ya","no","not",
    "do","did","will","have","has","had","get","got","can","may","might",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_lines(raw: str) -> list[str]:
    """Return only lyric lines (strip section headers and blank lines)."""
    return [
        l.strip() for l in raw.split("\n")
        if l.strip() and not _SECTION_RE.match(l.strip())
    ]


def _last_vowel_sound(line: str) -> str:
    """Rough end-rhyme sound: vowels-onward from last vowel group in last word."""
    words = re.findall(r"[a-z]+", line.lower())
    if not words:
        return ""
    w = words[-1]
    for i, c in enumerate(w):
        if c in "aeiou":
            return w[i:]
    return w[-2:] if len(w) >= 2 else w


# ══════════════════════════════════════════════════════════════════════════════
# lf_  :  11 BASIC SONG-LEVEL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def get_lf_features(raw: str) -> dict:
    """
    Uses the existing _extract_song_features() function from analyzer.py.
    Adds 'lf_' prefix to each key.
    """
    feats = _extract_song_features(raw)
    return {f"lf_{k}": v for k, v in feats.items()}


# ══════════════════════════════════════════════════════════════════════════════
# lf2_  :  17 SECTION-LEVEL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def get_lf2_features(raw: str, result: dict) -> dict:
    """
    Pulls section-level signals from LyricAnalyzer.analyze() result.
    Also computes VADER sentiment, section count, line count.
    """
    best = result.get("best_section", {})
    lines = result.get("lines", [])
    sections = result.get("sections", [])
    lyric_lines = _parse_lines(raw)

    line_scores = [l["score"] for l in lines] if lines else []

    plain = _SECTION_RE.sub("", raw).strip()
    vader_compound = round(_vader.polarity_scores(plain)["compound"], 4)

    return {
        "lf2_hook_vocabulary":        best.get("hook_vocabulary",       0.0),
        "lf2_cross_song_repetition":  best.get("cross_song_repetition", 0.0),
        "lf2_internal_repetition":    best.get("internal_repetition",   0.0),
        "lf2_sentiment_extremity":    best.get("sentiment_extremity",   0.0),
        "lf2_end_rhyme_density":      best.get("end_rhyme_density",     0.0),
        "lf2_line_conciseness":       best.get("line_conciseness",      0.0),
        "lf2_syllable_consistency":   best.get("syllable_consistency",  0.0),
        "lf2_phonetic_openness":      best.get("phonetic_openness",     0.0),
        "lf2_pronoun_relatability":   best.get("pronoun_relatability",  0.0),
        "lf2_internal_rhyme":         best.get("internal_rhyme",        0.0),
        "lf2_energy_markers":         best.get("energy_markers",        0.0),
        "lf2_best_section_viral_score": best.get("viral_score",         0.0),
        "lf2_top_line_score":         round(max(line_scores), 4) if line_scores else 0.0,
        "lf2_avg_line_score":         round(float(np.mean(line_scores)), 4) if line_scores else 0.0,
        "lf2_sentiment_compound":     vader_compound,
        "lf2_n_sections":             len(sections),
        "lf2_n_lyric_lines":          len(lyric_lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# lf3_  :  12 TIER-1 NLP FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def get_lf3_features(raw: str, result: dict) -> dict:
    lyric_lines = _parse_lines(raw)
    if not lyric_lines:
        return {k: 0.0 for k in [
            "lf3_flesch_reading_ease", "lf3_avg_syllables_per_line",
            "lf3_std_syllables_per_line", "lf3_alliteration_density",
            "lf3_imperative_ratio", "lf3_question_density",
            "lf3_explicit_density", "lf3_lexical_density",
            "lf3_line_semantic_cohesion", "lf3_opening_line_score",
            "lf3_closing_line_score", "lf3_unique_rhyme_sounds",
        ]}

    plain = _SECTION_RE.sub("", raw).strip()

    # Flesch reading ease (no clamping — matches training data behaviour)
    try:
        flesch = float(textstat.flesch_reading_ease(plain))
    except Exception:
        flesch = float("nan")

    # Syllables per line  (uses pyphen via signals._count_syllables)
    syls_per_line = []
    for line in lyric_lines:
        words = re.findall(r"[a-z']+", line.lower())
        syls_per_line.append(sum(_count_syllables(w) for w in words))
    avg_syl = float(np.mean(syls_per_line)) if syls_per_line else 0.0
    std_syl = float(np.std(syls_per_line)) if len(syls_per_line) > 1 else 0.0

    # Alliteration density (adjacent word pairs sharing first consonant)
    allit = pairs = 0
    for line in lyric_lines:
        words = re.findall(r"[a-z]{2,}", line.lower())
        for a, b in zip(words, words[1:]):
            if a[0] not in "aeiou" and a[0] == b[0]:
                allit += 1
            pairs += 1
    alliteration_density = round(allit / max(pairs, 1), 4)

    # Imperative ratio — lines whose first word is an action verb
    imperative_count = sum(
        1 for l in lyric_lines
        if re.split(r"[^a-z]", l.lower())[0] in _ACTION_VERBS
    )
    imperative_ratio = round(imperative_count / len(lyric_lines), 4)

    # Question density
    question_density = round(
        sum(1 for l in lyric_lines if "?" in l) / len(lyric_lines), 4
    )

    # Explicit density
    all_words = re.sub(r"[^\w\s]", " ", " ".join(lyric_lines).lower()).split()
    explicit_density = round(
        sum(1 for w in all_words if w in _EXPLICIT) / max(len(all_words), 1), 4
    )

    # Lexical density (content words / total words)
    content = [w for w in all_words if w not in _FUNCTION_WORDS and len(w) > 1]
    lexical_density = round(len(content) / max(len(all_words), 1), 4)

    # Line semantic cohesion — avg TF-IDF cosine between adjacent lines
    cohesion = 0.0
    if len(lyric_lines) >= 2:
        try:
            tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
            X = tfidf.fit_transform(lyric_lines)
            sims = [
                float(cosine_similarity(X[i], X[i + 1])[0, 0])
                for i in range(len(lyric_lines) - 1)
            ]
            cohesion = round(float(np.mean(sims)), 4)
        except Exception:
            cohesion = 0.0

    # Opening / closing line scores (from pre-computed line scores in result)
    scored_lines = result.get("lines", [])
    opening_score = round(scored_lines[0]["score"], 4) if scored_lines else 0.0
    closing_score = round(scored_lines[-1]["score"], 4) if scored_lines else 0.0

    # Unique rhyme sounds (normalised by line count)
    end_sounds = [_last_vowel_sound(l) for l in lyric_lines]
    unique_sounds = len({s for s in end_sounds if len(s) >= 2})
    unique_rhyme_sounds = round(unique_sounds / max(len(lyric_lines), 1), 4)

    return {
        "lf3_flesch_reading_ease":    round(flesch, 4),
        "lf3_avg_syllables_per_line": round(avg_syl, 4),
        "lf3_std_syllables_per_line": round(std_syl, 4),
        "lf3_alliteration_density":   alliteration_density,
        "lf3_imperative_ratio":       imperative_ratio,
        "lf3_question_density":       question_density,
        "lf3_explicit_density":       explicit_density,
        "lf3_lexical_density":        lexical_density,
        "lf3_line_semantic_cohesion": cohesion,
        "lf3_opening_line_score":     opening_score,
        "lf3_closing_line_score":     closing_score,
        "lf3_unique_rhyme_sounds":    unique_rhyme_sounds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# lf_emb_  :  SENTENCE EMBEDDINGS + CHORUS/VERSE CONTRAST
# ══════════════════════════════════════════════════════════════════════════════

def get_lf_emb_features(raw: str, model, all_lyrics: list[str]) -> dict:
    """
    Compute sentence embeddings and PCA(10) reduction.

    PCA is fit on all 6 Tatiana songs together (not the original training set),
    so lf_emb_0..9 are intra-Tatiana comparable, not directly comparable to
    model training features.

    lf_chorus_verse_contrast is independent of PCA.
    """
    zero = {f"lf_emb_{i}": 0.0 for i in range(10)}
    zero["lf_chorus_verse_contrast"] = 0.0

    if model is None:
        return zero

    plain = _SECTION_RE.sub("", raw).strip()
    if not plain:
        return zero

    emb = model.encode(plain, show_progress_bar=False, normalize_embeddings=True)

    # Chorus / verse contrast (cosine distance — does not need PCA)
    chorus_text = ""
    verse_text  = ""
    current_sec = "intro"
    chorus_lines, verse_lines = [], []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        m = _SECTION_RE.match(s)
        if m:
            current_sec = m.group(1).strip().lower()
        else:
            if any(k in current_sec for k in ("chorus","hook","refrain")):
                chorus_lines.append(s)
            elif "verse" in current_sec:
                verse_lines.append(s)

    contrast = 0.0
    if chorus_lines and verse_lines:
        c_emb = model.encode(
            " ".join(chorus_lines), show_progress_bar=False, normalize_embeddings=True
        )
        v_emb = model.encode(
            " ".join(verse_lines), show_progress_bar=False, normalize_embeddings=True
        )
        sim = float(cosine_similarity(c_emb.reshape(1, -1), v_emb.reshape(1, -1))[0, 0])
        contrast = round(1.0 - sim, 4)

    return {"_raw_emb": emb, "lf_chorus_verse_contrast": contrast}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Tatiana DeMaria — Lyric Feature Extraction")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    df = df[df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")].copy()
    df = df.reset_index(drop=True)
    print(f"\n  Songs found: {len(df)}")
    for s in df["song"].tolist():
        print(f"    · {s}")

    # ── Load sentence transformer ─────────────────────────────────────────────
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        print("\n  Loading all-MiniLM-L6-v2 …")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Model loaded.")
    except Exception as e:
        print(f"  [warn] sentence_transformers not available: {e}")
        print("         lf_emb_0..9 will be 0.")

    # ── Extract features for each song ───────────────────────────────────────
    print("\n  Extracting features …")
    rows = []
    raw_embs = []

    for _, row in df.iterrows():
        song   = row["song"]
        artist = row.get("artist", "Tatiana DeMaria")
        raw    = str(row["lyrics"])
        print(f"    → {song}")

        result = _ANALYZER.analyze(raw)

        feats: dict = {"artist": artist, "song": song}
        feats.update(get_lf_features(raw))
        feats.update(get_lf2_features(raw, result))
        feats.update(get_lf3_features(raw, result))

        emb_feats = get_lf_emb_features(raw, model, [])
        raw_emb = emb_feats.pop("_raw_emb", None)
        feats.update(emb_feats)         # lf_chorus_verse_contrast only for now

        rows.append(feats)
        if raw_emb is not None:
            raw_embs.append(raw_emb)
        else:
            raw_embs.append(np.zeros(384))

    result_df = pd.DataFrame(rows)

    # ── PCA(10) on all Tatiana songs together ─────────────────────────────────
    emb_matrix = np.vstack(raw_embs)
    if model is not None and emb_matrix.any():
        n_components = min(10, emb_matrix.shape[0])
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(emb_matrix)
        var = round(sum(pca.explained_variance_ratio_) * 100, 1)
        print(f"\n  PCA({n_components}) on {emb_matrix.shape[0]} songs — "
              f"variance explained: {var}%")

        for i in range(10):
            if i < pca_coords.shape[1]:
                result_df[f"lf_emb_{i}"] = np.round(pca_coords[:, i], 6)
            else:
                result_df[f"lf_emb_{i}"] = 0.0

    # Reorder columns to match standard feature order
    id_cols   = ["artist", "song"]
    lf_cols   = [c for c in result_df.columns
                 if c.startswith("lf_")
                 and not c.startswith("lf_emb_")
                 and not c.startswith("lf2_")
                 and not c.startswith("lf3_")
                 and c != "lf_chorus_verse_contrast"]
    lf2_cols  = [c for c in result_df.columns if c.startswith("lf2_")]
    lf3_cols  = [c for c in result_df.columns if c.startswith("lf3_")]
    emb_cols  = [f"lf_emb_{i}" for i in range(10)] + ["lf_chorus_verse_contrast"]

    ordered_cols = id_cols + lf_cols + lf2_cols + lf3_cols + emb_cols
    result_df = result_df[ordered_cols]

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved → {OUTPUT_CSV}")
    print(f"  Shape : {result_df.shape[0]} songs × {result_df.shape[1]} columns")
    print(f"\n  Feature groups:")
    print(f"    lf_  (basic)    : {len(lf_cols)} features")
    print(f"    lf2_ (section)  : {len(lf2_cols)} features")
    print(f"    lf3_ (NLP)      : {len(lf3_cols)} features")
    print(f"    lf_emb (embed)  : {len(emb_cols)} features")

    # ── Quick preview ─────────────────────────────────────────────────────────
    print(f"\n  Top features preview:")
    preview_cols = ["song", "lf_vocab_richness", "lf_rhyme_density",
                    "lf_hook_strength", "lf2_best_section_viral_score",
                    "lf3_flesch_reading_ease", "lf_chorus_verse_contrast"]
    print(result_df[preview_cols].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()

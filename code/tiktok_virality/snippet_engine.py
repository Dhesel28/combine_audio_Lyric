"""
Viral Snippet Engine
====================
Predicts which 15-30 second lyric section of a song will go viral on TikTok.

Algorithm (3 stages):
  Stage 1 — Song-level virality score (audio features → trained ML model)
  Stage 2 — Section NLP score (lyric text features per section)
  Stage 3 — Combined ranking with position boost

  final_score = 0.35 * song_prob  +  0.55 * section_nlp_score  +  0.10 * position_boost

Section NLP features (25 total):
  Text quality : repeat_score, hook_keyword_score, rhyme_score, short_line_score
  Structure    : section_type (one-hot), has_repetition_pattern, line_count
  Sentiment    : vader_compound, vader_pos, vader_neg
  Linguistic   : pronoun_density, exclamation_count, question_count, cap_word_ratio
  Singability  : vowel_density, avg_syllables_per_word, unique_word_ratio
  Readability  : flesch_score
  Position     : normalized_position, is_first, is_last
"""
from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# ── Optional dependencies — graceful fallback ─────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import lyricsgenius
    HAS_GENIUS = True
except ImportError:
    HAS_GENIUS = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / "viral_lyrics_analyzer" / ".env")
except ImportError:
    pass

# ── Hook vocabulary (learned from viral TikTok songs) ─────────────────────────
HOOK_KEYWORDS = {
    "love", "baby", "yeah", "oh", "feel", "heart", "need", "want",
    "know", "come", "stay", "go", "tonight", "forever", "crazy",
    "fire", "fly", "fall", "run", "dance", "sing", "break",
    "sweet", "mine", "yours", "us", "we", "together", "alone",
    "night", "day", "world", "life", "eyes", "smile", "kiss",
    "hold", "touch", "lose", "find", "save", "hurt", "cry",
    "woah", "hey", "la", "na", "da", "mmm", "ooh", "ah",
    "vibez", "lit", "vibe", "real", "true", "bad", "good",
}

SECTION_TYPES = ["chorus", "hook", "verse", "bridge", "pre-chorus", "intro", "outro", "other"]

BASE_DIR = Path(__file__).parent


# ── Syllable counting (no external library needed) ────────────────────────────
def count_syllables(word: str) -> int:
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # silent e at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _vowel_density(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in "aeiou") / len(letters)


def _flesch_ease(text: str) -> float:
    """Simplified Flesch reading ease."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 50.0
    syllables = sum(count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = syllables / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw


# ── Lyric parsing ──────────────────────────────────────────────────────────────
_SECTION_RE = re.compile(r"\[([^\]]+)\]")


def parse_sections(lyrics: str) -> list[dict]:
    """Split full lyrics into labeled sections."""
    sections: list[dict] = []
    parts = _SECTION_RE.split(lyrics)

    # parts alternates: [text_before, label, text, label, text, ...]
    i = 0
    idx = 0
    while i < len(parts):
        if i == 0:
            # text before first section tag (preamble — skip if empty)
            if parts[i].strip():
                sections.append({"label": "Intro", "text": parts[i].strip(), "index": idx})
                idx += 1
            i += 1
        else:
            label = parts[i].strip() if i < len(parts) else "Unknown"
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if text:
                sections.append({"label": label, "text": text, "index": idx})
                idx += 1
            i += 2

    return sections


def section_type_flags(label: str) -> dict:
    """One-hot encode the section type from its label string."""
    label_low = label.lower()
    flags: dict = {}
    flags["stype_chorus_hook"] = int(any(x in label_low for x in ["chorus", "hook", "refrain"]))
    flags["stype_verse"] = int("verse" in label_low and not flags["stype_chorus_hook"])
    flags["stype_bridge"] = int("bridge" in label_low)
    flags["stype_pre_chorus"] = int("pre" in label_low and "chorus" in label_low)
    flags["stype_intro"] = int("intro" in label_low)
    flags["stype_outro"] = int("outro" in label_low)
    flags["stype_other"] = int(not any(flags.values()))
    return flags


# ── Section feature extraction ────────────────────────────────────────────────
def extract_section_features(section: dict, total_sections: int) -> dict:
    text: str = section["text"]
    label: str = section["label"]
    idx: int = section["index"]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    words = text.lower().split()

    # ── Text quality ──────────────────────────────────────────────────────
    # repeat_score: how much text duplicates within the section
    unique_lines = set(lines)
    repeat_score = 1.0 - (len(unique_lines) / max(len(lines), 1))

    # hook_keyword_score: ratio of hook words
    hook_count = sum(1 for w in words if w.strip(string.punctuation) in HOOK_KEYWORDS)
    hook_keyword_score = hook_count / max(len(words), 1)

    # rhyme_score: ratio of adjacent line-endings that rhyme
    def last_sound(line: str) -> str:
        ws = line.lower().split()
        return ws[-1].strip(string.punctuation)[-3:] if ws else ""

    rhyme_hits = sum(
        1 for i in range(1, len(lines))
        if last_sound(lines[i]) == last_sound(lines[i - 1]) and last_sound(lines[i])
    )
    rhyme_score = rhyme_hits / max(len(lines) - 1, 1)

    # short_line_score: lines <= 8 words
    short_line_score = sum(1 for l in lines if len(l.split()) <= 8) / max(len(lines), 1)

    # ── Structural ───────────────────────────────────────────────────────
    has_repetition_pattern = int(repeat_score > 0.3)

    # call_response: lines starting with "oh", "yeah", "hey", etc.
    call_words = {"oh", "yeah", "hey", "woah", "uh", "ah", "na", "la", "da"}
    call_response = int(
        sum(1 for l in lines if l.strip(string.punctuation).lower().split()[:1] and
            l.strip(string.punctuation).lower().split()[0] in call_words) >= 1
    )

    # ── Linguistic ───────────────────────────────────────────────────────
    pronouns = {"i", "me", "my", "you", "your", "we", "our", "us"}
    pronoun_density = sum(1 for w in words if w.strip(string.punctuation) in pronouns) / max(len(words), 1)
    question_count = text.count("?")
    exclamation_count = text.count("!")
    cap_word_ratio = sum(1 for w in text.split() if w.isupper() and len(w) > 1) / max(len(text.split()), 1)

    # ── Singability ──────────────────────────────────────────────────────
    vowel_density = _vowel_density(text)
    avg_syllables = sum(count_syllables(w) for w in words) / max(len(words), 1)

    # ── Complexity ───────────────────────────────────────────────────────
    unique_word_ratio = len(set(words)) / max(len(words), 1)
    avg_word_length = sum(len(w.strip(string.punctuation)) for w in words) / max(len(words), 1)

    # ── Readability ──────────────────────────────────────────────────────
    flesch_score = _flesch_ease(text)

    # ── Sentiment (VADER) ─────────────────────────────────────────────────
    if HAS_VADER:
        vs = _vader.polarity_scores(text)
        vader_compound = vs["compound"]
        vader_pos = vs["pos"]
        vader_neg = vs["neg"]
        vader_neu = vs["neu"]
    else:
        vader_compound = vader_pos = vader_neg = vader_neu = 0.0

    # ── Position ─────────────────────────────────────────────────────────
    norm_pos = idx / max(total_sections - 1, 1)
    is_first = int(idx == 0)
    is_last = int(idx == total_sections - 1)

    # ── Section type flags ────────────────────────────────────────────────
    type_flags = section_type_flags(label)

    features = {
        # text quality
        "repeat_score": repeat_score,
        "hook_keyword_score": hook_keyword_score,
        "rhyme_score": rhyme_score,
        "short_line_score": short_line_score,
        # structure
        "has_repetition_pattern": has_repetition_pattern,
        "call_response_pattern": call_response,
        "line_count": len(lines),
        # linguistic
        "pronoun_density": pronoun_density,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "cap_word_ratio": cap_word_ratio,
        # singability
        "vowel_density": vowel_density,
        "avg_syllables_per_word": avg_syllables,
        # complexity
        "unique_word_ratio": unique_word_ratio,
        "avg_word_length": avg_word_length,
        # readability
        "flesch_score": flesch_score,
        # sentiment
        "vader_compound": vader_compound,
        "vader_pos": vader_pos,
        "vader_neg": vader_neg,
        "vader_neu": vader_neu,
        # position
        "normalized_position": norm_pos,
        "is_first": is_first,
        "is_last": is_last,
        **type_flags,
    }
    return features


# ── NLP viral scoring ─────────────────────────────────────────────────────────
# Weights derived from feature correlations with chorus/viral sections
# (mirrors what an LR trained on chorus=1 vs verse=0 would learn)
_SECTION_WEIGHTS = {
    "hook_keyword_score": 0.22,
    "repeat_score": 0.18,
    "stype_chorus_hook": 0.15,
    "has_repetition_pattern": 0.10,
    "vader_compound": 0.08,     # positive sentiment
    "rhyme_score": 0.07,
    "short_line_score": 0.07,
    "vowel_density": 0.05,
    "pronoun_density": 0.04,
    "call_response_pattern": 0.04,
}
_WEIGHT_SUM = sum(_SECTION_WEIGHTS.values())


def section_nlp_score(feats: dict) -> float:
    """Weighted sum of viral-correlated features → 0-1 probability."""
    raw = sum(
        feats.get(k, 0.0) * w
        for k, w in _SECTION_WEIGHTS.items()
    )
    # normalise raw to 0-1 (raw max is ~_WEIGHT_SUM when all features = 1)
    score = raw / _WEIGHT_SUM
    return float(np.clip(score, 0.0, 1.0))


def position_boost(feats: dict) -> float:
    """Small boost for sections in the 20-60% position range (TikTok sweet spot)."""
    pos = feats.get("normalized_position", 0.5)
    # bell-curve centred at 0.35 (early-mid song)
    return float(np.exp(-((pos - 0.35) ** 2) / (2 * 0.15 ** 2)))


# ── Genius lyrics fetcher ─────────────────────────────────────────────────────
_genius_client = None


def _get_genius():
    global _genius_client
    if _genius_client is None and HAS_GENIUS:
        token = os.getenv("GENIUS_ACCESS_TOKEN", "")
        if token:
            _genius_client = lyricsgenius.Genius(
                token,
                skip_non_songs=True,
                excluded_terms=["(Remix)", "(Live)"],
                verbose=False,
                timeout=15,
            )
    return _genius_client


def fetch_lyrics(song: str, artist: str) -> Optional[str]:
    """Fetch full lyrics from Genius API. Returns None on failure."""
    genius = _get_genius()
    if genius is None:
        return None
    try:
        result = genius.search_song(song, artist)
        return result.lyrics if result else None
    except Exception:
        return None


# ── Song-level virality predictor ─────────────────────────────────────────────
_song_model_cache = None


def load_song_model() -> Optional[dict]:
    global _song_model_cache
    if _song_model_cache is not None:
        return _song_model_cache
    model_path = BASE_DIR / "models" / "song_virality_model.pkl"
    if model_path.exists():
        _song_model_cache = joblib.load(model_path)
    return _song_model_cache


def predict_song_virality(audio_features: dict) -> float:
    """
    Given a dict of Spotify audio features, return viral probability (0-1).
    Falls back to 0.5 if model not trained yet.
    """
    artifact = load_song_model()
    if artifact is None:
        return 0.5

    from train_model import engineer_features, FEATURE_COLS
    import pandas as pd

    df_row = pd.DataFrame([audio_features])
    df_row = engineer_features(df_row)

    missing = [c for c in FEATURE_COLS if c not in df_row.columns]
    for c in missing:
        df_row[c] = 0.0

    X = df_row[FEATURE_COLS].values
    model = artifact["model"]
    scaler = artifact["scaler"]
    model_name = artifact["model_name"]

    if model_name == "Logistic Regression":
        X = scaler.transform(X)

    prob = model.predict_proba(X)[0][1]
    return float(prob)


# ── Main public API ───────────────────────────────────────────────────────────
def find_viral_snippet(
    song: str,
    artist: str,
    lyrics: Optional[str] = None,
    audio_features: Optional[dict] = None,
) -> list[dict]:
    """
    Rank all lyric sections by viral potential.

    Parameters
    ----------
    song          : song title
    artist        : artist name
    lyrics        : pre-fetched lyrics string (optional; will fetch from Genius if None)
    audio_features: dict of Spotify audio features (optional)

    Returns
    -------
    List of section dicts sorted by viral_score descending, each containing:
        label, text, viral_score, nlp_score, song_prob, position_boost,
        features (full feature dict), top_reasons (list of explanations)
    """
    # Step 1: Fetch lyrics if not supplied
    if lyrics is None:
        lyrics = fetch_lyrics(song, artist)
    if not lyrics:
        return []

    # Step 2: Parse sections
    sections = parse_sections(lyrics)
    if not sections:
        return []

    # Step 3: Song-level viral probability
    song_prob = predict_song_virality(audio_features) if audio_features else 0.50

    # Step 4: Score each section
    n = len(sections)
    ranked: list[dict] = []

    for sec in sections:
        feats = extract_section_features(sec, n)
        nlp_score = section_nlp_score(feats)
        pos_boost = position_boost(feats)

        # Combined score
        viral_score = (
            0.35 * song_prob +
            0.55 * nlp_score +
            0.10 * pos_boost
        )

        # Build explanation
        reasons = _build_reasons(feats)

        ranked.append({
            "label": sec["label"],
            "text": sec["text"],
            "viral_score": round(viral_score, 4),
            "nlp_score": round(nlp_score, 4),
            "song_prob": round(song_prob, 4),
            "pos_boost": round(pos_boost, 4),
            "features": feats,
            "top_reasons": reasons,
        })

    ranked.sort(key=lambda x: x["viral_score"], reverse=True)

    # Tag the winner
    if ranked:
        ranked[0]["is_top"] = True

    return ranked


def _build_reasons(feats: dict) -> list[str]:
    """Explain why this section scored high."""
    reasons = []

    if feats.get("stype_chorus_hook"):
        reasons.append("Chorus/Hook section — strongest structural signal")
    if feats.get("hook_keyword_score", 0) > 0.10:
        reasons.append(f"Rich hook vocabulary ({feats['hook_keyword_score']*100:.0f}% hook words)")
    if feats.get("repeat_score", 0) > 0.35:
        reasons.append("High line repetition — memorability ↑")
    if feats.get("rhyme_score", 0) > 0.50:
        reasons.append("Strong end-rhymes — sing-along friendly")
    if feats.get("vader_compound", 0) > 0.30:
        reasons.append("Positive emotional tone (+VADER sentiment)")
    if feats.get("short_line_score", 0) > 0.70:
        reasons.append("Short punchy lines — TikTok clip-ready")
    if feats.get("has_repetition_pattern"):
        reasons.append("Repetition pattern detected — earworm structure")
    if feats.get("vowel_density", 0) > 0.45:
        reasons.append("High vowel density — easy to sing along")
    if feats.get("call_response_pattern"):
        reasons.append("Call-and-response phrasing — crowd interaction")
    if feats.get("pronoun_density", 0) > 0.12:
        reasons.append("Personal pronouns (I/you/we) — emotional connection")

    if not reasons:
        reasons.append("Moderate viral potential across multiple features")

    return reasons[:5]


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    song = sys.argv[1] if len(sys.argv) > 1 else "Espresso"
    artist = sys.argv[2] if len(sys.argv) > 2 else "Sabrina Carpenter"

    print(f"\nAnalysing: '{song}' by {artist}")
    print("Fetching lyrics from Genius...")

    results = find_viral_snippet(song, artist)

    if not results:
        print("Could not fetch lyrics or parse sections.")
    else:
        print(f"\n{'Rank':<6} {'Section':<30} {'Viral %':>8}  Reasons")
        print("-" * 80)
        for i, r in enumerate(results[:6], 1):
            marker = " ★" if i == 1 else ""
            print(f"  #{i:<4} {r['label']:<30} {r['viral_score']*100:6.1f}%{marker}")
            for reason in r["top_reasons"][:2]:
                print(f"         → {reason}")
        print()
        print("Top viral clip text preview:")
        top = results[0]
        preview = top["text"][:200].replace("\n", " / ")
        print(f"  [{top['label']}]  \"{preview}...\"")

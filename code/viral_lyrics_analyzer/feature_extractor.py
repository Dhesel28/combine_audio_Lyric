"""
Feature extractor: 32 features per lyric section for viral prediction.

Key function: extract_features(lines, section_index, total_sections, stype) -> dict
Exports: FEATURE_NAMES (list of 32 feature names in fixed order)
"""

from __future__ import annotations

import re
from typing import List, Optional

import pyphen
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── Singletons ─────────────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()
_dic = pyphen.Pyphen(lang="en_US")

# ── Pronouns for density calculation ──────────────────────────────────────────
_PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
             "we", "us", "our", "ours", "ourselves", "he", "she", "they"}

# ── Hook keywords (same as genius_analyzer.score_section) ─────────────────────
_HOOK_WORDS = {
    # Core emotional
    "love", "baby", "feel", "yeah", "oh", "never", "always", "tonight",
    "heart", "wanna", "gonna", "need", "want", "know", "fire", "crazy",
    # Emotional depth
    "miss", "cry", "pain", "soul", "dream", "broken", "beautiful",
    # Action / energy
    "dance", "run", "fall", "rise", "stay", "fly",
    # Relational
    "mine", "kiss", "hold", "real", "together", "alone",
    # Viral amplifiers
    "forever", "everything", "nothing", "way", "eyes", "wild",
}

# ── Section type keys (for one-hot) ───────────────────────────────────────────
_STYPE_KEYS = ["Chorus/Hook", "Verse", "Bridge", "Pre-Chorus", "Intro", "Outro", "Other"]

# ── Fixed feature name list (32 features) ─────────────────────────────────────
FEATURE_NAMES: List[str] = [
    # Existing text scores (4)
    "repeat_score", "short_line_score", "rhyme_score", "hook_keyword_score",
    # Position (4)
    "section_index", "normalized_position", "is_first_section", "is_last_section",
    # Section type one-hot (7)
    "stype_chorus_hook", "stype_verse", "stype_bridge", "stype_pre_chorus",
    "stype_intro", "stype_outro", "stype_other",
    # VADER sentiment (4)
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    # TextBlob (2)
    "textblob_polarity", "textblob_subjectivity",
    # Linguistic (4)
    "pronoun_density", "question_count", "exclamation_count", "capitalized_word_ratio",
    # Singability (2)
    "vowel_density", "avg_syllables_per_word",
    # Complexity (2)
    "unique_word_ratio", "avg_word_length",
    # Structural (2)
    "has_repetition_pattern", "call_response_pattern",
    # Readability (1)
    "flesch_reading_ease",
    # Section structure (2)
    "word_count", "line_count",
    # Emotional intensity (1)
    "emotional_intensity",
    # Hook position (2)
    "ends_strong", "starts_strong",
    # Hook quality (1)
    "hook_word_variety",
]

assert len(FEATURE_NAMES) == 38, f"Expected 38 features, got {len(FEATURE_NAMES)}"


# ── Syllable counting ──────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    """Count syllables in a word using pyphen. Falls back to vowel count."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    pairs = _dic.positions(word)
    # pyphen returns hyphen positions; syllable count = positions + 1
    return max(1, len(pairs) + 1)


def _text_syllables(text: str) -> int:
    """Total syllable count for a block of text."""
    return sum(_count_syllables(w) for w in text.split() if w.strip(".,!?;:'\""))


# ── Core text scores (mirroring genius_analyzer.score_section) ────────────────

def _text_scores(lines: List[str]) -> dict:
    """Recompute the 4 text scores from genius_analyzer without importing it."""
    full_text = " ".join(lines)
    words = full_text.lower().split()

    unique_lines = set(l.strip().lower() for l in lines if l.strip())
    repeat_score = 1 - (len(unique_lines) / max(len(lines), 1))

    avg_words = sum(len(l.split()) for l in lines) / max(len(lines), 1)
    short_score = max(0.0, 1 - avg_words / 20)

    last_words = [l.split()[-1].lower().rstrip(",.!?") for l in lines if l.split()]
    rhyme_count = 0
    for i in range(len(last_words)):
        for j in range(i + 1, len(last_words)):
            if last_words[i][-2:] == last_words[j][-2:] and last_words[i] != last_words[j]:
                rhyme_count += 1
    rhyme_score = min(1.0, rhyme_count / max(len(lines), 1))

    hook_hit = sum(1 for w in words if w in _HOOK_WORDS)
    hook_score = min(1.0, hook_hit / max(len(words), 1) * 10)

    return {
        "repeat_score": round(repeat_score, 4),
        "short_line_score": round(short_score, 4),
        "rhyme_score": round(rhyme_score, 4),
        "hook_keyword_score": round(hook_score, 4),
    }


# ── Main feature extraction ────────────────────────────────────────────────────

def extract_features(
    lines: List[str],
    section_index: int,
    total_sections: int,
    stype: str,
) -> dict:
    """
    Extract 32 features for one lyric section.

    Parameters
    ----------
    lines          : list of lyric lines in this section
    section_index  : 0-based index of this section in the song
    total_sections : total number of sections in the song
    stype          : section type string (e.g. "Chorus/Hook", "Verse", ...)

    Returns
    -------
    dict with exactly the keys in FEATURE_NAMES
    """
    full_text = " ".join(lines)
    words_raw = full_text.split()
    words_lower = [w.lower().strip(".,!?;:'\"") for w in words_raw]
    n_words = max(len(words_lower), 1)

    # ── Text scores ──────────────────────────────────────────────────────────
    ts = _text_scores(lines)

    # ── Position ─────────────────────────────────────────────────────────────
    norm_pos = section_index / max(total_sections - 1, 1)
    is_first = 1 if section_index == 0 else 0
    is_last = 1 if section_index == total_sections - 1 else 0

    # ── Section type one-hot ─────────────────────────────────────────────────
    stype_norm = stype.strip()
    stype_chorus_hook = 1 if stype_norm == "Chorus/Hook" else 0
    stype_verse = 1 if stype_norm == "Verse" else 0
    stype_bridge = 1 if stype_norm == "Bridge" else 0
    stype_pre_chorus = 1 if stype_norm == "Pre-Chorus" else 0
    stype_intro = 1 if stype_norm == "Intro" else 0
    stype_outro = 1 if stype_norm == "Outro" else 0
    stype_other = 1 if stype_norm == "Other" else 0

    # ── VADER sentiment ───────────────────────────────────────────────────────
    vader_scores = _vader.polarity_scores(full_text)

    # ── TextBlob ──────────────────────────────────────────────────────────────
    blob = TextBlob(full_text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    # ── Linguistic ────────────────────────────────────────────────────────────
    pronoun_count = sum(1 for w in words_lower if w in _PRONOUNS)
    pronoun_density = pronoun_count / n_words
    question_count = full_text.count("?")
    exclamation_count = full_text.count("!")
    cap_words = sum(1 for w in words_raw if w and w[0].isupper() and not w.isupper())
    capitalized_word_ratio = cap_words / n_words

    # ── Singability ───────────────────────────────────────────────────────────
    vowels = sum(1 for ch in full_text.lower() if ch in "aeiou")
    vowel_density = vowels / max(len(full_text.replace(" ", "")), 1)
    total_syllables = _text_syllables(full_text)
    avg_syllables_per_word = total_syllables / n_words

    # ── Complexity ────────────────────────────────────────────────────────────
    unique_words = set(words_lower)
    unique_word_ratio = len(unique_words) / n_words
    avg_word_length = sum(len(w) for w in words_lower) / n_words

    # ── Structural ────────────────────────────────────────────────────────────
    # has_repetition_pattern: any line appears 2+ times
    line_counts: dict[str, int] = {}
    for ln in lines:
        key = ln.strip().lower()
        line_counts[key] = line_counts.get(key, 0) + 1
    has_repetition_pattern = 1 if any(c >= 2 for c in line_counts.values()) else 0

    # call_response_pattern: alternating short/long lines suggests call-and-response
    if len(lines) >= 4:
        lengths = [len(l.split()) for l in lines]
        alternations = sum(
            1 for i in range(len(lengths) - 1)
            if abs(lengths[i] - lengths[i + 1]) >= 3
        )
        call_response_pattern = 1 if alternations >= len(lines) // 2 else 0
    else:
        call_response_pattern = 0

    # ── Readability: Flesch Reading Ease ─────────────────────────────────────
    n_sentences = max(len(re.split(r"[.!?]+", full_text.strip())), 1)
    # FRE = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (total_syllables / n_words)
    # Clamp to [0, 100]
    flesch_reading_ease = max(0.0, min(100.0, fre))

    # ── Section structure ─────────────────────────────────────────────────────
    word_count = len(words_lower)
    line_count = len(lines)

    # ── Emotional intensity (absolute sentiment strength) ─────────────────────
    emotional_intensity = round(abs(vader_scores["compound"]), 4)

    # ── Hook position: do first / last lines contain hook words? ─────────────
    def _line_has_hook(line: str) -> int:
        return 1 if any(w in _HOOK_WORDS for w in line.lower().split()) else 0

    starts_strong = _line_has_hook(lines[0]) if lines else 0
    ends_strong = _line_has_hook(lines[-1]) if lines else 0

    # ── Hook word variety: unique hook words used ─────────────────────────────
    hook_word_variety = len({w for w in words_lower if w in _HOOK_WORDS})

    # ── Assemble in FEATURE_NAMES order ──────────────────────────────────────
    return {
        "repeat_score": ts["repeat_score"],
        "short_line_score": ts["short_line_score"],
        "rhyme_score": ts["rhyme_score"],
        "hook_keyword_score": ts["hook_keyword_score"],
        "section_index": section_index,
        "normalized_position": round(norm_pos, 4),
        "is_first_section": is_first,
        "is_last_section": is_last,
        "stype_chorus_hook": stype_chorus_hook,
        "stype_verse": stype_verse,
        "stype_bridge": stype_bridge,
        "stype_pre_chorus": stype_pre_chorus,
        "stype_intro": stype_intro,
        "stype_outro": stype_outro,
        "stype_other": stype_other,
        "vader_compound": round(vader_scores["compound"], 4),
        "vader_pos": round(vader_scores["pos"], 4),
        "vader_neg": round(vader_scores["neg"], 4),
        "vader_neu": round(vader_scores["neu"], 4),
        "textblob_polarity": round(tb_polarity, 4),
        "textblob_subjectivity": round(tb_subjectivity, 4),
        "pronoun_density": round(pronoun_density, 4),
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "capitalized_word_ratio": round(capitalized_word_ratio, 4),
        "vowel_density": round(vowel_density, 4),
        "avg_syllables_per_word": round(avg_syllables_per_word, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "avg_word_length": round(avg_word_length, 4),
        "has_repetition_pattern": has_repetition_pattern,
        "call_response_pattern": call_response_pattern,
        "flesch_reading_ease": round(flesch_reading_ease, 4),
        "word_count": word_count,
        "line_count": line_count,
        "emotional_intensity": emotional_intensity,
        "ends_strong": ends_strong,
        "starts_strong": starts_strong,
        "hook_word_variety": hook_word_variety,
    }


if __name__ == "__main__":
    # Quick sanity check
    test_lines = [
        "And I can't help falling in love with you",
        "Take my hand, take my whole life too",
        "For I can't help falling in love with you",
        "And I can't help falling in love with you",
    ]
    feats = extract_features(test_lines, section_index=1, total_sections=4, stype="Chorus/Hook")
    assert set(feats.keys()) == set(FEATURE_NAMES), "Feature name mismatch!"
    print("Features extracted successfully:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    print(f"\nTotal features: {len(feats)} ✓ (expected 38)")

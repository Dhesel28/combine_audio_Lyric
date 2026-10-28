"""
lyric_analyzer/signals.py
=========================
All NLP signal functions — section-level (12) and line-level (6).

Section-level signals
---------------------
  hook_vocabulary       : hook keyword density
  cross_song_repetition : lines that repeat across the full song
  internal_repetition   : bigram/trigram repeats within a section
  sentiment_extremity   : absolute VADER compound score
  end_rhyme_density     : fraction of end-word pairs that rhyme
  line_conciseness      : short punchy lines (TikTok-clip friendly)
  syllable_consistency  : low CV in syllable counts per line
  phonetic_openness     : vowel density
  pronoun_relatability  : I/me/you/we density
  internal_rhyme        : rhyming words within a single line
  energy_markers        : !/? / ALL-CAPS word count
  position_score        : bell curve centred at 40% into song

Line-level signals
------------------
  repetition      : how many times this exact line recurs in the song
  energy          : high-energy word density
  rhyme           : end-rhymes with adjacent line
  brevity         : 4-9 words = optimal for lip-sync / text overlay
  section_bonus   : lives in a Chorus / Hook / Bridge / Pre-Chorus
  emotional_punch : strong positive OR negative sentiment
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import pyphen
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import HOOK_WORDS, ENERGY_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS

_vader = SentimentIntensityAnalyzer()
_dic = pyphen.Pyphen(lang="en_US")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION-LEVEL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def sig_hook_vocabulary(lines: List[str]) -> float:
    """Hook keyword density. min(1, hits/words * 8)."""
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in HOOK_WORDS)
    return round(min(1.0, hits / len(words) * 8), 4)


def sig_cross_song_repetition(lines: List[str], raw_lyrics: str) -> float:
    """Lines that appear 2+ times in the full song (true chorus lines)."""
    raw_lower = raw_lyrics.lower()
    qualifying = [l.strip() for l in lines if len(l.strip()) > 10]
    if not qualifying:
        return 0.0
    repeated = sum(1 for l in qualifying if raw_lower.count(l.lower()) >= 2)
    return round(min(1.0, repeated / len(qualifying)), 4)


def sig_internal_repetition(lines: List[str]) -> float:
    """Bigrams / trigrams that repeat within the section (oh oh oh, baby baby)."""
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if len(words) < 4:
        return 0.0
    bigrams = Counter(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))
    trigrams = Counter(
        f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)
    )
    repeated = (
        sum(1 for c in bigrams.values() if c >= 2)
        + sum(1 for c in trigrams.values() if c >= 2)
    )
    total = len(bigrams) + len(trigrams)
    return round(min(1.0, repeated / max(total, 1) * 3), 4)


def sig_sentiment_extremity(lines: List[str]) -> float:
    """Absolute VADER compound — extreme emotion is more memorable."""
    scores = _vader.polarity_scores(" ".join(lines))
    return round(abs(scores["compound"]), 4)


def sig_end_rhyme_density(lines: List[str]) -> float:
    """Fraction of end-word pairs that rhyme (last 2 chars match, words differ)."""
    last_words: List[str] = []
    for line in lines:
        wds = re.findall(r"[a-z']+", line.lower())
        if wds:
            last_words.append(wds[-1])
    if len(last_words) < 2:
        return 0.0
    rhyme_pairs = total_pairs = 0
    for i in range(len(last_words)):
        for j in range(i + 1, len(last_words)):
            total_pairs += 1
            if last_words[i][-2:] == last_words[j][-2:] and last_words[i] != last_words[j]:
                rhyme_pairs += 1
    return round(min(1.0, rhyme_pairs / max(total_pairs, 1)), 4)


def sig_line_conciseness(lines: List[str]) -> float:
    """Short punchy lines (5-8 words peak) fit in a 15-30s TikTok clip."""
    if not lines:
        return 0.0
    avg = sum(len(re.findall(r"[a-z']+", l.lower())) for l in lines) / len(lines)
    if avg <= 3:
        return round(avg / 3 * 0.55, 4)
    if avg <= 8:
        return round(0.55 + (avg - 3) / 5 * 0.45, 4)
    return round(max(0.0, 1.0 - (avg - 8) / 14), 4)


def _count_syllables(word: str) -> int:
    w = word.lower().strip(".,!?;:'\"")
    if not w:
        return 0
    return max(1, len(_dic.positions(w)) + 1)


def sig_syllable_consistency(lines: List[str]) -> float:
    """Low CV in syllable counts per line = consistent rhythm = easy to loop."""
    counts: List[int] = []
    for line in lines:
        wds = re.findall(r"[a-z']+", line.lower())
        syl = sum(_count_syllables(w) for w in wds)
        if syl > 0:
            counts.append(syl)
    if len(counts) < 2:
        return 0.5
    mean = sum(counts) / len(counts)
    std = math.sqrt(sum((x - mean) ** 2 for x in counts) / len(counts))
    cv = std / max(mean, 1)
    return round(1.0 / (1.0 + cv), 4)


def sig_phonetic_openness(lines: List[str]) -> float:
    """Vowel density — open sounds are easier to sing and sustain."""
    text = "".join(lines).replace(" ", "")
    if not text:
        return 0.0
    vowels = sum(1 for ch in text.lower() if ch in "aeiou")
    return round(vowels / len(text), 4)


def sig_pronoun_relatability(lines: List[str]) -> float:
    """I/me/my/you/we density — personal = shareable. Scaled: 20% → 1.0."""
    PRONOUNS = {"i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our"}
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in PRONOUNS)
    return round(min(1.0, hits / len(words) * 5), 4)


def sig_internal_rhyme(lines: List[str]) -> float:
    """Fraction of lines containing at least one non-adjacent rhyming word pair."""
    count = total = 0
    for line in lines:
        words = re.findall(r"[a-z']+", line.lower())
        if len(words) < 4:
            continue
        total += 1
        found = False
        for i in range(len(words)):
            if found:
                break
            for j in range(i + 2, len(words)):
                if (len(words[i]) > 2 and len(words[j]) > 2
                        and words[i][-2:] == words[j][-2:]
                        and words[i] != words[j]):
                    found = True
                    break
        if found:
            count += 1
    return round(count / max(total, 1), 4) if total > 0 else 0.0


def sig_energy_markers(lines: List[str]) -> float:
    """Exclamation marks, question marks, and ALL-CAPS words signal urgency."""
    text = " ".join(lines)
    excl = text.count("!")
    ques = text.count("?")
    caps = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    score = (excl + ques * 0.5 + caps * 0.5) / max(len(lines), 1)
    return round(min(1.0, score / 2), 4)


def sig_position_score(idx: int, total: int) -> float:
    """Bell curve centred at 40% position. Choruses typically land at 25-65%."""
    if total <= 1:
        return 0.5
    pos = idx / (total - 1)
    return round(math.exp(-((pos - 0.40) ** 2) / (2 * 0.20 ** 2)), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  LINE-LEVEL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def _last_suffix(line: str, n: int = 3) -> str:
    words = line.strip().split()
    return words[-1].lower()[-n:] if words else ""


def line_sig_repetition(line: str, line_counts: Counter, max_count: int) -> float:
    """Normalised repetition count for this line across the whole song."""
    count = line_counts[line.strip().lower()]
    return (count - 1) / max(max_count - 1, 1) if max_count > 1 else 0.0


def line_sig_energy(line: str) -> float:
    """High-energy word density."""
    words = set(line.lower().split())
    return round(min(len(words & ENERGY_WORDS) / 3.0, 1.0), 4)


def line_sig_rhyme(line: str, prev: str, nxt: str) -> float:
    """End-rhymes with the previous or next line."""
    ea = _last_suffix(line)
    ep = _last_suffix(prev)
    en = _last_suffix(nxt)
    return 1.0 if (len(ea) >= 2 and (ea == ep or ea == en)) else 0.0


def line_sig_brevity(line: str) -> float:
    """4-9 words = optimal for lip-sync / text overlay."""
    n = len(line.split())
    if 4 <= n <= 9:
        return 1.0
    if n < 4:
        return round(0.5 + n * 0.1, 4)
    return round(max(0.0, 1.0 - (n - 9) * 0.08), 4)


def line_sig_section_bonus(section: str, hook_keys: set) -> float:
    """1.0 if the line lives in a chorus / hook / bridge / pre-chorus."""
    return 1.0 if any(k in section for k in hook_keys) else 0.0


def line_sig_emotional_punch(line: str) -> float:
    """Abs(positive - negative) hit count — emotional regardless of polarity."""
    words = set(line.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return round(min(abs(pos - neg) / 3.0, 1.0), 4)

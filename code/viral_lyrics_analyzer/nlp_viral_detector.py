"""
NLP Viral Section Detector — Standalone
========================================
Which lyric section of a song is most likely to go viral on TikTok?

12 NLP signals (weights sum to 1.0):

  Signal                  Weight  What it measures
  ─────────────────────── ──────  ────────────────────────────────────────────
  hook_vocabulary          0.15   Hook keyword density (love/yeah/baby/oh...)
  cross_song_repetition    0.13   Lines that repeat across the whole song = hook
  internal_repetition      0.12   Bigram/trigram repeats within section (oh oh oh)
  sentiment_extremity      0.10   |VADER compound| — extreme emotion = memorable
  end_rhyme_density        0.09   End-word rhyme scheme (AABB / ABAB / ABCB)
  line_conciseness         0.09   Short punchy lines = TikTok clip-friendly
  syllable_consistency     0.08   Low syllable-variance per line = rhythmic
  phonetic_openness        0.07   Vowel density — open sounds = singable
  pronoun_relatability     0.07   I/me/you/we density — personal = shareable
  internal_rhyme           0.05   Rhyming words within a single line
  energy_markers           0.03   !/? / ALL-CAPS word count
  position_score           0.02   Bell curve: chorus appears ~40% into song

Usage:
  python nlp_viral_detector.py "Espresso" "Sabrina Carpenter"
  python nlp_viral_detector.py "APT." "ROSÉ" --save
  python nlp_viral_detector.py "Cruel Summer" "Taylor Swift"
"""

from __future__ import annotations

import re
import sys
import math
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyphen
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lyricsgenius
import pandas as pd
from dotenv import load_dotenv
import os

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")

_vader = SentimentIntensityAnalyzer()
_dic = pyphen.Pyphen(lang="en_US")

SECTION_RE = re.compile(r"\[([^\]]+)\]")

# ── Hook vocabulary (domain-specific viral pop/hip-hop keywords) ───────────────
HOOK_WORDS = {
    # Core emotional anchors
    "love", "baby", "feel", "yeah", "oh", "never", "always", "tonight",
    "heart", "wanna", "gonna", "need", "want", "know", "fire", "crazy",
    # Emotional depth
    "miss", "cry", "pain", "soul", "dream", "broken", "beautiful", "hurt",
    "sorry", "lonely", "empty", "lost", "scared", "brave",
    # Energy / action
    "dance", "run", "fall", "rise", "stay", "fly", "let", "go", "burn",
    # Relational
    "mine", "kiss", "hold", "real", "together", "alone", "close", "touch",
    # Viral amplifiers
    "forever", "everything", "nothing", "way", "eyes", "wild", "world",
    # TikTok-era signals
    "ooh", "wow", "show", "glow", "flow", "free", "deep", "high", "life",
    "right", "light", "night", "fight", "bright", "tight",
}

# ── Signal weights (must sum to 1.0) ──────────────────────────────────────────
WEIGHTS: Dict[str, float] = {
    "hook_vocabulary":       0.15,
    "cross_song_repetition": 0.13,
    "internal_repetition":   0.12,
    "sentiment_extremity":   0.10,
    "end_rhyme_density":     0.09,
    "line_conciseness":      0.09,
    "syllable_consistency":  0.08,
    "phonetic_openness":     0.07,
    "pronoun_relatability":  0.07,
    "internal_rhyme":        0.05,
    "energy_markers":        0.03,
    "position_score":        0.02,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ── Genius helpers ─────────────────────────────────────────────────────────────

def _clean_header(text: str) -> str:
    """Strip Genius page metadata (contributor counts, translations) before actual lyrics."""
    lines = text.split("\n")
    cleaned: List[str] = []
    in_header = True
    for line in lines:
        s = line.strip()
        if in_header:
            if SECTION_RE.match(s):
                in_header = False
                cleaned.append(s)
            elif any(kw in s for kw in [
                "Contributors", "Translations", "Read More",
                "Embed", "You might also like", "Lyrics",
            ]):
                continue
            elif len(s) > 5 and not any(c.isdigit() for c in s[:5]):
                in_header = False
                cleaned.append(s)
        else:
            cleaned.append(s)
    return "\n".join(cleaned)


def auto_label_sections(raw_lyrics: str, verse_size: int = 4) -> str:
    """
    Add [Section] headers to raw lyrics that have none (e.g. no [Chorus] tags).

    Algorithm
    ---------
    1. Split into lines (strip blanks).
    2. Slide a window of sizes 8→2 to find the largest block of lines that
       appears 2+ times verbatim → that is the Chorus.
    3. Walk through the song line by line:
       - When a chorus block starts, flush any pending verse lines and emit [Chorus].
       - Accumulate remaining lines into verse chunks of `verse_size` lines each.
    4. If no repeated block found, split uniformly into [Verse N] chunks.

    Returns a string with [Section] labels injected, ready for _parse_sections().
    """
    lines = [l.strip() for l in raw_lyrics.split("\n") if l.strip()]
    n = len(lines)
    if n == 0:
        return raw_lyrics

    # ── Find largest repeated block ───────────────────────────────────────────
    chorus_lines: List[str] = []
    chorus_len = 0
    for block_size in range(min(8, n // 2), 1, -1):
        seen: Dict[str, int] = {}
        for i in range(n - block_size + 1):
            key = "\n".join(l.lower() for l in lines[i: i + block_size])
            seen[key] = seen.get(key, 0) + 1
        best = max(seen.items(), key=lambda x: x[1])
        if best[1] >= 2:
            chorus_lines = lines[
                next(
                    i for i in range(n - block_size + 1)
                    if "\n".join(l.lower() for l in lines[i: i + block_size]) == best[0]
                ): next(
                    i for i in range(n - block_size + 1)
                    if "\n".join(l.lower() for l in lines[i: i + block_size]) == best[0]
                ) + block_size
            ]
            chorus_len = block_size
            break

    # ── Label ─────────────────────────────────────────────────────────────────
    result: List[str] = []
    verse_num = 0
    verse_buf: List[str] = []
    chorus_key = "\n".join(l.lower() for l in chorus_lines) if chorus_lines else ""

    def flush_verse() -> None:
        nonlocal verse_num
        if verse_buf:
            verse_num += 1
            result.append(f"[Verse {verse_num}]")
            result.extend(verse_buf)
            verse_buf.clear()

    if not chorus_lines:
        # No repetition — uniform verse chunks
        for i in range(0, n, verse_size):
            verse_num += 1
            result.append(f"[Verse {verse_num}]")
            result.extend(lines[i: i + verse_size])
        return "\n".join(result)

    i = 0
    while i < n:
        if i + chorus_len <= n:
            candidate = "\n".join(l.lower() for l in lines[i: i + chorus_len])
            if candidate == chorus_key:
                flush_verse()
                result.append("[Chorus]")
                result.extend(lines[i: i + chorus_len])
                i += chorus_len
                continue
        verse_buf.append(lines[i])
        if len(verse_buf) >= verse_size:
            flush_verse()
        i += 1

    flush_verse()
    return "\n".join(result)


def _parse_sections(raw: str) -> Dict[str, List[str]]:
    """
    Split raw Genius lyrics into {label: [lines]} keeping duplicate labels merged.
    e.g. [Chorus] appearing twice → all lines under one '[Chorus]' key.
    """
    raw = _clean_header(raw)
    sections: Dict[str, List[str]] = {}
    current, buf = "[Intro]", []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        if SECTION_RE.match(s):
            if buf:
                sections.setdefault(current, []).extend(buf)
            current, buf = s, []
        else:
            buf.append(s)
    if buf:
        sections.setdefault(current, []).extend(buf)
    # Drop near-empty sections
    return {k: v for k, v in sections.items() if len(v) >= 2}


def _section_type(label: str) -> str:
    lbl = label.lower()
    if "chorus" in lbl or "hook" in lbl or "refrain" in lbl:
        return "Chorus/Hook"
    if "verse" in lbl:
        return "Verse"
    if "bridge" in lbl:
        return "Bridge"
    if "pre-chorus" in lbl or "pre chorus" in lbl:
        return "Pre-Chorus"
    if "outro" in lbl:
        return "Outro"
    if "intro" in lbl:
        return "Intro"
    return "Other"


def _count_syllables(word: str) -> int:
    """Syllable count via pyphen; falls back to 1."""
    w = word.lower().strip(".,!?;:'\"")
    if not w:
        return 0
    return max(1, len(_dic.positions(w)) + 1)


# ── NLP Signal Implementations ─────────────────────────────────────────────────

def sig_hook_vocabulary(lines: List[str]) -> float:
    """
    Hook keyword density.
    Formula: min(1.0, (hook_hits / total_words) * 8)
    Rationale: in a 40-word chorus, 5 hook words (12.5%) → score 1.0
    """
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in HOOK_WORDS)
    return round(min(1.0, hits / len(words) * 8), 4)


def sig_cross_song_repetition(lines: List[str], raw_lyrics: str) -> float:
    """
    Lines that appear 2+ times in the full raw lyrics = the true hook.

    Genius writes chorus lyrics once but the raw text shows how many times
    the song actually repeats them. Searching the raw string catches this.
    Only lines longer than 10 chars are checked (avoid false matches on 'oh').
    """
    raw_lower = raw_lyrics.lower()
    qualifying = [l.strip() for l in lines if len(l.strip()) > 10]
    if not qualifying:
        return 0.0
    repeated = sum(1 for l in qualifying if raw_lower.count(l.lower()) >= 2)
    return round(min(1.0, repeated / len(qualifying)), 4)


def sig_internal_repetition(lines: List[str]) -> float:
    """
    Bigrams and trigrams that appear 2+ times WITHIN the section.
    'oh oh oh', 'baby baby', 'I need I need' → hook signature.
    Score = (repeated_ngrams / total_unique_ngrams) * 3 (scaled up).
    """
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if len(words) < 4:
        return 0.0
    bigrams = Counter(
        f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)
    )
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
    """
    Absolute VADER compound score.
    Extreme emotion — very positive ('I love you forever') or
    very negative ('I can't breathe without you') — is more memorable
    and more likely to hit emotionally on TikTok.
    """
    scores = _vader.polarity_scores(" ".join(lines))
    return round(abs(scores["compound"]), 4)


def sig_end_rhyme_density(lines: List[str]) -> float:
    """
    Fraction of end-word pairs that rhyme (last 2 chars match, words differ).
    Measures AABB / ABAB / ABCB rhyme schemes.
    """
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
            if (last_words[i][-2:] == last_words[j][-2:]
                    and last_words[i] != last_words[j]):
                rhyme_pairs += 1
    return round(min(1.0, rhyme_pairs / max(total_pairs, 1)), 4)


def sig_line_conciseness(lines: List[str]) -> float:
    """
    Short, punchy lines fit in a 15-30s TikTok clip.
    Score peaks at 5-8 words/line; longer lines get penalised.
    """
    if not lines:
        return 0.0
    avg = sum(
        len(re.findall(r"[a-z']+", l.lower())) for l in lines
    ) / len(lines)
    if avg <= 3:
        return round(avg / 3 * 0.55, 4)
    if avg <= 8:
        return round(0.55 + (avg - 3) / 5 * 0.45, 4)
    return round(max(0.0, 1.0 - (avg - 8) / 14), 4)


def sig_syllable_consistency(lines: List[str]) -> float:
    """
    Low coefficient of variation in syllable counts per line = rhythmic grid.
    Consistent rhythm makes a section easy to sing along with (and loop on TikTok).
    Score = 1 / (1 + cv).
    """
    counts: List[int] = []
    for line in lines:
        wds = re.findall(r"[a-z']+", line.lower())
        syl = sum(_count_syllables(w) for w in wds)
        if syl > 0:
            counts.append(syl)
    if len(counts) < 2:
        return 0.5  # not enough data → neutral
    mean = sum(counts) / len(counts)
    std = math.sqrt(sum((x - mean) ** 2 for x in counts) / len(counts))
    cv = std / max(mean, 1)
    return round(1.0 / (1.0 + cv), 4)


def sig_phonetic_openness(lines: List[str]) -> float:
    """
    Vowel density in the section text.
    Open vowel sounds (a, e, i, o, u) are easier to sustain, belt, and
    loop — the physical reason why choruses tend to be vowel-heavy.
    """
    text = "".join(lines).replace(" ", "")
    if not text:
        return 0.0
    vowels = sum(1 for ch in text.lower() if ch in "aeiou")
    return round(vowels / len(text), 4)


def sig_pronoun_relatability(lines: List[str]) -> float:
    """
    1st- and 2nd-person pronoun density: I / me / my / you / your / we / us.
    'You make me feel alive' is more shareable than 'She makes him feel alive'.
    Scaled: 20% pronoun rate → 1.0.
    """
    PRONOUNS = {"i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our"}
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in PRONOUNS)
    return round(min(1.0, hits / len(words) * 5), 4)


def sig_internal_rhyme(lines: List[str]) -> float:
    """
    Fraction of lines that contain at least one pair of rhyming non-adjacent words.
    e.g. 'I'm on fire, yeah my heart burns higher' (fire / higher).
    Only lines with ≥4 words are evaluated.
    """
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
    """
    Exclamation marks (1.0), question marks (0.5), and ALL-CAPS words (0.5)
    signal excitement and urgency. Normalised per line.
    """
    text = " ".join(lines)
    excl = text.count("!")
    ques = text.count("?")
    caps = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    score = (excl + ques * 0.5 + caps * 0.5) / max(len(lines), 1)
    return round(min(1.0, score / 2), 4)


def sig_position_score(idx: int, total: int) -> float:
    """
    Bell curve centred at 40% position in the song.
    Choruses typically land at 25-65% (after intro + verse 1).
    Intros and outros score near 0.
    """
    if total <= 1:
        return 0.5
    pos = idx / (total - 1)
    return round(math.exp(-((pos - 0.40) ** 2) / (2 * 0.20 ** 2)), 4)


# ── Composite scorer ───────────────────────────────────────────────────────────

def score_section(
    lines: List[str],
    section_index: int,
    total_sections: int,
    raw_lyrics: str,
) -> Dict[str, float]:
    """
    Compute all 12 NLP signals and the weighted viral_score.

    Parameters
    ----------
    lines          : lyric lines in this section
    section_index  : 0-based position in the song
    total_sections : total sections in the song
    raw_lyrics     : full raw lyrics string (for cross-song repetition check)
    """
    signals: Dict[str, float] = {
        "hook_vocabulary":       sig_hook_vocabulary(lines),
        "cross_song_repetition": sig_cross_song_repetition(lines, raw_lyrics),
        "internal_repetition":   sig_internal_repetition(lines),
        "sentiment_extremity":   sig_sentiment_extremity(lines),
        "end_rhyme_density":     sig_end_rhyme_density(lines),
        "line_conciseness":      sig_line_conciseness(lines),
        "syllable_consistency":  sig_syllable_consistency(lines),
        "phonetic_openness":     sig_phonetic_openness(lines),
        "pronoun_relatability":  sig_pronoun_relatability(lines),
        "internal_rhyme":        sig_internal_rhyme(lines),
        "energy_markers":        sig_energy_markers(lines),
        "position_score":        sig_position_score(section_index, total_sections),
    }
    viral_score = sum(WEIGHTS[k] * v for k, v in signals.items())
    return {**signals, "viral_score": round(viral_score, 4)}


# ── Main detector ──────────────────────────────────────────────────────────────

def detect_viral_section(song_title: str, artist: str) -> Optional[pd.DataFrame]:
    """
    Full pipeline: Genius fetch → section parse → NLP scoring → ranked DataFrame.

    Returns DataFrame sorted by viral_score (descending), or None if song not found.
    """
    if not GENIUS_TOKEN:
        raise ValueError(
            "GENIUS_ACCESS_TOKEN not set. Add it to viral_lyrics_analyzer/.env"
        )

    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        verbose=False,
    )
    genius.remove_section_headers = False  # keep [Chorus], [Verse] labels

    print(f'\nFetching lyrics for "{song_title}" by {artist}...')
    song = genius.search_song(song_title, artist)
    if not song:
        print("  Song not found on Genius.")
        return None

    print(f"  Found: {song.title}  (Genius ID: {song.id})")

    raw_lyrics = song.lyrics  # full raw string — used for cross-song repetition
    sections = _parse_sections(raw_lyrics)
    if not sections:
        print("  No lyric sections found in lyrics.")
        return None

    labels = list(sections.keys())
    total = len(labels)
    print(f"  Parsed {total} sections: {', '.join(labels)}\n")

    rows = []
    for idx, label in enumerate(labels):
        lines = sections[label]
        stype = _section_type(label)
        scores = score_section(lines, idx, total, raw_lyrics)
        rows.append({
            "section":     label,
            "type":        stype,
            "lines":       len(lines),
            "preview":     " / ".join(lines[:2]),
            "full_lyrics": "\n".join(lines),
            **scores,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("viral_score", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


# ── Pretty terminal output ──────────────────────────────────────────────────────

# Human-readable names for each signal
_SIGNAL_NAMES = {
    "hook_vocabulary":       "Hook Keywords",
    "cross_song_repetition": "Cross-Song Repetition",
    "internal_repetition":   "Internal N-gram Repeat",
    "sentiment_extremity":   "Emotional Extremity",
    "end_rhyme_density":     "End Rhyme Density",
    "line_conciseness":      "Line Conciseness",
    "syllable_consistency":  "Syllable Rhythm",
    "phonetic_openness":     "Phonetic Openness",
    "pronoun_relatability":  "Pronoun Relatability",
    "internal_rhyme":        "Internal Rhyme",
    "energy_markers":        "Energy Markers (!/?)",
    "position_score":        "Song Position",
}

# Insight strings: (high_explanation, low_explanation)
_INSIGHTS = {
    "hook_vocabulary":       ("dense hook vocab: 'love','yeah','oh','wanna'...",
                              "few hook keywords — more narrative/descriptive"),
    "cross_song_repetition": ("these lines repeat across the song → clear hook",
                              "these lines appear only once → not the chorus"),
    "internal_repetition":   ("bigrams repeat within section ('oh oh','baby baby')",
                              "no strong internal repetition"),
    "sentiment_extremity":   ("strong emotion (positive or negative) — unforgettable",
                              "low emotional charge — won't hit as hard"),
    "end_rhyme_density":     ("tight end-rhyme scheme (AABB / ABAB)",
                              "loose or no end rhymes"),
    "line_conciseness":      ("short punchy lines — TikTok clip-friendly",
                              "long lines — harder to clip in 15-30s"),
    "syllable_consistency":  ("consistent syllable count/line — easy sing-along rhythm",
                              "irregular syllable rhythm — harder to loop"),
    "phonetic_openness":     ("high vowel density — open sounds, easy to sing/belt",
                              "consonant-heavy — less singable"),
    "pronoun_relatability":  ("high I/me/you/we — personal and relatable",
                              "low personal pronouns — less relatable"),
    "internal_rhyme":        ("rhyming words within single lines (musical)",
                              "no internal rhymes detected"),
    "energy_markers":        ("exclamation / CAPS signals excitement",
                              "low energy markers"),
    "position_score":        ("chorus-zone position (25-65% into song)",
                              "intro or outro position — less expected"),
}


def _bar(value: float, width: int = 14) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_results(df: pd.DataFrame, song_title: str, artist: str) -> None:
    W = 74
    print("\n" + "=" * W)
    print(f'  NLP Viral Section Detector  |  "{song_title}" by {artist}')
    print("=" * W)

    # ── Ranked section summary ───────────────────────────────────────────────
    print(f"\n  {'Rk':<4} {'Section':<22} {'Type':<14} {'Score':>6}  {'Lines':>5}  Preview")
    print("  " + "─" * (W - 2))
    for _, row in df.iterrows():
        marker = "  ◄ VIRAL" if row["rank"] == 1 else ""
        preview = str(row["preview"])[:26]
        print(
            f"  {int(row['rank']):<4} {row['section']:<22} {row['type']:<14} "
            f"{row['viral_score']:>6.3f}  {int(row['lines']):>5}  {preview}...{marker}"
        )

    # ── Deep-dive on top section ─────────────────────────────────────────────
    top = df.iloc[0]
    print("\n" + "═" * W)
    print(
        f"  PREDICTED VIRAL SECTION: {top['section']}  "
        f"(type: {top['type']}, score: {top['viral_score']:.3f})"
    )
    print("═" * W)

    print(f"\n  {'Signal':<28} {'Bar':>16}  {'Score':>6}  {'Level':<5}  Insight")
    print("  " + "─" * (W - 2))
    for sig, name in _SIGNAL_NAMES.items():
        val = float(top[sig])
        wt = WEIGHTS[sig]
        level = "HIGH" if val >= 0.65 else ("MED " if val >= 0.35 else "LOW ")
        hi, lo = _INSIGHTS[sig]
        insight = hi if val >= 0.50 else lo
        weighted_contrib = round(wt * val, 4)
        print(
            f"  {name:<28} {_bar(val):>16}  {val:>6.3f}  {level}  "
            f"{insight[:38]}"
        )

    # ── Weight contribution breakdown ────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print(f"  {'Signal':<28}  {'Weight':>7}  {'Score':>6}  {'Contribution':>12}")
    print(f"  {'─'*40}")
    contribs = [(sig, WEIGHTS[sig], float(top[sig])) for sig in _SIGNAL_NAMES]
    contribs.sort(key=lambda x: x[1] * x[2], reverse=True)
    for sig, wt, val in contribs:
        contrib = wt * val
        bar_len = round(contrib / 0.15 * 10)  # scale bar
        print(
            f"  {_SIGNAL_NAMES[sig]:<28}  {wt:>7.2f}  {val:>6.3f}  "
            f"  {'▪' * bar_len}  {contrib:.4f}"
        )
    print(f"\n  Weighted viral score: {top['viral_score']:.4f}")

    # ── Predicted viral lyrics ───────────────────────────────────────────────
    print(f"\n  Predicted viral lyrics ({top['section']}):")
    print("  " + "─" * 50)
    for line in str(top["full_lyrics"]).split("\n")[:8]:
        if line.strip():
            print(f"    \"{line.strip()}\"")

    print("\n" + "=" * W + "\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLP algorithm to detect which lyric section goes viral on TikTok",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlp_viral_detector.py "Espresso" "Sabrina Carpenter"
  python nlp_viral_detector.py "APT." "ROSÉ" --save
  python nlp_viral_detector.py "Cruel Summer" "Taylor Swift"
        """,
    )
    parser.add_argument("title",  help="Song title (use quotes if multi-word)")
    parser.add_argument("artist", help="Artist name")
    parser.add_argument(
        "--save", action="store_true",
        help="Save signal breakdown to data/nlp_analysis.csv",
    )
    args = parser.parse_args()

    df = detect_viral_section(args.title, args.artist)
    if df is None:
        sys.exit(1)

    print_results(df, args.title, args.artist)

    if args.save:
        out = Path(__file__).parent / "data" / "nlp_analysis.csv"
        sig_cols = list(WEIGHTS.keys()) + ["viral_score"]
        save_df = df[["section", "type", "lines", "preview"] + sig_cols].copy()
        save_df.to_csv(out, index=False)
        print(f"  Saved signal breakdown → {out}\n")


if __name__ == "__main__":
    main()

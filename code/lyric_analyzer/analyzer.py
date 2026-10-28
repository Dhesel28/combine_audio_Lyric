"""
lyric_analyzer/analyzer.py
==========================
Main entry point for lyric analysis.

Usage (from lyrics string):
    from lyric_analyzer import LyricAnalyzer
    result = LyricAnalyzer().analyze(lyrics_text)

Usage (fetch from Genius automatically):
    analyzer = LyricAnalyzer(genius_token="YOUR_TOKEN")
    result = analyzer.analyze_song("Espresso", "Sabrina Carpenter")

Result dict keys
----------------
  sections     : list[dict] — sections ranked by viral_score (descending)
  best_section : dict       — top-ranked section with all 12 signal scores
  lines        : list[dict] — every line scored with 6 signals
  viral_line   : dict       — single highest-scoring line
  viral_clip   : list[dict] — CLIP_LINES lines centred on viral_line
  runner_up    : dict|None  — second-best line
  song_features: dict       — 11 song-level aggregate features
  summary      : str        — human-readable markdown reasoning
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import pandas as pd

from .config import (
    SECTION_WEIGHTS, LINE_WEIGHTS, HOOK_SECTION_KEYS, CLIP_LINES, STOPWORDS,
    ENERGY_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS,
)
from .parser import parse_sections, parse_lines, section_type, auto_label_sections
from .signals import (
    sig_hook_vocabulary, sig_cross_song_repetition, sig_internal_repetition,
    sig_sentiment_extremity, sig_end_rhyme_density, sig_line_conciseness,
    sig_syllable_consistency, sig_phonetic_openness, sig_pronoun_relatability,
    sig_internal_rhyme, sig_energy_markers, sig_position_score,
    line_sig_repetition, line_sig_energy, line_sig_rhyme,
    line_sig_brevity, line_sig_section_bonus, line_sig_emotional_punch,
)

_SECTION_RE = re.compile(r"\[([^\]]+)\]")


# ── Song-level aggregate features ─────────────────────────────────────────────

def _extract_song_features(raw: str) -> dict:
    """
    11 normalised song-level features in [0, 1].
    Used by tiktok_trend/ as input to the song virality model.
    """
    lines_raw = [l.strip() for l in raw.split("\n") if l.strip()]
    section_labels = _SECTION_RE.findall(raw)
    sections_norm = [re.sub(r'\s*\d+$', '', s).strip().lower() for s in section_labels]
    lyric_lines = [l for l in lines_raw if not _SECTION_RE.match(l)]
    plain = _SECTION_RE.sub("", raw).strip()

    def _tokenize(text: str) -> list[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in text.split() if w and w not in STOPWORDS and len(w) > 1]

    tokens = _tokenize(plain)
    total_lines = max(len(lyric_lines), 1)
    total_tokens = max(len(tokens), 1)
    unique_tok = len(set(tokens))

    lc = Counter(l.lower() for l in lyric_lines)

    # rhyme density (adjacent line pairs, last 3 chars)
    pairs = rhymes = 0
    for a, b in zip(lyric_lines, lyric_lines[1:]):
        wa = a.split()[-1].lower() if a.split() else ""
        wb = b.split()[-1].lower() if b.split() else ""
        if len(wa) >= 3 and len(wb) >= 3:
            pairs += 1
            if wa[-3:] == wb[-3:]:
                rhymes += 1
    rhyme_den = round(rhymes / max(pairs, 1), 4)

    token_set = set(tokens)
    n_sections = max(len(sections_norm), 1)
    chorus_cnt = sum(1 for s in sections_norm if any(k in s for k in ("chorus", "hook", "refrain")))
    avg_len = sum(len(l.split()) for l in lyric_lines) / total_lines if lyric_lines else 0
    repeated = sum(1 for c in lc.values() if c > 1)
    verse_cnt = sum(1 for s in sections_norm if "verse" in s)

    return {
        "vocab_richness":    round(min(unique_tok / total_tokens, 1.0), 4),
        "rhyme_density":     rhyme_den,
        "hook_strength":     round(min(lc.most_common(1)[0][1] / total_lines, 1.0) if lc else 0.0, 4),
        "energy_score":      round(min(len(token_set & ENERGY_WORDS) / 10, 1.0), 4),
        "positive_sent":     round(min(len(token_set & POSITIVE_WORDS) / max(unique_tok, 1) * 5, 1.0), 4),
        "negative_sent":     round(min(len(token_set & NEGATIVE_WORDS) / max(unique_tok, 1) * 5, 1.0), 4),
        "repetition_rate":   round(repeated / total_lines, 4),
        "chorus_density":    round(chorus_cnt / n_sections, 4),
        "avg_line_length":   round(min(avg_len / 20, 1.0), 4),
        "verse_density":     round(min(verse_cnt / n_sections, 1.0), 4),
        "compression_ratio": round(len(lc) / total_lines, 4),
    }


# ── Section scorer ─────────────────────────────────────────────────────────────

def _score_section(lines: list[str], idx: int, total: int, raw: str) -> dict:
    signals = {
        "hook_vocabulary":       sig_hook_vocabulary(lines),
        "cross_song_repetition": sig_cross_song_repetition(lines, raw),
        "internal_repetition":   sig_internal_repetition(lines),
        "sentiment_extremity":   sig_sentiment_extremity(lines),
        "end_rhyme_density":     sig_end_rhyme_density(lines),
        "line_conciseness":      sig_line_conciseness(lines),
        "syllable_consistency":  sig_syllable_consistency(lines),
        "phonetic_openness":     sig_phonetic_openness(lines),
        "pronoun_relatability":  sig_pronoun_relatability(lines),
        "internal_rhyme":        sig_internal_rhyme(lines),
        "energy_markers":        sig_energy_markers(lines),
        "position_score":        sig_position_score(idx, total),
    }
    viral_score = sum(SECTION_WEIGHTS[k] * v for k, v in signals.items())
    return {**signals, "viral_score": round(viral_score, 4)}


# ── Line scorer ────────────────────────────────────────────────────────────────

def _score_lines(raw: str) -> list[dict]:
    annotated = parse_lines(raw)
    lyric_lines = [a for a in annotated if not a["is_header"]]
    if not lyric_lines:
        return []

    line_counts = Counter(a["text"].strip().lower() for a in lyric_lines)
    max_count = max(line_counts.values(), default=1)
    texts = [a["text"] for a in lyric_lines]

    scored = []
    for i, ann in enumerate(lyric_lines):
        text = ann["text"]
        sec = ann["section"]
        prev_text = texts[i - 1] if i > 0 else ""
        next_text = texts[i + 1] if i < len(texts) - 1 else ""

        rep_score = line_sig_repetition(text, line_counts, max_count)
        rep_raw   = line_counts[text.strip().lower()]

        factors = {
            "repetition":      round(rep_score, 3),
            "energy":          round(line_sig_energy(text), 3),
            "rhyme":           round(line_sig_rhyme(text, prev_text, next_text), 3),
            "brevity":         round(line_sig_brevity(text), 3),
            "section_bonus":   round(line_sig_section_bonus(sec, HOOK_SECTION_KEYS), 3),
            "emotional_punch": round(line_sig_emotional_punch(text), 3),
        }
        score = sum(LINE_WEIGHTS[k] * factors[k] for k in LINE_WEIGHTS)

        if score >= 0.65:
            label = "Viral"
        elif score >= 0.45:
            label = "Strong Hook"
        elif score >= 0.28:
            label = "Solid"
        else:
            label = "-"

        why_parts = []
        if factors["repetition"] >= 0.5:
            why_parts.append(f"repeats {rep_raw}x")
        if factors["energy"] >= 0.33:
            why_parts.append("high energy")
        if factors["rhyme"] == 1.0:
            why_parts.append("rhymes")
        if factors["section_bonus"] == 1.0:
            why_parts.append("in chorus/hook")
        if factors["emotional_punch"] >= 0.33:
            why_parts.append("emotional punch")
        if factors["brevity"] >= 0.9:
            why_parts.append("perfect length")

        scored.append({
            "idx":     ann["idx"],
            "section": sec,
            "text":    text,
            "score":   round(score, 4),
            "factors": factors,
            "label":   label,
            "why":     ", ".join(why_parts) if why_parts else "low virality",
            "repeats": rep_raw,
        })

    return scored


# ── Viral clip builder ─────────────────────────────────────────────────────────

def _build_clip(scored: list[dict]) -> dict:
    if not scored:
        return {"viral_line": None, "runner_up": None, "viral_clip": [], "section": "N/A"}

    sorted_lines = sorted(scored, key=lambda x: x["score"], reverse=True)
    viral = sorted_lines[0]
    runner_up = sorted_lines[1] if len(sorted_lines) > 1 else None

    viral_pos = next((i for i, s in enumerate(scored) if s["idx"] == viral["idx"]), 0)
    half = CLIP_LINES // 2
    start = max(0, viral_pos - half)
    end = min(len(scored), start + CLIP_LINES)
    start = max(0, end - CLIP_LINES)
    clip = scored[start:end]

    return {
        "viral_line": viral,
        "runner_up":  runner_up,
        "viral_clip": clip,
        "section":    viral["section"].title(),
    }


def _build_summary(viral_line: dict, runner_up: Optional[dict], best_section: dict) -> str:
    lines = []
    sc = round(viral_line["score"] * 100)
    lines.append(
        f"**Viral moment lives in [{viral_line['section'].title()}]** "
        f"(line score: {sc}%)"
    )
    lines.append(f"\n> *\"{viral_line['text']}\"*\n")

    f = viral_line["factors"]
    if f["repetition"] >= 0.5:
        lines.append(f"- Repeats {viral_line['repeats']}x in the song")
    if f["energy"] >= 0.33:
        lines.append("- High-energy language")
    if f["rhyme"] == 1.0:
        lines.append("- End-rhymes with adjacent line")
    if f["section_bonus"] == 1.0:
        lines.append("- Lives in the Chorus/Hook")
    if f["emotional_punch"] >= 0.33:
        lines.append("- Strong emotional hit")
    if f["brevity"] >= 0.9:
        lines.append(f"- Perfect length ({len(viral_line['text'].split())} words)")

    sec_score = round(best_section.get("viral_score", 0) * 100)
    lines.append(
        f"\n**Best section:** {best_section.get('section', '?')} "
        f"(section score: {sec_score}%)"
    )

    if runner_up:
        lines.append(
            f"\n**Also worth clipping:** *\"{runner_up['text']}\"* "
            f"({round(runner_up['score']*100)}%) in [{runner_up['section'].title()}]"
        )

    lines.append("\n**TikTok clip:** Use the 5 lines shown in the clip window.")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  LyricAnalyzer — public API
# ══════════════════════════════════════════════════════════════════════════════

class LyricAnalyzer:
    """
    Standalone NLP analyzer for song lyrics.

    Parameters
    ----------
    genius_token : str, optional
        Genius API access token. Required only for analyze_song().
    """

    def __init__(self, genius_token: str = "") -> None:
        self._genius_token = genius_token

    # ── Core analysis ──────────────────────────────────────────────────────────

    def analyze(self, raw_lyrics: str) -> dict:
        """
        Analyze raw lyrics text.

        Returns a result dict with keys:
          sections, best_section, lines, viral_line, viral_clip,
          runner_up, song_features, summary
        """
        sections_map = parse_sections(raw_lyrics)
        if not sections_map:
            return self._empty_result()

        labels = list(sections_map.keys())
        total = len(labels)

        # Score sections
        section_rows = []
        for idx, label in enumerate(labels):
            lines = sections_map[label]
            stype = section_type(label)
            scores = _score_section(lines, idx, total, raw_lyrics)
            section_rows.append({
                "section":     label,
                "type":        stype,
                "lines_count": len(lines),
                "preview":     " / ".join(lines[:2]),
                "lyrics":      "\n".join(lines),
                **scores,
            })

        section_rows.sort(key=lambda r: r["viral_score"], reverse=True)
        best_section = section_rows[0]

        # Score lines
        scored_lines = _score_lines(raw_lyrics)
        clip_data = _build_clip(scored_lines)

        viral_line = clip_data["viral_line"]
        summary = (
            _build_summary(viral_line, clip_data["runner_up"], best_section)
            if viral_line else "No lyric lines found."
        )

        return {
            "sections":      section_rows,
            "best_section":  best_section,
            "lines":         scored_lines,
            "viral_line":    viral_line,
            "viral_clip":    clip_data["viral_clip"],
            "runner_up":     clip_data["runner_up"],
            "song_features": _extract_song_features(raw_lyrics),
            "summary":       summary,
        }

    def analyze_song(self, title: str, artist: str) -> dict:
        """
        Fetch lyrics from Genius, then analyze.

        Requires genius_token to be set on the LyricAnalyzer instance.
        """
        if not self._genius_token:
            raise ValueError(
                "Pass genius_token= to LyricAnalyzer() to use analyze_song()."
            )

        import lyricsgenius
        genius = lyricsgenius.Genius(
            self._genius_token,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"],
            verbose=False,
        )
        genius.remove_section_headers = False

        song = genius.search_song(title, artist)
        if not song:
            return self._empty_result()

        result = self.analyze(song.lyrics)
        result["song_title"] = song.title
        result["artist"]     = artist
        return result

    def sections_dataframe(self, raw_lyrics: str) -> pd.DataFrame:
        """Convenience: return section scores as a sorted DataFrame."""
        result = self.analyze(raw_lyrics)
        rows = result["sections"]
        signal_cols = list(SECTION_WEIGHTS.keys()) + ["viral_score"]
        df = pd.DataFrame(rows)[
            ["section", "type", "lines_count", "preview"] + signal_cols
        ].copy()
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    @staticmethod
    def _empty_result() -> dict:
        return {
            "sections":      [],
            "best_section":  {},
            "lines":         [],
            "viral_line":    None,
            "viral_clip":    [],
            "runner_up":     None,
            "song_features": {},
            "summary":       "No lyrics found or parsed.",
        }

"""
lyric_analyzer
==============
Standalone NLP module for song lyric analysis.

Outputs per analysis:
  - Section-level viral scores (12 NLP signals)
  - Line-level viral scores (6 signals)
  - Best viral snippet / TikTok clip (5-line window)
  - 11 song-level aggregate features (for downstream ML models)

Quick start
-----------
    from lyric_analyzer import LyricAnalyzer

    analyzer = LyricAnalyzer()
    result = analyzer.analyze(raw_lyrics_string)

    # or fetch from Genius automatically
    analyzer = LyricAnalyzer(genius_token="YOUR_TOKEN")
    result = analyzer.analyze_song("Espresso", "Sabrina Carpenter")

    print(result["summary"])
    print(result["viral_line"]["text"])
    print(result["song_features"])
"""

from .analyzer import LyricAnalyzer
from .parser import parse_sections, parse_lines, section_type, auto_label_sections
from .signals import (
    sig_hook_vocabulary, sig_cross_song_repetition, sig_internal_repetition,
    sig_sentiment_extremity, sig_end_rhyme_density, sig_line_conciseness,
    sig_syllable_consistency, sig_phonetic_openness, sig_pronoun_relatability,
    sig_internal_rhyme, sig_energy_markers, sig_position_score,
    line_sig_repetition, line_sig_energy, line_sig_rhyme,
    line_sig_brevity, line_sig_section_bonus, line_sig_emotional_punch,
)
from .config import SECTION_WEIGHTS, LINE_WEIGHTS, CLIP_LINES

__all__ = [
    "LyricAnalyzer",
    "parse_sections",
    "parse_lines",
    "section_type",
    "auto_label_sections",
    "SECTION_WEIGHTS",
    "LINE_WEIGHTS",
    "CLIP_LINES",
]

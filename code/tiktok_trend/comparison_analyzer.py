"""
Lyric Comparison: Billboard Hot 100 vs TikTok/YouTube Trending
==============================================================
Loads both datasets, computes lyric feature stats per group,
identifies key differences, and outputs a report + CSV to the external drive.
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, BILLBOARD_CSV, CM_TRAINING_CSV
TIKTOK_CSV     = CM_TRAINING_CSV
REPORT_CSV     = DATA_DIR / "lyric_comparison.csv"
REPORT_TXT     = DATA_DIR / "lyric_comparison_report.txt"

# ── Feature extraction ─────────────────────────────────────────────────────────

_SECTION_RE  = re.compile(r'\[([^\]]+)\]', re.I)
_HOOK_SECS   = {"chorus", "hook", "refrain", "bridge", "pre-chorus"}

ENERGY_WORDS = {
    "fire","lit","hype","wild","crazy","turnt","vibe","bounce","drop","bang",
    "banger","loud","scream","jump","move","go","fast","hard","rage","beast",
    "king","queen","boss","flex","drip","swag","pop","rock","hit","epic","run",
    "rise","shine","burn","blaze","fly","rush","push","break",
}
POSITIVE_WORDS = {
    "love","happy","joy","smile","beautiful","wonderful","amazing","great",
    "good","best","perfect","shine","bright","warm","sweet","dream","hope",
    "win","glory","bless","grace","light","gold","free","peace","forever",
}
NEGATIVE_WORDS = {
    "hate","pain","cry","tears","hurt","break","lost","alone","dark","cold",
    "dead","fall","fail","lie","cheat","betray","gone","miss","sad","empty",
    "numb","sorrow","grief","regret","fear","wrong","bad","worst","hell","die",
}


def _last3(line: str) -> str:
    words = line.strip().split()
    return words[-1].lower()[-3:] if words else ""


def extract_features(lyrics: str) -> dict:
    """Compute lyric features from raw lyrics text."""
    if not lyrics or not str(lyrics).strip():
        return {}

    raw_lines = str(lyrics).split("\n")
    current_sec = "intro"
    lyric_lines, section_labels = [], []

    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        m = _SECTION_RE.match(s)
        if m:
            current_sec = m.group(1).strip().lower()
        else:
            lyric_lines.append(s)
            section_labels.append(current_sec)

    if not lyric_lines:
        return {}

    n = len(lyric_lines)

    # Hook / chorus density
    hook_lines = sum(1 for sec in section_labels if any(k in sec for k in _HOOK_SECS))
    chorus_density = hook_lines / n

    # Repetition rate
    from collections import Counter
    counts = Counter(l.strip().lower() for l in lyric_lines)
    repeated = sum(1 for v in counts.values() if v > 1)
    repetition_rate = repeated / max(len(counts), 1)
    hook_strength   = (max(counts.values()) - 1) / max(n - 1, 1)

    # Energy density
    energy_score = np.mean([
        min(len(set(l.lower().split()) & ENERGY_WORDS) / 3.0, 1.0)
        for l in lyric_lines
    ])

    # Rhyme density
    endings = [_last3(l) for l in lyric_lines]
    rhyme_pairs = sum(
        1 for i in range(1, n) if len(endings[i]) >= 2 and endings[i] == endings[i-1]
    )
    rhyme_density = rhyme_pairs / max(n - 1, 1)

    # Brevity (4–9 words = ideal)
    def _brev(line):
        wc = len(line.split())
        if 4 <= wc <= 9: return 1.0
        if wc < 4:       return 0.5 + wc * 0.1
        return max(0.0, 1.0 - (wc - 9) * 0.08)

    brevity = np.mean([_brev(l) for l in lyric_lines])

    # Emotional punch
    sentiment_scores = []
    for l in lyric_lines:
        words = set(l.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        sentiment_scores.append(min(abs(pos - neg) / 3.0, 1.0))
    emotional_punch = float(np.mean(sentiment_scores))

    # Vocabulary richness
    all_words = " ".join(lyric_lines).lower().split()
    vocab_richness = len(set(all_words)) / max(len(all_words), 1)

    # Average line length (words)
    avg_line_len = np.mean([len(l.split()) for l in lyric_lines])

    # Total line count
    line_count = n

    return {
        "hook_strength":    round(float(hook_strength),    4),
        "energy_score":     round(float(energy_score),     4),
        "rhyme_density":    round(float(rhyme_density),    4),
        "repetition_rate":  round(float(repetition_rate),  4),
        "chorus_density":   round(float(chorus_density),   4),
        "brevity":          round(float(brevity),           4),
        "emotional_punch":  round(float(emotional_punch),  4),
        "vocab_richness":   round(float(vocab_richness),   4),
        "avg_line_len":     round(float(avg_line_len),     2),
        "line_count":       line_count,
    }


# ── Load + compute features ────────────────────────────────────────────────────

def _load_with_features(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")].copy()
    print(f"  [{label}] {len(df)} songs with lyrics")

    feat_rows = []
    for _, row in df.iterrows():
        feats = extract_features(row["lyrics"])
        feats["song"]   = row.get("song", "")
        feats["artist"] = row.get("artist", "")
        feats["source"] = label
        feat_rows.append(feats)

    return pd.DataFrame(feat_rows)


# ── Comparison report ──────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "hook_strength":   "Hook Strength (most-repeated line frequency)",
    "energy_score":    "Energy Score (high-energy word density)",
    "rhyme_density":   "Rhyme Density (consecutive end-rhyme rate)",
    "repetition_rate": "Repetition Rate (fraction of lines that repeat)",
    "chorus_density":  "Chorus/Hook Density (proportion of hook-section lines)",
    "brevity":         "Brevity (avg line shortness, 4-9 words = 1.0)",
    "emotional_punch": "Emotional Punch (strong sentiment intensity)",
    "vocab_richness":  "Vocabulary Richness (unique word ratio)",
    "avg_line_len":    "Avg Line Length (words per line)",
    "line_count":      "Song Length (total lyric lines)",
}


def _diff_label(tiktok_val: float, billboard_val: float, feat: str) -> str:
    diff = tiktok_val - billboard_val
    pct  = (diff / max(abs(billboard_val), 0.001)) * 100
    direction = "HIGHER" if diff > 0 else "LOWER"
    return f"TikTok is {direction} by {abs(pct):.1f}%"


def run_comparison() -> None:
    """Load both datasets, compute features, print + save report."""
    print("\n[Comparison] Loading datasets ...")

    if not BILLBOARD_CSV.exists():
        print(f"  [!] Billboard CSV not found: {BILLBOARD_CSV}")
        return
    if not TIKTOK_CSV.exists():
        print(f"  [!] TikTok CSV not found: {TIKTOK_CSV}")
        return

    bb_df = _load_with_features(BILLBOARD_CSV, "Billboard")
    tt_df = _load_with_features(TIKTOK_CSV,    "TikTok/YouTube")

    if bb_df.empty or tt_df.empty:
        print("  [!] One or both datasets are empty — run fetchers first.")
        return

    # Combined CSV for downstream use
    combined = pd.concat([bb_df, tt_df], ignore_index=True)
    combined.to_csv(REPORT_CSV, index=False)
    print(f"\n[Comparison] Combined features saved → {REPORT_CSV}")

    # Stats per group
    feat_cols = list(FEATURE_LABELS.keys())
    bb_stats  = bb_df[feat_cols].mean()
    tt_stats  = tt_df[feat_cols].mean()

    lines = []
    lines.append("=" * 72)
    lines.append("  BILLBOARD HOT 100 (2026)  vs  TIKTOK / YOUTUBE TRENDING")
    lines.append("=" * 72)
    lines.append(f"  Billboard songs analysed : {len(bb_df)}")
    lines.append(f"  TikTok/YT songs analysed : {len(tt_df)}")
    lines.append("")
    lines.append(f"{'Feature':<42} {'Billboard':>10} {'TikTok/YT':>10}  Difference")
    lines.append("-" * 72)

    insights = []
    for feat, label in FEATURE_LABELS.items():
        bb_val = bb_stats[feat]
        tt_val = tt_stats[feat]
        diff   = tt_val - bb_val
        pct    = (diff / max(abs(bb_val), 0.001)) * 100
        arrow  = "↑" if diff > 0 else "↓"
        lines.append(
            f"{label[:41]:<42} {bb_val:>10.4f} {tt_val:>10.4f}  {arrow} {abs(pct):.1f}%"
        )
        if abs(pct) >= 10:
            insights.append((feat, label, bb_val, tt_val, pct))

    lines.append("")
    lines.append("KEY DIFFERENCES (≥10% gap)")
    lines.append("-" * 72)

    insights.sort(key=lambda x: abs(x[4]), reverse=True)
    for feat, label, bb_val, tt_val, pct in insights:
        direction = "HIGHER" if pct > 0 else "LOWER"
        lines.append(f"\n► {label}")
        lines.append(
            f"  Billboard: {bb_val:.4f}  |  TikTok/YT: {tt_val:.4f}  "
            f"→  TikTok is {direction} by {abs(pct):.1f}%"
        )
        _explain(feat, pct, lines)

    lines.append("")
    lines.append("WHAT MAKES TIKTOK SONGS POP")
    lines.append("-" * 72)
    _write_summary(insights, lines)

    lines.append("")
    lines.append("TOP 10 TIKTOK SONGS BY VIRALITY SCORE")
    lines.append("-" * 72)
    if "virality_score" in tt_df.columns:
        top = tt_df.nlargest(10, "virality_score")[["song","artist","virality_score"]]
        lines.append(top.to_string(index=False))

    lines.append("")
    lines.append("TOP 10 BILLBOARD SONGS BY VIRALITY SCORE")
    lines.append("-" * 72)
    if "virality_score" in bb_df.columns:
        top = bb_df.nlargest(10, "virality_score")[["song","artist","virality_score"]]
        lines.append(top.to_string(index=False))

    report = "\n".join(lines)
    print(report)
    REPORT_TXT.write_text(report)
    print(f"\n[Comparison] Report saved → {REPORT_TXT}")


def _explain(feat: str, pct: float, lines: list[str]) -> None:
    higher = pct > 0
    explanations = {
        "hook_strength": (
            "TikTok songs repeat their best line far more often — "
            "the brain locks onto that one phrase, making it perfect for lip-syncs.",
            "Billboard songs are more lyrically varied; the story matters more than one sticky line."
        ),
        "energy_score": (
            "TikTok thrives on hype words (fire, lit, vibe, drop) — "
            "energy pulls users into duets and reaction videos.",
            "Billboard hits can be slower, more emotional ballads — energy isn't the primary hook."
        ),
        "rhyme_density": (
            "TikTok lyrics rhyme almost every couplet — the satisfying 'click' at line endings "
            "makes viewers want to repeat the clip.",
            "Billboard allows more free-verse and narrative structures that don't rhyme as tightly."
        ),
        "repetition_rate": (
            "Repetition is TikTok's secret weapon — repeated lines become earworms "
            "that stick in your head after a 15-second clip.",
            "Billboard songs tell more of a story with unique lines throughout."
        ),
        "chorus_density": (
            "TikTok songs are chorus-heavy — more of the song is the hook itself, "
            "not verses leading up to it.",
            "Billboard songs invest more in verse storytelling before the payoff chorus."
        ),
        "brevity": (
            "TikTok lines are shorter (4–9 words) — ideal for text overlays, "
            "captions and duets where long lines get cut off.",
            "Billboard lyrics use longer, more detailed lines to paint a picture."
        ),
        "emotional_punch": (
            "TikTok content thrives on extreme emotion (love, pain, hype) — "
            "strong feelings = reaction videos and saves.",
            "Billboard hits can be more subtle and nuanced emotionally."
        ),
        "vocab_richness": (
            "TikTok songs use a smaller, simpler vocabulary — "
            "easy to sing along to on first listen.",
            "Billboard songs have richer vocabulary, rewarding repeated listens."
        ),
        "avg_line_len": (
            "TikTok lines are shorter — snappier, punchier, easier to remember.",
            "Billboard lines are longer — more room for storytelling and detail."
        ),
        "line_count": (
            "TikTok songs tend to be longer by line count — more repetition pads the line count.",
            "Billboard songs are often more concise lyrically."
        ),
    }
    if feat in explanations:
        lines.append(f"  → {explanations[feat][0 if higher else 1]}")


def _write_summary(insights: list, lines: list[str]) -> None:
    tiktok_higher = [f for f, _, _, _, p in insights if p > 0]
    tiktok_lower  = [f for f, _, _, _, p in insights if p < 0]

    lines.append(
        "TikTok/YouTube trending songs vs Billboard Hot 100 break down into "
        "clear lyric strategies:\n"
    )

    if tiktok_higher:
        lines.append("TikTok songs score HIGHER on:")
        for feat in tiktok_higher:
            lines.append(f"  • {FEATURE_LABELS[feat]}")

    if tiktok_lower:
        lines.append("\nTikTok songs score LOWER on:")
        for feat in tiktok_lower:
            lines.append(f"  • {FEATURE_LABELS[feat]}")

    lines.append("""
The TikTok formula: SHORT lines, REPEATED hooks, HIGH energy words,
dense RHYMES, and STRONG emotions. A listener hears 15 seconds and
already knows the line — that's when they film themselves singing it.

Billboard hits allow more variety — longer lines, richer vocabulary,
complex verse structures — because listeners pay attention for the
full 3 minutes. TikTok only needs 15 seconds to work.
""")


if __name__ == "__main__":
    run_comparison()

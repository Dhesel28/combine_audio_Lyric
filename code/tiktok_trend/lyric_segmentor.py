"""
Lyric Segmentor — Section Identification + TikTok Viral Section Predictor
==========================================================================
Parses song lyrics into labeled sections (Chorus, Verse, Bridge, Intro, etc.)
and scores each section for TikTok viral potential.

Works two ways:
  1. Explicit markers  — reads [Chorus], [Verse], [Bridge] etc. from Genius lyrics
  2. Auto-detection    — finds repeated line blocks when markers are missing

TikTok score per section combines:
  - Average + peak viral probability of lines in the section
  - Section type weight  (Chorus 100% > Pre-Chorus 80% > Bridge 70% > Verse 40%)
  - How many times the section repeats in the song
  - Hook density (% lines with viral_prob >= 0.55)

Usage (module):
    from lyric_segmentor import LyricSegmentor
    seg    = LyricSegmentor()
    result = seg.analyze(lyrics, song="Title", artist="Artist")
    print(seg.format_report(result))

Usage (CLI — single song from lyrics CSV):
    python lyric_segmentor.py --song "poster boy" --artist "2hollis"

Usage (CLI — batch all songs):
    python lyric_segmentor.py --batch
    python lyric_segmentor.py --batch --output /path/to/results.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

HD_DIR     = Path("/Volumes/Dhesel Driv /Work")
LYRICS_CSV = HD_DIR / "tiktok_new_to_dhesel_lyrics.csv"
OUTPUT_CSV = HD_DIR / "tiktok_section_analysis.csv"

# ── Section type constants ─────────────────────────────────────────────────────

SECTION_WEIGHT = {
    "chorus":     1.00,
    "hook":       1.00,
    "refrain":    0.95,
    "drop":       0.90,
    "pre-chorus": 0.80,
    "prechorus":  0.80,
    "bridge":     0.70,
    "verse":      0.40,
    "intro":      0.30,
    "outro":      0.20,
    "interlude":  0.15,
    "spoken":     0.10,
    "breakdown":  0.25,
    "unknown":    0.30,
}

_SECTION_RE  = re.compile(r'^\[([^\]]+)\]$', re.IGNORECASE)
_PAREN_SEC   = re.compile(r'^\(([^)]{1,40})\)\s*$')
_COLON_SEC   = re.compile(r'^(.{1,40}?)\s*:\s*$')
_SEC_KEYWORDS = frozenset({
    "verse", "chorus", "hook", "bridge", "intro", "outro", "refrain",
    "pre-chorus", "prechorus", "interlude", "break", "spoken", "drop",
    "coda", "tag", "transition", "pre chorus", "breakdown",
})


def _strip_num(label: str) -> str:
    """'verse 2' → 'verse', 'chorus 1' → 'chorus'"""
    return re.sub(r'\s*\d+\s*$', '', label.lower().strip()).strip()


def _has_sec_kw(text: str) -> bool:
    t = text.lower().replace("-", " ").replace("_", " ")
    return any(kw.replace("-", " ") in t for kw in _SEC_KEYWORDS)


def _detect_header(s: str) -> Optional[str]:
    """Detect non-bracketed section headers like (Chorus), VERSE 1:, BRIDGE."""
    # (Chorus), (Verse 1)
    m = _PAREN_SEC.match(s)
    if m and _has_sec_kw(m.group(1)):
        return _strip_num(m.group(1))
    # Verse 1:, Chorus:
    m = _COLON_SEC.match(s)
    if m and _has_sec_kw(m.group(1)) and len(m.group(1).split()) <= 4:
        return _strip_num(m.group(1))
    # Bare: CHORUS, Verse 2, BRIDGE
    if 1 <= len(s.split()) <= 4 and _has_sec_kw(s):
        return _strip_num(s)
    return None


def _line_sig(line: str) -> str:
    """Normalize line for repetition matching."""
    return re.sub(r"[^\w\s]", "", line.lower()).strip()


# ── Core segmentor ─────────────────────────────────────────────────────────────

class LyricSegmentor:
    """
    Identifies song sections and ranks them by TikTok viral potential.
    """

    def __init__(self):
        self._detector = None  # lazy-load ViralLineDetector

    def _get_detector(self):
        if self._detector is None:
            from nlp_models import ViralLineDetector
            self._detector = ViralLineDetector()
        return self._detector

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, lyrics: str) -> tuple[list[dict], bool]:
        """
        Parse lyrics into line entries.
        Returns (entries, auto_detected).
        Each entry: {text, section, sec_num, auto_detected}
        """
        raw_entries: list[dict] = []
        current_sec = "unknown"
        sec_counts: dict[str, int] = defaultdict(int)
        prev_sec = None

        for raw in lyrics.split("\n"):
            s = raw.strip()
            if not s:
                continue
            # Bracketed marker [Chorus]
            m = _SECTION_RE.match(s)
            if m:
                base = _strip_num(m.group(1))
                if base != prev_sec:
                    sec_counts[base] += 1
                    prev_sec = base
                current_sec = base
                continue
            # Non-bracketed header
            alt = _detect_header(s)
            if alt is not None:
                if alt != prev_sec:
                    sec_counts[alt] += 1
                    prev_sec = alt
                current_sec = alt
                continue
            raw_entries.append({"text": s, "section": current_sec, "sec_num": sec_counts.get(current_sec, 1)})

        if not raw_entries:
            return [], False

        # Check if we got meaningful section labels
        found_secs = {e["section"] for e in raw_entries}
        has_markers = any(s in SECTION_WEIGHT and s not in ("unknown",) for s in found_secs)

        if has_markers:
            return raw_entries, False

        # ── Auto-detect sections via repetition analysis ───────────────────
        texts = [e["text"] for e in raw_entries]
        sigs  = [_line_sig(t) for t in texts]
        n     = len(sigs)

        # Find the most-repeated consecutive block (size 2–8) → chorus
        best_block:  list[str] = []
        best_count:  int       = 1
        best_starts: list[int] = []

        for size in range(2, min(9, n // 2 + 1)):
            block_pos: dict[tuple, list[int]] = defaultdict(list)
            for i in range(n - size + 1):
                key = tuple(sigs[i:i + size])
                block_pos[key].append(i)
            for key, positions in block_pos.items():
                if len(positions) >= 2 and len(positions) > best_count:
                    best_count  = len(positions)
                    best_block  = list(key)
                    best_starts = positions[:]

        labels = ["verse"] * n

        if best_block and best_count >= 2:
            size = len(best_block)
            # Mark chorus regions
            for start in best_starts:
                for j in range(start, min(start + size, n)):
                    labels[j] = "chorus"

            first_chorus = best_starts[0]
            last_chorus_end = best_starts[-1] + size

            # Intro: before first chorus
            if first_chorus > 0:
                for j in range(first_chorus):
                    labels[j] = "intro"

            # Bridge: unique block between second-to-last and last chorus
            if len(best_starts) >= 2:
                bridge_start = best_starts[-2] + size
                bridge_end   = best_starts[-1]
                if bridge_end - bridge_start >= 2:
                    for j in range(bridge_start, bridge_end):
                        if labels[j] == "verse":
                            labels[j] = "bridge"

            # Outro: after last chorus
            if last_chorus_end < n:
                for j in range(last_chorus_end, n):
                    labels[j] = "outro"

        # Assign section numbers (increment when section type changes)
        sec_num_map: dict[str, int] = defaultdict(int)
        prev_lbl = None
        for i, entry in enumerate(raw_entries):
            lbl = labels[i]
            if lbl != prev_lbl:
                sec_num_map[lbl] += 1
                prev_lbl = lbl
            entry["section"]      = lbl
            entry["sec_num"]      = sec_num_map[lbl]
            entry["auto_detected"] = True

        return raw_entries, True

    # ── Scoring ───────────────────────────────────────────────────────────────

    @staticmethod
    def _score_section(lines: list[dict], sec_type: str, repeat_count: int) -> float:
        """
        TikTok viral score for one section instance (0–1).

          35% avg viral_prob
          25% top viral_prob
          20% section type weight
          10% repetition bonus  (appears 3+ times = full bonus)
          10% hook density  (lines with viral_prob >= 0.55)
        """
        probs     = [l["viral_prob"] for l in lines]
        avg_prob  = float(np.mean(probs)) if probs else 0.0
        top_prob  = max(probs)            if probs else 0.0
        hook_dens = sum(1 for p in probs if p >= 0.55) / max(len(probs), 1)
        type_w    = SECTION_WEIGHT.get(sec_type, 0.30)
        rep_bonus = min(repeat_count / 3.0, 1.0)

        return round(
            0.35 * avg_prob
            + 0.25 * top_prob
            + 0.20 * type_w
            + 0.10 * rep_bonus
            + 0.10 * hook_dens,
            4,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, lyrics: str, song: str = "", artist: str = "") -> dict:
        """
        Segment lyrics and rank sections by TikTok potential.

        Returns:
          sections     — all section instances with scores
          ranking      — sections sorted by tiktok_score (best first)
          best_section — top-ranked section
          tiktok_clip  — recommended 6-line clip from the best section
          summary      — one-line text summary
        """
        entries, auto_detected = self._parse(lyrics)
        if not entries:
            return {"error": "Could not parse lyrics — too short or empty"}

        # Get line-level viral probabilities
        detector  = self._get_detector()
        pred      = detector.predict(lyrics)
        all_lines = pred.get("all_lines", [])

        # Map line text → viral_prob (using text as key since indices may differ)
        text_prob: dict[str, float] = {}
        for l in all_lines:
            text_prob[l["text"].strip()] = l["viral_prob"]

        for entry in entries:
            entry["viral_prob"] = text_prob.get(entry["text"].strip(), 0.0)

        # Group into section instances: consecutive run of same (section, sec_num)
        instances: list[dict] = []
        i = 0
        while i < len(entries):
            sec  = entries[i]["section"]
            num  = entries[i]["sec_num"]
            j    = i
            while j < len(entries) and entries[j]["section"] == sec and entries[j]["sec_num"] == num:
                j += 1
            instances.append({
                "type":          sec,
                "num":           num,
                "lines":         entries[i:j],
                "auto_detected": entries[i].get("auto_detected", False),
            })
            i = j

        # How many times does each section type appear?
        type_counts = Counter(s["type"] for s in instances)

        # Score each instance
        for sec in instances:
            probs = [l["viral_prob"] for l in sec["lines"]]
            sec["tiktok_score"]    = self._score_section(sec["lines"], sec["type"], type_counts[sec["type"]])
            sec["avg_viral_prob"]  = round(float(np.mean(probs)) if probs else 0.0, 4)
            sec["top_viral_prob"]  = round(max(probs) if probs else 0.0, 4)
            sec["line_count"]      = len(sec["lines"])
            sec["top_line"]        = max(sec["lines"], key=lambda l: l["viral_prob"])

        ranking = sorted(instances, key=lambda s: s["tiktok_score"], reverse=True)
        best    = ranking[0]

        # Build TikTok clip: 6 lines centred on the top viral line in best section
        b_lines  = best["lines"]
        top_idx  = max(range(len(b_lines)), key=lambda k: b_lines[k]["viral_prob"])
        c_start  = max(0, top_idx - 2)
        c_end    = min(len(b_lines), c_start + 6)
        c_start  = max(0, c_end - 6)
        clip     = b_lines[c_start:c_end]

        label = f"[{best['type'].title()}]" + (f" #{best['num']}" if best["num"] > 1 else "")
        summary = (
            f"{label} — score {best['tiktok_score']:.0%} | "
            f"top line: \"{best['top_line']['text']}\" ({best['top_line']['viral_prob']:.0%})"
        )

        return {
            "song":          song,
            "artist":        artist,
            "sections":      instances,
            "ranking":       ranking,
            "best_section":  best,
            "tiktok_clip":   clip,
            "summary":       summary,
            "auto_detected": auto_detected,
        }

    # ── Pretty print ──────────────────────────────────────────────────────────

    def format_report(self, result: dict) -> str:
        if "error" in result:
            return f"ERROR: {result['error']}"

        out = []
        song   = result.get("song", "")
        artist = result.get("artist", "")

        if song:
            title_line = f"  {song}" + (f"  —  {artist}" if artist else "")
            out += ["=" * 62, title_line, "=" * 62, ""]

        out += ["SECTIONS  (ranked by TikTok potential)", "-" * 62]

        for rank, sec in enumerate(result["ranking"], 1):
            label = f"[{sec['type'].title()}]"
            if sec["num"] > 1:
                label += f" #{sec['num']}"
            auto  = " *" if sec.get("auto_detected") else ""
            bar   = "█" * int(sec["tiktok_score"] * 20) + "░" * (20 - int(sec["tiktok_score"] * 20))
            out.append(f"  #{rank}  {label:<22}{auto}  [{bar}]  {sec['tiktok_score']:.0%}")
            out.append(
                f"       avg={sec['avg_viral_prob']:.0%}  "
                f"peak={sec['top_viral_prob']:.0%}  "
                f"lines={sec['line_count']}"
            )
            out.append(f"       \"{sec['top_line']['text']}\"")
            out.append("")

        best  = result["best_section"]
        label = f"[{best['type'].title()}]" + (f" #{best['num']}" if best["num"] > 1 else "")
        out += [
            "RECOMMENDED TIKTOK CLIP",
            "-" * 62,
            f"  From: {label}  (TikTok score: {best['tiktok_score']:.0%})",
            "",
        ]
        top_text = best["top_line"]["text"]
        for l in result["tiktok_clip"]:
            arrow = "  ► " if l["text"] == top_text else "    "
            out.append(f"{arrow}{l['text']}")
        out.append("")

        if result.get("auto_detected"):
            out.append("  * Sections auto-detected (lyrics had no [Chorus]/[Verse] markers)")

        return "\n".join(out)

    # ── Batch processing ──────────────────────────────────────────────────────

    def batch_analyze(
        self,
        lyrics_csv: Path = LYRICS_CSV,
        output_csv: Path = OUTPUT_CSV,
    ) -> pd.DataFrame:
        """
        Analyze all songs in a lyrics CSV and save section analysis results.

        Output columns per song:
          artist, title, best_section_type, tiktok_score,
          best_viral_line, best_viral_prob, tiktok_clip, all_sections_summary,
          auto_detected
        """
        df = pd.read_csv(lyrics_csv)
        has_lyrics = df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip().ne(""))
        df_lyr = df[has_lyrics].copy().reset_index(drop=True)
        total  = len(df_lyr)
        print(f"Analyzing {total} songs...\n")

        rows = []
        for i, row in df_lyr.iterrows():
            artist = str(row.get("artist", ""))
            title  = str(row.get("title",  ""))
            lyrics = str(row["lyrics"])

            try:
                result = self.analyze(lyrics, song=title, artist=artist)
            except Exception as e:
                print(f"  [{i+1}/{total}] ERROR {title}: {e}")
                continue

            if "error" in result:
                continue

            best  = result["best_section"]
            clip  = "\n".join(l["text"] for l in result["tiktok_clip"])

            # Compact section summary: e.g. "chorus(92%) verse(48%) bridge(61%)"
            sec_summary = "  |  ".join(
                f"{s['type']}{'#'+str(s['num']) if s['num']>1 else ''} "
                f"({s['tiktok_score']:.0%})"
                for s in result["ranking"]
            )

            rows.append({
                "artist":           artist,
                "title":            title,
                "best_section":     best["type"],
                "tiktok_score":     best["tiktok_score"],
                "best_viral_line":  best["top_line"]["text"],
                "best_viral_prob":  best["top_line"]["viral_prob"],
                "tiktok_clip":      clip,
                "sections_ranked":  sec_summary,
                "auto_detected":    result["auto_detected"],
            })

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{total} done")

        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDone. {len(results_df)}/{total} songs analyzed.")
        print(f"Saved → {output_csv}")
        return results_df


# ── CLI ────────────────────────────────────────────────────────────────────────

def _find_song_lyrics(title: str, artist: str) -> Optional[str]:
    """Look up lyrics in the saved CSV by song title + artist."""
    if not LYRICS_CSV.exists():
        return None
    df = pd.read_csv(LYRICS_CSV)
    mask = (
        df["title"].str.lower().str.strip().str.contains(title.lower(), na=False)
        & df["artist"].str.lower().str.strip().str.contains(artist.lower(), na=False)
    )
    hits = df[mask & df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip().ne(""))]
    if hits.empty:
        return None
    return str(hits.iloc[0]["lyrics"])


def main():
    p = argparse.ArgumentParser(description="Lyric section segmentor + TikTok viral predictor")
    p.add_argument("--song",   type=str, default="", help="Song title to look up")
    p.add_argument("--artist", type=str, default="", help="Artist name")
    p.add_argument("--batch",  action="store_true",  help="Analyze all songs in the lyrics CSV")
    p.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Output CSV for batch mode")
    args = p.parse_args()

    seg = LyricSegmentor()

    if args.batch:
        seg.batch_analyze(lyrics_csv=LYRICS_CSV, output_csv=args.output)
        return

    if args.song:
        lyrics = _find_song_lyrics(args.song, args.artist)
        if not lyrics:
            print(f"Song not found in {LYRICS_CSV.name}. Check title/artist spelling.")
            sys.exit(1)
        result = seg.analyze(lyrics, song=args.song, artist=args.artist)
        print(seg.format_report(result))
        return

    p.print_help()


if __name__ == "__main__":
    main()

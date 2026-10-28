"""
lyric_analyzer/parser.py
========================
Lyrics parsing utilities — sections, lines, auto-labelling.
"""

from __future__ import annotations

import re
from typing import Dict, List

SECTION_RE = re.compile(r"\[([^\]]+)\]")

# ── Genius header cleaner ──────────────────────────────────────────────────────

def clean_genius_header(text: str) -> str:
    """Strip Genius page metadata (contributor counts, embed footers, etc.)."""
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


# ── Section type classifier ────────────────────────────────────────────────────

def section_type(label: str) -> str:
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


# ── Auto section labeller ──────────────────────────────────────────────────────

def auto_label_sections(raw_lyrics: str, verse_size: int = 4) -> str:
    """
    Inject [Section] headers into raw lyrics that have none.

    Finds the largest block of lines that repeats 2+ times (the chorus),
    then labels remaining chunks as verses. Falls back to uniform verse
    chunks if no repeated block is found.
    """
    lines = [l.strip() for l in raw_lyrics.split("\n") if l.strip()]
    n = len(lines)
    if n == 0:
        return raw_lyrics

    # Find largest repeated block
    chorus_lines: List[str] = []
    chorus_len = 0
    for block_size in range(min(8, n // 2), 1, -1):
        seen: Dict[str, int] = {}
        for i in range(n - block_size + 1):
            key = "\n".join(l.lower() for l in lines[i: i + block_size])
            seen[key] = seen.get(key, 0) + 1
        best = max(seen.items(), key=lambda x: x[1])
        if best[1] >= 2:
            first_idx = next(
                i for i in range(n - block_size + 1)
                if "\n".join(l.lower() for l in lines[i: i + block_size]) == best[0]
            )
            chorus_lines = lines[first_idx: first_idx + block_size]
            chorus_len = block_size
            break

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


# ── Section parser ─────────────────────────────────────────────────────────────

def parse_sections(raw: str) -> Dict[str, List[str]]:
    """
    Split raw lyrics into {label: [lines]}.
    Duplicate labels (e.g. two [Chorus] blocks) are merged.
    Auto-labels if no section headers are found.
    """
    cleaned = clean_genius_header(raw)

    # Auto-label if no section headers present
    if not SECTION_RE.search(cleaned):
        cleaned = auto_label_sections(cleaned)

    sections: Dict[str, List[str]] = {}
    current, buf = "[Intro]", []
    for line in cleaned.split("\n"):
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

    return {k: v for k, v in sections.items() if len(v) >= 2}


# ── Line parser ────────────────────────────────────────────────────────────────

def parse_lines(raw: str) -> list[dict]:
    """
    Return a flat list of lyric lines with section context.

    Each entry:
      idx        : int — 0-based index across all lyric lines
      section    : str — section label (lowercase)
      text       : str — line text
      is_header  : bool — True if this entry is a section header
    """
    current_sec = "intro"
    annotated: list[dict] = []
    lyric_idx = 0

    for raw_line in raw.split("\n"):
        s = raw_line.strip()
        if not s:
            continue
        m = SECTION_RE.match(s)
        if m:
            current_sec = m.group(1).strip().lower()
            annotated.append({
                "idx": None,
                "section": current_sec,
                "text": s,
                "is_header": True,
            })
        else:
            annotated.append({
                "idx": lyric_idx,
                "section": current_sec,
                "text": s,
                "is_header": False,
            })
            lyric_idx += 1

    return annotated

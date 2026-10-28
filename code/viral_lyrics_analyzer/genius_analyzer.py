"""
Fetch song lyrics from Genius and identify which sections/lines are most trending
by analyzing Genius annotation counts per lyric fragment.

High annotation count = community is discussing that line = likely the "viral hook"
"""

from __future__ import annotations

import re
import time
import requests
import lyricsgenius
import pandas as pd
from typing import Optional
from config import GENIUS_TOKEN, DATA_DIR


# ── Genius API client ──────────────────────────────────────────────────────────

def _genius_client():
    if not GENIUS_TOKEN or GENIUS_TOKEN == "your_genius_token_here":
        raise ValueError("Set GENIUS_ACCESS_TOKEN in your .env file")
    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        verbose=False,
    )
    genius.remove_section_headers = False  # keep [Verse], [Chorus] labels
    return genius


# ── Lyric section parsing ──────────────────────────────────────────────────────

SECTION_PATTERN = re.compile(r'\[([^\]]+)\]')


def _clean_genius_header(text: str) -> str:
    """Remove Genius page header metadata that appears before actual lyrics."""
    # Genius embeds contributor count, translations, and album info in the lyrics text
    # Strip everything before the first actual lyric marker or the song title
    # Heuristic: the header ends when we hit the first [Section] label or a real lyric line
    lines = text.split("\n")
    cleaned = []
    in_header = True
    for line in lines:
        stripped = line.strip()
        # Header detection: lines with contributor counts, translation lists, "Lyrics"
        if in_header:
            if SECTION_PATTERN.match(stripped):
                in_header = False
                cleaned.append(stripped)
            elif stripped.endswith("Lyrics") or stripped.endswith("lyrics"):
                # Still in header
                continue
            elif any(kw in stripped for kw in
                     ["Contributors", "Translations", "Read More", "Embed", "You might also like"]):
                continue
            elif len(stripped) > 5 and not any(c.isdigit() for c in stripped[:5]):
                # Looks like a real lyric line
                in_header = False
                cleaned.append(stripped)
        else:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def parse_lyrics_into_sections(raw_lyrics: str) -> dict[str, list[str]]:
    """
    Split raw Genius lyrics into named sections.
    Returns {section_label: [list_of_lines]}
    e.g. {"[Chorus]": ["I will always love you", ...], "[Verse 1]": [...]}
    """
    # Remove Genius metadata header first
    raw_lyrics = _clean_genius_header(raw_lyrics)

    sections = {}
    current_label = "[Intro]"
    current_lines = []

    for line in raw_lyrics.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = SECTION_PATTERN.match(line)
        if match:
            if current_lines:
                sections.setdefault(current_label, []).extend(current_lines)
            current_label = line  # e.g. "[Chorus]"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.setdefault(current_label, []).extend(current_lines)

    # Drop empty or near-empty sections
    sections = {k: v for k, v in sections.items() if len(v) >= 2}

    return sections


def section_type(label: str) -> str:
    """Categorize a section label into a simplified type."""
    label_lower = label.lower()
    if "chorus" in label_lower or "hook" in label_lower or "refrain" in label_lower:
        return "Chorus/Hook"
    elif "verse" in label_lower:
        return "Verse"
    elif "bridge" in label_lower:
        return "Bridge"
    elif "pre-chorus" in label_lower or "pre chorus" in label_lower:
        return "Pre-Chorus"
    elif "outro" in label_lower:
        return "Outro"
    elif "intro" in label_lower:
        return "Intro"
    else:
        return "Other"


# ── Annotation fetching ────────────────────────────────────────────────────────

def get_song_annotations(song_id: int, token: str) -> list[dict]:
    """
    Fetch Genius referents (annotated lyric fragments) for a song.
    Returns list of {fragment, annotation_count, votes_total}
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.genius.com/referents?song_id={song_id}&per_page=50&page=1&text_format=plain"

    all_referents = []
    page = 1

    while True:
        try:
            resp = requests.get(
                f"https://api.genius.com/referents",
                headers=headers,
                params={
                    "song_id": song_id,
                    "per_page": 50,
                    "page": page,
                    "text_format": "plain",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            referents = data.get("response", {}).get("referents", [])
            if not referents:
                break

            for ref in referents:
                fragment = ref.get("fragment", "")
                annotations = ref.get("annotations", [])
                votes = sum(a.get("votes_total", 0) for a in annotations)
                all_referents.append({
                    "fragment": fragment,
                    "annotation_count": len(annotations),
                    "votes_total": votes,
                    "is_pinned": any(a.get("pinned_role") for a in annotations),
                })

            if len(referents) < 50:
                break
            page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [!] Annotation fetch error (page {page}): {e}")
            break

    return all_referents


# ── Section scoring ────────────────────────────────────────────────────────────

def score_section(lines: list[str]) -> dict:
    """
    Score a lyric section for viral/hook potential using text features.
    Returns dict of feature scores.
    """
    full_text = " ".join(lines)
    words = full_text.lower().split()

    # Repetition score: how many duplicate lines
    unique_lines = set(l.strip().lower() for l in lines if l.strip())
    repeat_score = 1 - (len(unique_lines) / max(len(lines), 1))

    # Short line score: shorter = more singable/memorable
    avg_words = sum(len(l.split()) for l in lines) / max(len(lines), 1)
    short_score = max(0, 1 - avg_words / 20)  # normalize: 0 words=1.0, 20+ words=0

    # Rhyme score: check last-word rhymes
    last_words = [l.split()[-1].lower().rstrip(",.!?") for l in lines if l.split()]
    rhyme_count = 0
    for i in range(len(last_words)):
        for j in range(i + 1, len(last_words)):
            if last_words[i][-2:] == last_words[j][-2:] and last_words[i] != last_words[j]:
                rhyme_count += 1
    rhyme_score = min(1.0, rhyme_count / max(len(lines), 1))

    # Hook keywords score
    hook_words = {"love", "baby", "feel", "yeah", "oh", "never", "always", "tonight",
                  "heart", "wanna", "gonna", "need", "want", "know", "fire", "crazy"}
    hook_hit = sum(1 for w in words if w in hook_words)
    hook_score = min(1.0, hook_hit / max(len(words), 1) * 10)

    # Composite virality score (equal weights)
    composite = (repeat_score + short_score + rhyme_score + hook_score) / 4

    return {
        "repeat_score": round(repeat_score, 3),
        "short_line_score": round(short_score, 3),
        "rhyme_score": round(rhyme_score, 3),
        "hook_keyword_score": round(hook_score, 3),
        "text_virality_score": round(composite, 3),
        "line_count": len(lines),
        "word_count": len(words),
    }


# ── Main analysis function ─────────────────────────────────────────────────────

def analyze_song(song_title: str, artist: str) -> Optional[pd.DataFrame]:
    """
    Full analysis for one song:
    1. Fetch lyrics from Genius
    2. Parse into sections
    3. Score each section with text features
    4. Fetch annotation counts from Genius API
    5. Map annotations to sections
    Returns a DataFrame, one row per section.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {song_title} - {artist}")
    print('='*60)

    genius = _genius_client()

    # 1. Fetch song
    print("  [1/4] Fetching lyrics from Genius...")
    song = genius.search_song(song_title, artist)
    if not song:
        print(f"  [!] Song not found on Genius: {song_title} by {artist}")
        return None

    print(f"  Found: {song.title} (ID: {song.id})")

    # 2. Parse sections
    print("  [2/4] Parsing lyric sections...")
    sections = parse_lyrics_into_sections(song.lyrics)
    print(f"  Found {len(sections)} sections: {list(sections.keys())}")

    # 3. Score each section
    print("  [3/4] Scoring sections...")
    rows = []
    for label, lines in sections.items():
        scores = score_section(lines)
        rows.append({
            "song": song.title,
            "artist": artist,
            "section_label": label,
            "section_type": section_type(label),
            "lyrics_preview": " / ".join(lines[:3]),
            "full_lyrics": "\n".join(lines),
            **scores,
        })

    df = pd.DataFrame(rows)

    # 4. Fetch Genius annotations and map to sections
    print("  [4/4] Fetching Genius annotation data...")
    referents = get_song_annotations(song.id, GENIUS_TOKEN)
    print(f"  Got {len(referents)} annotated fragments")

    # For each section, sum annotations of any fragment that appears in that section
    for idx, row in df.iterrows():
        section_text = row["full_lyrics"].lower()
        section_annotations = 0
        section_votes = 0
        for ref in referents:
            if ref["fragment"].lower().strip() in section_text:
                section_annotations += ref["annotation_count"]
                section_votes += ref["votes_total"]
        df.at[idx, "genius_annotation_count"] = section_annotations
        df.at[idx, "genius_votes_total"] = section_votes

    # Normalize annotation counts (0–1)
    max_ann = df["genius_annotation_count"].max()
    if max_ann > 0:
        df["annotation_score"] = (df["genius_annotation_count"] / max_ann).round(3)
    else:
        df["annotation_score"] = 0.0

    # Final combined viral score = text_virality (50%) + annotation_score (50%)
    df["combined_viral_score"] = (
        (df["text_virality_score"] * 0.5) + (df["annotation_score"] * 0.5)
    ).round(3)

    # Sort by combined score
    df = df.sort_values("combined_viral_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def analyze_multiple_songs(songs: list[tuple[str, str]], save: bool = True) -> pd.DataFrame:
    """
    Analyze a list of (title, artist) tuples.
    Returns combined DataFrame across all songs.
    """
    all_results = []

    for title, artist in songs:
        df = analyze_song(title, artist)
        if df is not None:
            all_results.append(df)
        time.sleep(1)  # be nice to the API

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    if save:
        out = DATA_DIR / "lyric_sections_analysis.csv"
        combined.to_csv(out, index=False)
        print(f"\nSaved combined analysis to {out}")

    return combined


if __name__ == "__main__":
    # Quick test with a known viral song
    test_songs = [
        ("Espresso", "Sabrina Carpenter"),
        ("APT.", "Rose"),
        ("Die With A Smile", "Lady Gaga"),
    ]

    results = analyze_multiple_songs(test_songs)
    if not results.empty:
        print("\n=== TOP VIRAL SECTIONS ===")
        cols = ["song", "section_label", "section_type", "combined_viral_score",
                "genius_annotation_count", "lyrics_preview"]
        print(results[cols].head(15).to_string(index=False))

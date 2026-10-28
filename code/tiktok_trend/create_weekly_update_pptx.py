"""
Weekly Update — Minimalistic PowerPoint
Sep 30, 2026
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path
from config import DATA_DIR

OUT = DATA_DIR / "Weekly_Update_Sep30.pptx"

# ── Palette ────────────────────────────────────────────────────────────────────
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BLACK  = RGBColor(0x11, 0x11, 0x11)
GRAY   = RGBColor(0x88, 0x88, 0x88)
LGRAY  = RGBColor(0xF2, 0xF2, 0xF2)
DGRAY  = RGBColor(0x44, 0x44, 0x44)
ACCENT = RGBColor(0x1A, 0x1A, 0x1A)   # near-black accent

# ── Helpers ────────────────────────────────────────────────────────────────────
def _bg(slide):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = WHITE

def _box(slide, left, top, w, h, color=LGRAY):
    s = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    return s

def _t(slide, text, left, top, w, h, size=12, bold=False,
       color=BLACK, align=PP_ALIGN.LEFT, italic=False):
    txb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf  = txb.text_frame; tf.word_wrap = True
    p   = tf.paragraphs[0]; p.alignment = align
    r   = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.italic = italic
    return txb

def _rule(slide, left, top, w, color=LGRAY, h=0.018):
    _box(slide, left, top, w, h, color)

def _footer(slide, note=""):
    _rule(slide, 0.5, 6.9, 9.0, LGRAY)
    _t(slide, note or "TikTok Viral Detector  ·  Sep 30, 2026",
       0.5, 7.0, 9.0, 0.35, size=8, color=GRAY)


# ── Slides ─────────────────────────────────────────────────────────────────────

def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    # Thin accent bar
    _box(s, 0.5, 2.6, 0.06, 1.4, BLACK)
    _t(s, "Week in Review", 0.75, 2.55, 8.5, 0.8, size=40, bold=True, color=BLACK)
    _t(s, "September 30, 2026", 0.75, 3.4, 8.5, 0.45, size=14, color=GRAY)
    _t(s, "TikTok Viral Detector  ·  Lyric Segmentation Project",
       0.75, 3.9, 8.5, 0.4, size=12, color=GRAY)
    _footer(s)


def slide_overview(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "This Week", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    items = [
        ("01", "Integrated friend's dataset",
         "634 TikTok songs · converted from .numbers → CSV · saved to hard drive"),
        ("02", "Fetched lyrics",
         "464 / 634 songs found on Genius API (73.2%) · auto-resumed on interruption"),
        ("03", "Built two new models",
         "K-means segmentation model  +  Lyric section segmentor with TikTok scoring"),
    ]
    for i, (num, head, sub) in enumerate(items):
        y = 1.3 + i * 1.65
        _box(s, 0.5, y, 9.0, 1.45, LGRAY)
        _t(s, num,  0.75, y + 0.3,  0.7, 0.6, size=22, bold=True, color=GRAY)
        _t(s, head, 1.5,  y + 0.22, 7.5, 0.5, size=16, bold=True, color=BLACK)
        _t(s, sub,  1.5,  y + 0.72, 7.5, 0.5, size=11, color=GRAY)

    _footer(s)


def slide_dataset(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Friend's Dataset", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    stats = [
        ("634",  "songs in dataset"),
        ("83",   "overlap with Summer Dataset"),
        ("4",    "columns  ·  artist, title, source, duration"),
    ]
    for i, (num, lbl) in enumerate(stats):
        x = 0.5 + i * 3.1
        _box(s, x, 1.25, 2.85, 1.5, LGRAY)
        _t(s, num, x + 0.25, 1.35, 2.4, 0.75, size=36, bold=True, color=BLACK)
        _t(s, lbl, x + 0.25, 2.05, 2.4, 0.45, size=10, color=GRAY)

    _rule(s, 0.5, 3.0, 9.0)

    _t(s, "Source", 0.5, 3.15, 2.0, 0.35, size=9, bold=True, color=GRAY)
    _t(s, '"the tiktok songs you have been looking for" — playlist collected by friend (Arya)',
       0.5, 3.5, 9.0, 0.4, size=12, color=BLACK)

    _t(s, "Format", 0.5, 4.1, 2.0, 0.35, size=9, bold=True, color=GRAY)
    _t(s, "Original file was Apple Numbers (.numbers)  →  converted to CSV using numbers-parser",
       0.5, 4.45, 9.0, 0.4, size=12, color=BLACK)

    _t(s, "Storage", 0.5, 5.05, 2.0, 0.35, size=9, bold=True, color=GRAY)
    _t(s, "/Volumes/Dhesel Driv /Work/tiktok_new_to_dhesel.csv",
       0.5, 5.4, 9.0, 0.4, size=11, color=DGRAY, italic=True)

    _footer(s)


def slide_lyrics(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Lyrics Collection", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    # Big stat
    _t(s, "464", 0.5, 1.2, 3.0, 1.1, size=60, bold=True, color=BLACK)
    _t(s, "/ 634 songs",   3.3, 1.5, 2.5, 0.5, size=18, color=GRAY)
    _t(s, "lyrics found  (73.2%)", 3.3, 2.0, 4.0, 0.4, size=11, color=GRAY)

    _rule(s, 0.5, 2.7, 9.0)

    rows = [
        ("Genius API",         "lyricsgenius library · 0.5 s delay between requests"),
        ("Title cleaning",     "Stripped 'sped up', 'slowed', 'nightcore' suffixes before search"),
        ("170 not found",      "Foreign language songs, very obscure, or remix-only releases"),
        ("Auto-resume",        "Saves progress every 10 songs — safe to interrupt and restart"),
        ("Storage",            "/Volumes/Dhesel Driv /Work/tiktok_new_to_dhesel_lyrics.csv"),
    ]
    for i, (label, detail) in enumerate(rows):
        y = 2.9 + i * 0.72
        if i % 2 == 0:
            _box(s, 0.5, y - 0.04, 9.0, 0.68, LGRAY)
        _t(s, label,  0.65, y + 0.08, 2.3,  0.45, size=10, bold=True, color=DGRAY)
        _t(s, detail, 3.0,  y + 0.08, 6.35, 0.45, size=10, color=GRAY)

    _footer(s)


def slide_segmentation(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Segmentation Model", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    _t(s, "K-Means clustering on 18 lyrical features per song",
       0.5, 1.15, 9.0, 0.4, size=12, color=GRAY)

    # Features list
    _t(s, "FEATURES", 0.5, 1.7, 4.0, 0.3, size=8, bold=True, color=GRAY)
    feats = [
        "Energy · Positive sentiment · Negative sentiment · Brevity",
        "Repetition rate · Hook line ratio · Rhyme density",
        "Vocab richness · Line count · Duration",
        "7 emotional themes  (love, heartbreak, confidence, party…)",
    ]
    for i, f in enumerate(feats):
        _t(s, f"— {f}", 0.5, 2.0 + i * 0.38, 4.3, 0.35, size=10, color=DGRAY)

    # Clusters
    _t(s, "CLUSTERS  (k = 2, silhouette = 0.110)", 5.0, 1.7, 4.5, 0.3, size=8, bold=True, color=GRAY)

    for i, (name, count, detail) in enumerate([
        ("High-Energy / Party",  "282 songs", "energy=0.048  hook_ratio=41.6%  repetition=55.8%"),
        ("Love & Romance",       "177 songs", "positive=0.068  hook_ratio=24.5%  repetition=41.0%"),
    ]):
        y = 2.0 + i * 1.1
        _box(s, 5.0, y - 0.05, 4.5, 0.95, LGRAY)
        _t(s, name,   5.15, y,        4.2, 0.38, size=13, bold=True, color=BLACK)
        _t(s, count,  5.15, y + 0.38, 2.0, 0.3,  size=10, color=GRAY)
        _t(s, detail, 5.15, y + 0.62, 4.2, 0.28, size=8,  color=GRAY, italic=True)

    _rule(s, 0.5, 4.6, 9.0)
    _t(s, "Note", 0.5, 4.72, 0.8, 0.3, size=8, bold=True, color=GRAY)
    _t(s, "Low silhouette score (0.11) reflects that TikTok songs share similar lyrical structures. "
          "Audio features would sharpen cluster separation.",
       1.2, 4.72, 7.8, 0.5, size=10, color=GRAY, italic=True)

    _footer(s)


def slide_segmentor(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Lyric Segmentor", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    _t(s, "Identifies song sections and ranks each for TikTok viral potential",
       0.5, 1.15, 9.0, 0.4, size=12, color=GRAY)

    # How it works
    steps = [
        ("1  Parse sections",
         "Reads [Chorus] [Verse] [Bridge] markers from Genius lyrics (313/464 songs)"),
        ("2  Auto-detect",
         "Finds repeated line blocks when markers are missing (151/464 songs)"),
        ("3  Score each section",
         "Combines viral probability + section type weight + repetition count + hook density"),
        ("4  Recommend clip",
         "Returns best 6-line TikTok clip centred on the highest viral line"),
    ]
    for i, (step, detail) in enumerate(steps):
        y = 1.7 + i * 1.1
        _box(s, 0.5, y, 9.0, 0.95, LGRAY)
        _t(s, step,   0.7, y + 0.12, 2.8, 0.4, size=12, bold=True, color=BLACK)
        _t(s, detail, 3.6, y + 0.18, 5.7, 0.55, size=10, color=GRAY)

    _rule(s, 0.5, 6.2, 9.0)
    _t(s, "Section type weights:  Chorus 100%  ·  Pre-Chorus 80%  ·  Bridge 70%  ·  Verse 40%  ·  Intro 30%",
       0.5, 6.3, 9.0, 0.35, size=9, color=GRAY)

    _footer(s)


def slide_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Results  —  464 Songs Analyzed", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    # Summary stats row
    stats = [
        ("82%",  "of songs: Chorus\nis the TikTok section"),
        ("96%",  "top TikTok score\n(Rockabye — Clean Bandit)"),
        ("151",  "songs auto-detected\n(no section markers)"),
    ]
    for i, (num, lbl) in enumerate(stats):
        x = 0.5 + i * 3.05
        _box(s, x, 1.2, 2.8, 1.4, LGRAY)
        _t(s, num, x + 0.2, 1.28, 2.4, 0.7,  size=34, bold=True, color=BLACK)
        _t(s, lbl, x + 0.2, 1.95, 2.4, 0.5,  size=9,  color=GRAY)

    _rule(s, 0.5, 2.85, 9.0)

    # Top songs table
    _t(s, "TOP SONGS BY TIKTOK SECTION SCORE", 0.5, 2.98, 9.0, 0.3, size=8, bold=True, color=GRAY)

    headers = ["Song", "Artist", "Section", "Score"]
    col_x   = [0.5, 4.0, 6.7, 8.5]
    col_w   = [3.3, 2.5, 1.6, 1.2]

    for j, (h, x, w) in enumerate(zip(headers, col_x, col_w)):
        al = PP_ALIGN.RIGHT if j == 3 else PP_ALIGN.LEFT
        _t(s, h, x, 3.3, w, 0.3, size=8, bold=True, color=GRAY, align=al)
    _rule(s, 0.5, 3.6, 9.2, BLACK, 0.012)

    rows = [
        ("Rockabye (feat. Sean Paul & Anne-Marie)", "Clean Bandit",   "Chorus",  "99.7%"),
        ("Wild Thoughts (feat. Rihanna & Bryson)", "DJ Khaled",       "Chorus",  "99.7%"),
        ("Lights - Sped Up Version",               "Ellie Goulding",  "Chorus",  "99.6%"),
        ("Art Deco",                               "Lana Del Rey",    "Chorus",  "99.6%"),
        ("Drowning (feat. Kodak Black)",           "A Boogie",        "Chorus",  "99.5%"),
    ]
    for i, (song, artist, sec, score) in enumerate(rows):
        y = 3.68 + i * 0.58
        if i % 2 == 0:
            _box(s, 0.48, y - 0.03, 9.1, 0.55, LGRAY)
        _t(s, song,   col_x[0], y + 0.05, col_w[0], 0.42, size=10)
        _t(s, artist, col_x[1], y + 0.05, col_w[1], 0.42, size=10, color=GRAY)
        _t(s, sec,    col_x[2], y + 0.05, col_w[2], 0.42, size=10, color=GRAY)
        _t(s, score,  col_x[3], y + 0.05, col_w[3], 0.42, size=10, bold=True,
           align=PP_ALIGN.RIGHT)

    _footer(s, "Results saved → /Volumes/Dhesel Driv /Work/tiktok_section_analysis.csv")


def slide_next(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _box(s, 0.5, 2.0, 0.06, 1.8, BLACK)
    _t(s, "Next Steps", 0.75, 1.95, 8.5, 0.7, size=36, bold=True, color=BLACK)
    _rule(s, 0.75, 2.75, 8.5)

    steps = [
        ("Combine audio + lyrics",
         "Match songs to Spotify audio features (danceability, energy, valence, tempo) "
         "for sharper clustering and richer viral prediction"),
        ("Improve section detection",
         "Train a sequence model or add audio signal to detect section boundaries "
         "more accurately for songs without Genius markers"),
        ("Integrate into app",
         "Add the lyric segmentor to the Streamlit app so any song can be analyzed "
         "with a section breakdown and TikTok clip recommendation"),
    ]
    for i, (head, body) in enumerate(steps):
        y = 3.0 + i * 1.3
        _t(s, f"{i+1:02d}  {head}", 0.75, y, 8.5, 0.45, size=14, bold=True, color=BLACK)
        _t(s, body, 0.75, y + 0.45, 8.5, 0.6, size=11, color=GRAY)

    _footer(s)


# ── Build ───────────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_overview(prs)
    slide_dataset(prs)
    slide_lyrics(prs)
    slide_segmentation(prs)
    slide_segmentor(prs)
    slide_results(prs)
    slide_next(prs)

    prs.save(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    build()

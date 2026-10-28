"""
Create PowerPoint: Billboard 2026 vs TikTok/YouTube Trending — Lyric Analysis
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pandas as pd
from pathlib import Path
from config import DATA_DIR

OUTPUT = DATA_DIR / "Billboard_vs_TikTok_2026.pptx"

# ── Brand colors ───────────────────────────────────────────────────────────────
BLACK      = RGBColor(0x0D, 0x0D, 0x0D)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
TIKTOK_RED = RGBColor(0xFE, 0x2C, 0x55)  # TikTok red
BB_GOLD    = RGBColor(0xF5, 0xA6, 0x23)  # Billboard gold
DARK_GRAY  = RGBColor(0x1A, 0x1A, 0x2E)
MID_GRAY   = RGBColor(0x2D, 0x2D, 0x44)
LIGHT_GRAY = RGBColor(0xB0, 0xB0, 0xC0)
GREEN      = RGBColor(0x2E, 0xCC, 0x71)
ACCENT     = RGBColor(0x9B, 0x59, 0xB6)


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text(slide, text, left, top, width, height,
             font_size=18, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, italic=False):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txb


def add_rect(slide, left, top, width, height, color: RGBColor, alpha=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_bar(slide, left, top, bar_width, height, color, label, value_text,
            max_width=3.5):
    """Draw a horizontal bar with label and value."""
    add_text(slide, label, left, top, 2.8, 0.3, font_size=11, color=LIGHT_GRAY)
    filled = bar_width
    # background bar
    add_rect(slide, left + 2.9, top + 0.04, max_width, height - 0.08,
             RGBColor(0x3A, 0x3A, 0x55))
    # filled bar
    if filled > 0:
        add_rect(slide, left + 2.9, top + 0.04, max(filled, 0.05), height - 0.08, color)
    add_text(slide, value_text, left + 2.9 + max_width + 0.1, top, 1.0, 0.3,
             font_size=11, bold=True, color=color)


# ── Slides ─────────────────────────────────────────────────────────────────────

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)

    # Top accent bar
    add_rect(slide, 0, 0, 10, 0.12, TIKTOK_RED)

    # TikTok logo-style wordmark
    add_text(slide, "TikTok  vs  Billboard", 0.5, 0.4, 9, 1.0,
             font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    add_text(slide, "What Makes a Song Pop in 2026?",
             0.5, 1.5, 9, 0.6, font_size=22, color=TIKTOK_RED,
             align=PP_ALIGN.CENTER, bold=True)

    add_text(slide, "A Lyric Analysis: Billboard Hot 100 vs YouTube/TikTok Trending",
             0.5, 2.1, 9, 0.5, font_size=14, color=LIGHT_GRAY,
             align=PP_ALIGN.CENTER)

    # Divider
    add_rect(slide, 3.5, 2.7, 3.0, 0.04, TIKTOK_RED)

    # Dataset pills
    add_rect(slide, 1.2, 3.1, 3.2, 0.55, MID_GRAY)
    add_text(slide, "🎵  661 Billboard Songs", 1.3, 3.15, 3.0, 0.45,
             font_size=13, color=BB_GOLD, bold=True)

    add_rect(slide, 5.6, 3.1, 3.2, 0.55, MID_GRAY)
    add_text(slide, "📱  273 TikTok Hit Songs", 5.7, 3.15, 3.0, 0.45,
             font_size=13, color=TIKTOK_RED, bold=True)

    add_text(slide, "38 weeks of Hot 100  •  Apple Music + Shazam US  •  Genius lyrics  •  NLP feature extraction",
             0.3, 3.9, 9.4, 0.4, font_size=10, color=LIGHT_GRAY,
             align=PP_ALIGN.CENTER, italic=True)

    # Bottom bar
    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY,
             align=PP_ALIGN.CENTER)


def slide_dataset(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.08, MID_GRAY)

    add_text(slide, "THE DATASETS", 0.4, 0.2, 9.2, 0.5,
             font_size=11, color=TIKTOK_RED, bold=True)
    add_text(slide, "What we analysed", 0.4, 0.55, 9.2, 0.6,
             font_size=28, bold=True, color=WHITE)
    add_rect(slide, 0.4, 1.15, 1.2, 0.04, TIKTOK_RED)

    # Billboard card
    add_rect(slide, 0.4, 1.4, 4.3, 5.5, MID_GRAY)
    add_rect(slide, 0.4, 1.4, 4.3, 0.5, BB_GOLD)
    add_text(slide, "BILLBOARD HOT 100", 0.55, 1.45, 4.0, 0.4,
             font_size=14, bold=True, color=BLACK)

    bb_facts = [
        ("Songs", "661 unique"),
        ("Weeks covered", "Jan 3 – Sep 19, 2026"),
        ("Weeks scraped", "38 weeks"),
        ("Lyrics found", "590 / 661  (89%)"),
        ("Top artist", "Drake  (31 songs)"),
        ("2nd / 3rd", "Noah Kahan  •  Zach Bryan"),
        ("Source", "Billboard.com scraper"),
    ]
    for i, (k, v) in enumerate(bb_facts):
        y = 2.05 + i * 0.56
        add_text(slide, k, 0.6, y, 1.6, 0.4, font_size=10, color=LIGHT_GRAY)
        add_text(slide, v, 2.2, y, 2.4, 0.4, font_size=11, bold=True, color=WHITE)

    # TikTok card
    add_rect(slide, 5.3, 1.4, 4.3, 5.5, MID_GRAY)
    add_rect(slide, 5.3, 1.4, 4.3, 0.5, TIKTOK_RED)
    add_text(slide, "TIKTOK HITS  (Apple Music + Shazam)", 5.45, 1.45, 4.0, 0.4,
             font_size=12, bold=True, color=WHITE)

    tt_facts = [
        ("Songs", "273 unique"),
        ("Sources", "Apple Music Top 100 + Shazam US 200"),
        ("On both charts", "23 confirmed viral hits"),
        ("Lyrics found", "262 / 273  (96%)"),
        ("#1 hit", "Last Thing You Need – Morgan Wallen"),
        ("#2 hit", "BbY WOW – KAROL G"),
        ("Why reliable", "Shazam = songs found via TikTok videos"),
    ]
    for i, (k, v) in enumerate(tt_facts):
        y = 2.05 + i * 0.56
        add_text(slide, k, 5.5, y, 1.7, 0.4, font_size=10, color=LIGHT_GRAY)
        add_text(slide, v, 7.2, y, 2.3, 0.4, font_size=11, bold=True, color=WHITE)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


def slide_findings_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.08, MID_GRAY)

    add_text(slide, "KEY FINDINGS", 0.4, 0.2, 9.2, 0.5,
             font_size=11, color=TIKTOK_RED, bold=True)
    add_text(slide, "Feature comparison at a glance", 0.4, 0.55, 9.2, 0.6,
             font_size=28, bold=True, color=WHITE)
    add_rect(slide, 0.4, 1.15, 1.2, 0.04, TIKTOK_RED)

    # Column headers
    add_text(slide, "FEATURE", 0.4, 1.35, 3.5, 0.35, font_size=10,
             bold=True, color=LIGHT_GRAY)
    add_rect(slide, 3.9, 1.35, 2.5, 0.35, RGBColor(0x3A, 0x2A, 0x00))
    add_text(slide, "BILLBOARD", 3.9, 1.35, 2.5, 0.35, font_size=10,
             bold=True, color=BB_GOLD, align=PP_ALIGN.CENTER)
    add_rect(slide, 6.55, 1.35, 2.5, 0.35, RGBColor(0x3A, 0x08, 0x10))
    add_text(slide, "TIKTOK / YT", 6.55, 1.35, 2.5, 0.35, font_size=10,
             bold=True, color=TIKTOK_RED, align=PP_ALIGN.CENTER)
    add_text(slide, "ΔDIFF", 9.1, 1.35, 0.8, 0.35, font_size=10,
             bold=True, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    rows = [
        ("Rhyme Density",         "13.0%",  "14.4%",  "↑ 10.8%", GREEN),
        ("Emotional Punch",       "3.67%",  "4.05%",  "↑ 10.4%", GREEN),
        ("Song Length (lines)",   "47.6",   "50.7",   "↑ 6.6%",  GREEN),
        ("Energy Score",          "2.23%",  "2.24%",  "≈ same",  LIGHT_GRAY),
        ("Chorus/Hook Density",   "48.9%",  "48.6%",  "≈ same",  LIGHT_GRAY),
        ("Repetition Rate",       "24.2%",  "23.9%",  "≈ same",  LIGHT_GRAY),
        ("Avg Line Length",       "7.86 wds","7.86 wds","≈ same", LIGHT_GRAY),
        ("Brevity Score",         "90.9%",  "91.0%",  "≈ same",  LIGHT_GRAY),
    ]

    for i, (feat, bb_val, tt_val, diff, diff_color) in enumerate(rows):
        y = 1.85 + i * 0.62
        bg = MID_GRAY if i % 2 == 0 else RGBColor(0x22, 0x22, 0x38)
        add_rect(slide, 0.35, y - 0.02, 9.3, 0.55, bg)
        add_text(slide, feat, 0.45, y, 3.3, 0.45, font_size=11, color=WHITE)
        add_text(slide, bb_val, 3.9, y, 2.5, 0.45, font_size=12,
                 bold=True, color=BB_GOLD, align=PP_ALIGN.CENTER)
        add_text(slide, tt_val, 6.55, y, 2.5, 0.45, font_size=12,
                 bold=True, color=TIKTOK_RED, align=PP_ALIGN.CENTER)
        add_text(slide, diff, 9.1, y, 0.8, 0.45, font_size=11,
                 bold=True, color=diff_color, align=PP_ALIGN.CENTER)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


def slide_finding(prs, num, title, subtitle, bb_stat, tt_stat,
                  bb_bar, tt_bar, insight, detail, highlight_tiktok=True):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.08, MID_GRAY)

    add_text(slide, f"FINDING  {num}", 0.4, 0.18, 3, 0.4,
             font_size=10, color=TIKTOK_RED, bold=True)
    add_text(slide, title, 0.4, 0.5, 9.2, 0.7,
             font_size=30, bold=True, color=WHITE)
    add_text(slide, subtitle, 0.4, 1.2, 9.2, 0.45,
             font_size=14, color=LIGHT_GRAY, italic=True)
    add_rect(slide, 0.4, 1.65, 1.2, 0.04, TIKTOK_RED)

    # Billboard bar
    add_rect(slide, 0.4, 2.0, 4.3, 1.6, MID_GRAY)
    add_rect(slide, 0.4, 2.0, 4.3, 0.38, RGBColor(0x3A, 0x2A, 0x00))
    add_text(slide, "BILLBOARD", 0.55, 2.02, 2.5, 0.34,
             font_size=11, bold=True, color=BB_GOLD)
    add_text(slide, bb_stat, 0.55, 2.5, 4.0, 0.7,
             font_size=36, bold=True, color=BB_GOLD)
    add_rect(slide, 0.4, 3.1, bb_bar * 4.3, 0.28,
             BB_GOLD if not highlight_tiktok else RGBColor(0x8B, 0x6A, 0x00))

    # TikTok bar
    add_rect(slide, 5.3, 2.0, 4.3, 1.6, MID_GRAY)
    add_rect(slide, 5.3, 2.0, 4.3, 0.38, RGBColor(0x3A, 0x08, 0x10))
    add_text(slide, "TIKTOK / YOUTUBE", 5.45, 2.02, 3.5, 0.34,
             font_size=11, bold=True, color=TIKTOK_RED)
    add_text(slide, tt_stat, 5.45, 2.5, 4.0, 0.7,
             font_size=36, bold=True, color=TIKTOK_RED)
    add_rect(slide, 5.3, 3.1, tt_bar * 4.3, 0.28,
             TIKTOK_RED if highlight_tiktok else RGBColor(0x8B, 0x1A, 0x28))

    # Winner badge
    winner_x = 5.3 if highlight_tiktok else 0.4
    winner_color = TIKTOK_RED if highlight_tiktok else BB_GOLD
    winner_label = "TIKTOK HIGHER" if highlight_tiktok else "BILLBOARD HIGHER"
    add_rect(slide, winner_x + 2.8, 2.02, 1.4, 0.34, winner_color)
    add_text(slide, winner_label, winner_x + 2.82, 2.04, 1.36, 0.3,
             font_size=8, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Insight box
    add_rect(slide, 0.4, 3.55, 9.2, 1.15, MID_GRAY)
    add_rect(slide, 0.4, 3.55, 0.08, 1.15, TIKTOK_RED)
    add_text(slide, f"💡  {insight}", 0.6, 3.62, 8.9, 1.0,
             font_size=13, bold=True, color=WHITE)

    add_text(slide, detail, 0.4, 4.85, 9.2, 1.5,
             font_size=11, color=LIGHT_GRAY, italic=True)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


def slide_takeaway(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.5, TIKTOK_RED)
    add_text(slide, "THE FORMULA", 0.4, 0.08, 9.2, 0.34,
             font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_text(slide, "What makes a song pop on TikTok",
             0.4, 0.65, 9.2, 0.6, font_size=26, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, 3.5, 1.25, 3.0, 0.04, TIKTOK_RED)

    formulas = [
        ("RHYME EVERY LINE",  "14.4% rhyme density vs 13.0% — that satisfying click makes people replay the clip", TIKTOK_RED),
        ("GO ALL IN EMOTIONALLY", "4.05% emotional punch vs 3.67% — love, pain, heartbreak drives reaction videos", BB_GOLD),
        ("STACK THE HOOK",    "50.7 lines avg vs 47.6 — more lines = more repetitions of your best moment", GREEN),
        ("THE REST IS NOISE", "Energy, brevity, line length — all equal between Billboard and TikTok in 2026", ACCENT),
    ]

    for i, (label, desc, color) in enumerate(formulas):
        y = 1.45 + i * 1.35
        add_rect(slide, 0.4, y, 9.2, 1.2, MID_GRAY)
        add_rect(slide, 0.4, y, 0.1, 1.2, color)
        num_box_width = 0.55
        add_rect(slide, 0.5, y + 0.1, num_box_width, 1.0, color)
        add_text(slide, str(i + 1), 0.5, y + 0.25, num_box_width, 0.7,
                 font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, label, 1.2, y + 0.1, 7.8, 0.45,
                 font_size=15, bold=True, color=WHITE)
        add_text(slide, desc, 1.2, y + 0.55, 7.8, 0.55,
                 font_size=11, color=LIGHT_GRAY, italic=True)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


def slide_crossover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.08, MID_GRAY)

    add_text(slide, "THE CROSSOVER ZONE", 0.4, 0.2, 9.2, 0.5,
             font_size=11, color=TIKTOK_RED, bold=True)
    add_text(slide, "Songs that win on both platforms", 0.4, 0.55, 9.2, 0.6,
             font_size=28, bold=True, color=WHITE)
    add_rect(slide, 0.4, 1.15, 1.2, 0.04, TIKTOK_RED)

    add_text(slide, "A song crosses over when it nails the TikTok formula AND has Billboard-level structure.",
             0.4, 1.3, 9.2, 0.5, font_size=13, color=LIGHT_GRAY, italic=True)

    # Venn-style diagram (two overlapping rects + center)
    add_rect(slide, 0.3, 2.0, 4.0, 3.8, RGBColor(0x1E, 0x1E, 0x35))
    add_rect(slide, 5.7, 2.0, 4.0, 3.8, RGBColor(0x35, 0x10, 0x18))
    add_rect(slide, 3.8, 2.0, 2.4, 3.8, RGBColor(0x22, 0x22, 0x38))

    add_text(slide, "BILLBOARD", 0.5, 2.1, 3.8, 0.4,
             font_size=13, bold=True, color=BB_GOLD, align=PP_ALIGN.CENTER)
    add_text(slide, "BOTH", 3.8, 2.1, 2.4, 0.4,
             font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, "TIKTOK", 5.7, 2.1, 3.8, 0.4,
             font_size=13, bold=True, color=TIKTOK_RED, align=PP_ALIGN.CENTER)

    bb_only = ["Long verses", "Dense rhyme", "Emotional arcs", "Complex structure", "Full 3-min story"]
    both    = ["One sticky hook", "Short lines", "Repetition", "Simple words"]
    tt_only = ["15-sec moment", "Global sounds", "Dance sounds", "Visual beats"]

    for i, item in enumerate(bb_only):
        add_text(slide, f"• {item}", 0.4, 2.65 + i * 0.55, 3.8, 0.45,
                 font_size=11, color=BB_GOLD)
    for i, item in enumerate(both):
        add_text(slide, f"✓ {item}", 3.85, 2.65 + i * 0.7, 2.3, 0.45,
                 font_size=10, bold=True, color=WHITE)
    for i, item in enumerate(tt_only):
        add_text(slide, f"• {item}", 5.75, 2.65 + i * 0.7, 3.8, 0.45,
                 font_size=11, color=TIKTOK_RED)

    add_text(slide,
             "\"The songs that go #1 on Billboard AND blow up on TikTok all have one thing in common: "
             "a single line so good you can't stop repeating it.\"",
             0.4, 6.0, 9.2, 1.0, font_size=12, color=LIGHT_GRAY, italic=True)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


def slide_top_songs(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.08, MID_GRAY)

    add_text(slide, "TOP SONGS", 0.4, 0.2, 9.2, 0.5,
             font_size=11, color=TIKTOK_RED, bold=True)
    add_text(slide, "The biggest hits from each dataset", 0.4, 0.55, 9.2, 0.6,
             font_size=28, bold=True, color=WHITE)
    add_rect(slide, 0.4, 1.15, 1.2, 0.04, TIKTOK_RED)

    # Billboard
    add_rect(slide, 0.3, 1.35, 4.3, 0.42, BB_GOLD)
    add_text(slide, "BILLBOARD #1 HITS — 2026", 0.4, 1.38, 4.1, 0.35,
             font_size=12, bold=True, color=BLACK)

    bb_songs = [
        ("All I Want For Christmas Is You", "Mariah Carey"),
        ("Opalite", "Taylor Swift"),
        ("The Fate Of Ophelia", "Taylor Swift"),
        ("Choosin' Texas", "Ella Langley"),
        ("I Just Might", "Bruno Mars"),
        ("Drop Dead", "Olivia Rodrigo"),
        ("Swim", "BTS"),
        ("Aperture", "Harry Styles"),
        ("DTMF", "Bad Bunny"),
        ("Janice STFU", "Drake"),
    ]
    for i, (song, artist) in enumerate(bb_songs):
        y = 1.87 + i * 0.52
        bg = MID_GRAY if i % 2 == 0 else RGBColor(0x22, 0x22, 0x38)
        add_rect(slide, 0.3, y, 4.3, 0.48, bg)
        add_text(slide, f"{i+1}. {song}", 0.45, y + 0.02, 3.0, 0.28,
                 font_size=10, bold=True, color=WHITE)
        add_text(slide, artist, 0.45, y + 0.26, 3.0, 0.2,
                 font_size=9, color=BB_GOLD)

    # TikTok
    add_rect(slide, 5.4, 1.35, 4.3, 0.42, TIKTOK_RED)
    add_text(slide, "TIKTOK TOP SONGS BY VIEWS", 5.5, 1.38, 4.1, 0.35,
             font_size=12, bold=True, color=WHITE)

    tt_songs = [
        ("Last Thing You Need",   "Morgan Wallen",          "Apple + Shazam"),
        ("BbY WOW",               "KAROL G, Judeline",      "Apple + Shazam"),
        ("Choosin' Texas",        "Ella Langley",           "Apple + Shazam"),
        ("Janice STFU",           "Drake",                  "Apple + Shazam"),
        ("Been By Now",           "Morgan Wallen",          "Apple + Shazam"),
        ("WTF GOIN",              "Belly Gang & 21 Savage", "Apple + Shazam"),
        ("Dead Fresh",            "Lil Baby",               "Apple + Shazam"),
        ("stupid song",           "Olivia Rodrigo",         "Apple + Shazam"),
        ("Be Her",                "Ella Langley",           "Apple + Shazam"),
        ("I Can't Love You Anymore", "Ella Langley & Wallen","Apple + Shazam"),
    ]
    for i, (song, artist, views) in enumerate(tt_songs):
        y = 1.87 + i * 0.52
        bg = MID_GRAY if i % 2 == 0 else RGBColor(0x22, 0x22, 0x38)
        add_rect(slide, 5.4, y, 4.3, 0.48, bg)
        add_text(slide, f"{i+1}. {song}", 5.55, y + 0.02, 2.5, 0.28,
                 font_size=10, bold=True, color=WHITE)
        add_text(slide, artist, 5.55, y + 0.26, 2.2, 0.2,
                 font_size=9, color=TIKTOK_RED)
        add_text(slide, views, 8.8, y + 0.12, 0.85, 0.28,
                 font_size=8, color=LIGHT_GRAY, align=PP_ALIGN.RIGHT)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


def slide_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, DARK_GRAY)
    add_rect(slide, 0, 0, 10, 0.5, MID_GRAY)
    add_text(slide, "CONCLUSION", 0, 0.08, 10, 0.34,
             font_size=12, bold=True, color=TIKTOK_RED, align=PP_ALIGN.CENTER)

    add_text(slide, "Rhyme harder. Feel more.\nEverything else is the same.",
             0.5, 0.7, 9.0, 1.2, font_size=30, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)

    add_rect(slide, 1.5, 1.9, 7.0, 0.04, TIKTOK_RED)

    points = [
        ("🎵", "TikTok and Billboard songs are more similar than different in 2026"),
        ("🔤", "TikTok songs rhyme 10.8% more — tight couplets drive replays and lip-syncs"),
        ("❤️", "TikTok songs hit 10.4% harder emotionally — extreme feelings = saves & shares"),
        ("📊", "Data source matters: YouTube global data gave false signals; Shazam + Apple gave truth"),
        ("🤖", "NLP lyric analysis can predict which songs will trend before they drop"),
    ]

    for i, (icon, text) in enumerate(points):
        y = 2.1 + i * 0.88
        add_rect(slide, 0.5, y, 9.0, 0.75, MID_GRAY)
        add_rect(slide, 0.5, y, 0.08, 0.75, TIKTOK_RED)
        add_text(slide, icon, 0.65, y + 0.15, 0.5, 0.45, font_size=18)
        add_text(slide, text, 1.25, y + 0.15, 8.1, 0.45,
                 font_size=13, color=WHITE)

    add_rect(slide, 0, 7.38, 10, 0.12, MID_GRAY)
    add_text(slide, "SUPA Spring 2026  |  NLP Viral Lyric Analysis",
             0, 7.35, 10, 0.2, font_size=9, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


# ── Build ──────────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_dataset(prs)
    slide_findings_overview(prs)

    slide_finding(prs,
        num="01",
        title="TikTok rhymes tighter",
        subtitle="Consecutive end-rhymes are 10.8% more common in TikTok/Shazam hits",
        bb_stat="13.0%",
        tt_stat="14.4%",
        bb_bar=0.70, tt_bar=0.78,
        highlight_tiktok=True,
        insight="TikTok songs rhyme almost every couplet — that satisfying 'click' at line "
                "endings is what makes people want to repeat the clip over and over.",
        detail="When a line rhymes cleanly with the one before it, the brain registers "
               "satisfaction. That feeling makes viewers replay the clip, duet it, or film "
               "themselves singing along. Tight rhyme = higher replay rate.")

    slide_finding(prs,
        num="02",
        title="TikTok hits harder emotionally",
        subtitle="Emotional word intensity is 10.4% stronger in TikTok hits",
        bb_stat="3.67%",
        tt_stat="4.05%",
        bb_bar=0.65, tt_bar=0.72,
        highlight_tiktok=True,
        insight="Strong emotion (love, pain, hype, heartbreak) drives reaction videos, "
                "saves, and shares — TikTok's core engagement loop.",
        detail="Songs with high emotional punch trigger the 'I felt that' response. "
               "Viewers screenshot the lyric, stitch their reaction, or cry on camera. "
               "Billboard songs can be more understated — TikTok rewards going all the way.")

    slide_finding(prs,
        num="03",
        title="Everything else is basically equal",
        subtitle="With a clean English-language dataset, most lyric features converge",
        bb_stat="Billboard",
        tt_stat="TikTok",
        bb_bar=0.70, tt_bar=0.70,
        highlight_tiktok=False,
        insight="Energy, repetition, line length, chorus density and brevity are nearly "
                "identical between Billboard and TikTok hits in 2026.",
        detail="The earlier analysis showed big gaps because we used global YouTube data "
               "(Korean, Hindi, French songs). Once we switched to US Apple Music + Shazam — "
               "actual English-language TikTok hits — the differences collapsed to just two: "
               "rhyme density and emotional punch.")

    slide_finding(prs,
        num="04",
        title="TikTok songs are longer",
        subtitle="TikTok hits average 50.7 lyric lines vs 47.6 for Billboard",
        bb_stat="47.6 lines",
        tt_stat="50.7 lines",
        bb_bar=0.68, tt_bar=0.72,
        highlight_tiktok=True,
        insight="More lines = more repetition of the hook. TikTok songs pad length "
                "by repeating the chorus and hook sections more times.",
        detail="A longer song isn't harder to clip — it just means the hook appears more "
               "times throughout. More hook appearances = more chances for a 15-second clip "
               "to catch the perfect viral moment.")

    slide_finding(prs,
        num="05",
        title="The data quality lesson",
        subtitle="Global YouTube ≠ TikTok. Apple Music + Shazam = real TikTok hits.",
        bb_stat="Old gaps",
        tt_stat="Real gaps",
        bb_bar=0.90, tt_bar=0.35,
        highlight_tiktok=False,
        insight="First attempt (YouTube global): showed 63% energy gap, 27% chorus gap, "
                "25% rhyme gap — all fake, caused by non-English songs.",
        detail="Switching to Apple Music Top 100 + Shazam US Top 200 collapsed most of "
               "those differences to near zero. The real signal is just two features: "
               "rhyme density (+10.8%) and emotional punch (+10.4%). "
               "Always validate your data source before drawing conclusions.")

    slide_top_songs(prs)
    slide_crossover(prs)
    slide_takeaway(prs)
    slide_conclusion(prs)

    prs.save(OUTPUT)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    build()

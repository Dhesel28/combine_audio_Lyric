"""
Weekly Update Slide — October 7, 2026
Expanded standalone presentation.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

PPTX_OUT = Path("/Volumes/Dhesel Driv /Work") / "Weekly_Update_Oct7_2026.pptx"

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x11, 0x11, 0x11)
GRAY  = RGBColor(0x88, 0x88, 0x88)
LGRAY = RGBColor(0xE8, 0xE8, 0xE8)
FLGRAY= RGBColor(0xF7, 0xF7, 0xF7)
RED   = RGBColor(0xE5, 0x20, 0x3A)
GREEN = RGBColor(0x1A, 0xA3, 0x4A)
BLUE  = RGBColor(0x1A, 0x73, 0xE8)
DKRED = RGBColor(0xB0, 0x10, 0x20)


def _bg(slide):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = WHITE

def _box(slide, left, top, w, h, color=WHITE):
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

def _rule(slide, left, top, w, color=LGRAY, height=0.02):
    _box(slide, left, top, w, height, color)

def _footer(slide, note=""):
    _rule(slide, 0.6, 6.8, 8.8)
    if note:
        _t(slide, note, 0.6, 6.9, 8.8, 0.35, size=9, color=GRAY, italic=True)

def _section_label(slide, text):
    _t(slide, text, 0.6, 0.92, 8.8, 0.26, size=9, bold=True, color=GRAY)

def _slide_header(slide, title, subtitle=None):
    _bg(slide)
    _t(slide, title, 0.6, 0.28, 8.8, 0.58, size=26, bold=True)
    if subtitle:
        _t(slide, subtitle, 0.6, 0.88, 8.8, 0.28, size=10, color=GRAY, italic=True)
        _rule(slide, 0.6, 1.18, 8.8)
    else:
        _rule(slide, 0.6, 0.92, 8.8)


# ── Slide 1: Title ────────────────────────────────────────────────────────────

def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _box(s, 0.6, 1.5, 0.06, 2.8, RED)
    _t(s, "Weekly Update", 0.9, 1.5,  8.5, 1.0, size=48, bold=True)
    _t(s, "October 7, 2026",  0.9, 2.6,  8.5, 0.65, size=26, bold=True, color=RED)
    _t(s, "TikTok Viral Detector  ·  Chartmetric Data  ·  Billboard Comparison  ·  Model Retraining",
       0.9, 3.38, 8.5, 0.45, size=12, color=GRAY)
    _rule(s, 0.9, 3.92, 8.1, LGRAY)
    stats = [("2,737", "songs scraped"), ("995", "with lyrics"), ("590", "Billboard songs"), ("0.956", "new model F1")]
    for i, (val, lbl) in enumerate(stats):
        x = 0.9 + i * 2.3
        _t(s, val, x, 4.1,  2.1, 0.55, size=22, bold=True, color=RED)
        _t(s, lbl, x, 4.65, 2.1, 0.3,  size=9,  color=GRAY)
    _footer(s, "Dhesel  ·  SUPA Spring 2026")


# ── Slide 2: Agenda ───────────────────────────────────────────────────────────

def slide_agenda(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "This Week at a Glance")

    items = [
        ("01", "Chartmetric Data Scrape",      "40 weeks of TikTok charts → 2,737 songs"),
        ("02", "Genius Lyrics Fetch",           "995 songs now have full lyrics"),
        ("03", "Billboard vs TikTok Analysis",  "590 Billboard vs 995 TikTok songs compared"),
        ("04", "Model Retraining",              "Training set grew from 231 → 1,218 songs"),
        ("05", "Problem Found",                 "Class imbalance: 92% viral → scores too high"),
        ("06", "Next Week",                     "Add non-hits to fix model and re-score Tatiana"),
    ]
    for i, (num, head, sub) in enumerate(items):
        y = 1.08 + i * 0.92
        _box(s, 0.55, y, 8.95, 0.84, FLGRAY if i % 2 == 0 else WHITE)
        _box(s, 0.55, y, 0.04, 0.84, RED)
        _t(s, num,  0.72, y + 0.16, 0.55, 0.38, size=11, bold=True, color=RED)
        _t(s, head, 1.32, y + 0.1,  5.5,  0.35, size=13, bold=True)
        _t(s, sub,  1.32, y + 0.5,  7.8,  0.28, size=10, color=GRAY)

    _footer(s, "6 topics covered  ·  October 7, 2026")


# ── Slide 3: Chartmetric Scrape ───────────────────────────────────────────────

def slide_chartmetric_scrape(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Chartmetric — 40 Weeks of TikTok Data",
                  "Fetched every weekly chart from Jan 5 → Oct 5, 2026")

    # Big stats
    stats = [
        ("40",    "weeks\nscraped",    "Jan 5 → Oct 5, 2026"),
        ("7,000", "raw rows",          "100 sounds + 100 tracks\nper week"),
        ("2,737", "unique songs",      "After deduplication\nby song + artist"),
        ("2×",    "chart types",       "TikTok Sounds\n+ TikTok Tracks"),
    ]
    for i, (val, lbl, note) in enumerate(stats):
        x = 0.55 + i * 2.35
        _box(s, x, 1.3, 2.22, 2.0, FLGRAY)
        _box(s, x, 1.3, 2.22, 0.04, RED)
        _t(s, val,  x + 0.15, 1.42, 1.9, 0.72, size=32, bold=True, color=RED)
        _t(s, lbl,  x + 0.15, 2.18, 1.9, 0.45, size=10, bold=True)
        _t(s, note, x + 0.15, 2.65, 1.9, 0.55, size=8,  color=GRAY)

    _rule(s, 0.6, 3.45, 8.8)

    # How it works
    _t(s, "HOW IT WORKS", 0.6, 3.55, 8.8, 0.26, size=9, bold=True, color=GRAY)
    steps = [
        ("Authenticate", "POST /api/token with refresh token → 1-hour Bearer token cached to disk"),
        ("Fetch Sounds",  "GET /api/charts/tiktok/sounds?date=YYYY-MM-DD&limit=100"),
        ("Fetch Tracks",  "GET /api/charts/tiktok/tracks?date=YYYY-MM-DD&limit=100"),
        ("Deduplicate",   "Keep highest video_count row per (song, artist) pair across all weeks"),
        ("Save",          "cm_training_songs.csv → /Volumes/Dhesel Driv /Work/"),
    ]
    for i, (step, desc) in enumerate(steps):
        x = 0.55 + (i % 3) * 3.08
        y = 3.85 + (i // 3) * 1.05
        _box(s, x, y, 2.95, 0.95, FLGRAY)
        _t(s, step, x + 0.12, y + 0.08, 2.7, 0.3,  size=10, bold=True, color=RED)
        _t(s, desc, x + 0.12, y + 0.42, 2.7, 0.48, size=8,  color=GRAY)

    _footer(s, "Script: fetch_cm_2026.py  ·  2s delay between API calls  ·  Token auto-refreshes")


# ── Slide 4: Lyrics Fetch ─────────────────────────────────────────────────────

def slide_lyrics_fetch(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Genius Lyrics Fetch — 2,737 Songs",
                  "Fetched lyrics for all Chartmetric songs — saved to external drive")

    # Result bars
    _t(s, "RESULTS", 0.6, 1.28, 8.8, 0.26, size=9, bold=True, color=GRAY)

    BAR_MAX = 7.5
    bars = [
        ("Songs with lyrics",    995,  2737, RED,   "995 / 2,737  (36%)"),
        ("Without lyrics",      1742,  2737, LGRAY, "1,742 / 2,737  (64%)"),
    ]
    for i, (label, val, total, color, note) in enumerate(bars):
        y = 1.6 + i * 0.88
        _t(s, label, 0.6, y, 3.2, 0.36, size=11, bold=(i == 0))
        _box(s, 0.6, y + 0.42, BAR_MAX * (val / total), 0.26, color)
        _box(s, 0.6, y + 0.42, BAR_MAX, 0.26, LGRAY if i == 0 else WHITE)
        _box(s, 0.6, y + 0.42, BAR_MAX * (val / total), 0.26, color)
        _t(s, note, 8.2, y + 0.42, 1.5, 0.3, size=9, bold=True,
           color=RED if i == 0 else GRAY)

    _rule(s, 0.6, 3.42, 8.8)

    # Why 64% have no lyrics
    _t(s, "WHY 1,742 SONGS HAVE NO LYRICS", 0.6, 3.52, 8.8, 0.26, size=9, bold=True, color=GRAY)
    reasons = [
        ("Instrumentals & Beats",
         "Background music, lo-fi, binaural beats, piano pieces — "
         "no lyrics exist anywhere."),
        ('"Original Sound" clips',
         "TikTok user-created audio with no real song attached — "
         "Genius has nothing to search."),
        ("Non-English Tracks",
         "Thai, Indonesian, Vietnamese, Khmer, Arabic tracks — "
         "Genius coverage is limited outside English."),
        ("Obscure / Viral Sounds",
         "Niche regional music, meme sounds, and viral audio clips "
         "that never had official lyrics published."),
    ]
    for i, (head, body) in enumerate(reasons):
        x = 0.55 + (i % 2) * 4.6
        y = 3.82 + (i // 2) * 1.38
        _box(s, x, y, 4.45, 1.28, FLGRAY)
        _t(s, head, x + 0.15, y + 0.1,  4.1, 0.32, size=10, bold=True)
        _t(s, body, x + 0.15, y + 0.5,  4.1, 0.68, size=9,  color=GRAY)

    _footer(s, "Script: fetch_cm_lyrics.py  ·  0.5s delay  ·  Progress saved every 10 songs")


# ── Slide 5: Billboard vs TikTok Overview ────────────────────────────────────

def slide_bb_overview(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Billboard Hot 100 vs TikTok — Overview",
                  "Full year 2026  ·  Both datasets scraped, lyrics fetched, features computed")

    # Dataset comparison
    _t(s, "THE TWO DATASETS", 0.6, 1.28, 8.8, 0.26, size=9, bold=True, color=GRAY)

    # Billboard box
    _box(s, 0.55, 1.56, 4.35, 2.1, FLGRAY)
    _box(s, 0.55, 1.56, 4.35, 0.04, LGRAY)
    _t(s, "Billboard Hot 100", 0.75, 1.64, 3.9, 0.38, size=14, bold=True)
    bb_facts = [
        ("590", "songs with lyrics"),
        ("40",  "weekly charts scraped"),
        ("All genres", "pop, hip-hop, country, R&B"),
        ("Source", "billboard.com scraper"),
    ]
    for i, (val, lbl) in enumerate(bb_facts):
        y = 2.1 + i * 0.36
        _t(s, f"{val}  —  {lbl}", 0.75, y, 4.0, 0.3, size=10, color=BLACK)

    # TikTok box
    _box(s, 5.1, 1.56, 4.45, 2.1, FLGRAY)
    _box(s, 5.1, 1.56, 4.45, 0.04, RED)
    _t(s, "Chartmetric TikTok", 5.3, 1.64, 4.0, 0.38, size=14, bold=True, color=RED)
    tt_facts = [
        ("995",  "songs with lyrics"),
        ("2,737","total unique songs"),
        ("Global", "sounds + tracks charts"),
        ("Source", "Chartmetric API"),
    ]
    for i, (val, lbl) in enumerate(tt_facts):
        y = 2.1 + i * 0.36
        _t(s, f"{val}  —  {lbl}", 5.3, y, 4.0, 0.3, size=10, color=BLACK)

    _rule(s, 0.6, 3.78, 8.8)

    # Features analysed
    _t(s, "10 LYRIC FEATURES COMPARED", 0.6, 3.88, 8.8, 0.26, size=9, bold=True, color=GRAY)
    features = [
        "Hook Strength", "Energy Score", "Rhyme Density",
        "Repetition Rate", "Chorus/Hook Density", "Brevity",
        "Emotional Punch", "Vocab Richness", "Avg Line Length", "Song Length",
    ]
    for i, feat in enumerate(features):
        x = 0.55 + (i % 5) * 1.88
        y = 4.18 + (i // 5) * 0.55
        _box(s, x, y, 1.8, 0.42, FLGRAY)
        _t(s, feat, x + 0.1, y + 0.07, 1.6, 0.3, size=8, color=BLACK)

    _footer(s, "Report: /Volumes/Dhesel Driv /Work/lyric_comparison_report.txt")


# ── Slide 6: Billboard vs TikTok — Detailed Results ──────────────────────────

def slide_bb_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Billboard vs TikTok — Feature by Feature",
                  "590 Billboard songs  ·  995 Chartmetric TikTok songs")

    # Table header
    _box(s, 0.55, 1.26, 9.1, 0.36, BLACK)
    _t(s, "Feature",       0.7,  1.3,  3.8, 0.28, size=9, bold=True, color=WHITE)
    _t(s, "Billboard",     4.6,  1.3,  1.4, 0.28, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _t(s, "TikTok",        6.1,  1.3,  1.4, 0.28, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _t(s, "Difference",    7.6,  1.3,  1.8, 0.28, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    rows = [
        ("Hook Strength  (most-repeated line freq.)",   0.0817, 0.1044, +27.8),
        ("Rhyme Density  (consecutive end-rhymes)",     0.1297, 0.1535, +18.4),
        ("Repetition Rate  (fraction of lines that repeat)", 0.2415, 0.2834, +17.4),
        ("Song Length  (total lyric lines)",            47.56,  50.65,  +6.5),
        ("Brevity  (optimal 4–9 words = 1.0)",          0.9093, 0.9172, +0.9),
        ("Avg Line Length  (words per line)",           7.857,  7.062,  -10.1),
        ("Vocabulary Richness  (unique word ratio)",    0.4031, 0.3857, -4.3),
        ("Chorus/Hook Density  (hook section %)",       0.4887, 0.3897, -20.3),
        ("Emotional Punch  (sentiment intensity)",      0.0367, 0.0278, -24.4),
        ("Energy Score  (energy word density)",         0.0223, 0.0157, -29.6),
    ]

    for i, (feat, bb, tt, pct) in enumerate(rows):
        y      = 1.64 + i * 0.51
        row_bg = FLGRAY if i % 2 == 0 else WHITE
        _box(s, 0.55, y, 9.1, 0.49, row_bg)
        _t(s, feat, 0.68, y + 0.1, 3.8, 0.3, size=8.5, color=BLACK)
        _t(s, f"{bb:.4f}", 4.6,  y + 0.1, 1.4, 0.3, size=9, align=PP_ALIGN.CENTER)
        _t(s, f"{tt:.4f}", 6.1,  y + 0.1, 1.4, 0.3, size=9,
           color=RED if pct > 0 else GRAY, bold=abs(pct) >= 10, align=PP_ALIGN.CENTER)
        arrow  = "↑" if pct > 0 else "↓"
        diff_c = RED if pct > 0 else GRAY
        _t(s, f"{arrow} {abs(pct):.1f}%", 7.6, y + 0.1, 1.8, 0.3,
           size=9, bold=abs(pct) >= 10, color=diff_c, align=PP_ALIGN.CENTER)

    _footer(s, "↑ = TikTok higher  ·  ↓ = TikTok lower  ·  Bold = ≥10% gap")


# ── Slide 7: The TikTok Formula ───────────────────────────────────────────────

def slide_tiktok_formula(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "What Makes a TikTok Song Different?",
                  "Key takeaways from the Billboard vs TikTok comparison")

    higher = [
        ("+28%", "Hook Strength",
         "TikTok songs repeat their best line far more often. "
         "The brain locks onto one phrase — perfect for lip-syncs and duets."),
        ("+18%", "Rhyme Density",
         "Almost every couplet rhymes. The satisfying 'click' at line "
         "endings makes viewers want to repeat the 15-second clip."),
        ("+17%", "Repetition Rate",
         "More lines repeat across the whole song. Repeated lines become "
         "earworms that stick after just one short listen."),
    ]
    lower = [
        ("−30%", "Energy Score",
         "Lower energy-word density than Billboard. TikTok doesn't need "
         "hype words — one sticky hook does all the work."),
        ("−24%", "Emotional Punch",
         "Less overt sentiment. TikTok songs are minimal and repetitive "
         "rather than emotionally complex ballads."),
        ("−20%", "Chorus Density",
         "Less of the song is the chorus section. The hook line carries "
         "the song, not a big traditional chorus."),
    ]

    _t(s, "TIKTOK SCORES HIGHER ↑", 0.6, 1.28, 4.5, 0.26, size=9, bold=True, color=RED)
    for i, (pct, feat, note) in enumerate(higher):
        y = 1.56 + i * 1.52
        _box(s, 0.55, y, 4.35, 1.42, FLGRAY)
        _box(s, 0.55, y, 0.04, 1.42, RED)
        _t(s, pct,  0.72, y + 0.1,  1.1, 0.5,  size=22, bold=True, color=RED)
        _t(s, feat, 1.9,  y + 0.1,  2.8, 0.35, size=12, bold=True)
        _t(s, note, 0.72, y + 0.65, 3.8, 0.7,  size=9,  color=GRAY)

    _t(s, "TIKTOK SCORES LOWER ↓", 5.2, 1.28, 4.5, 0.26, size=9, bold=True, color=GRAY)
    for i, (pct, feat, note) in enumerate(lower):
        y = 1.56 + i * 1.52
        _box(s, 5.2, y, 4.45, 1.42, FLGRAY)
        _box(s, 5.2, y, 0.04, 1.42, LGRAY)
        _t(s, pct,  5.37, y + 0.1,  1.1, 0.5,  size=22, bold=True, color=GRAY)
        _t(s, feat, 6.55, y + 0.1,  2.9, 0.35, size=12, bold=True)
        _t(s, note, 5.37, y + 0.65, 4.0, 0.7,  size=9,  color=GRAY)

    _box(s, 0.55, 6.14, 9.1, 0.52, RGBColor(0xFF, 0xF0, 0xF0))
    _t(s, "The TikTok Formula:", 0.75, 6.2, 2.4, 0.3, size=10, bold=True, color=RED)
    _t(s, "Short lines  ·  One hook repeated  ·  Dense rhymes  ·  Minimal structure  ·  15 seconds is all it needs",
       3.2, 6.2, 6.2, 0.3, size=10, color=BLACK)

    _footer(s, "Based on 590 Billboard + 995 TikTok songs  ·  Full year 2026")


# ── Slide 8: Model Retraining ─────────────────────────────────────────────────

def slide_retraining(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Model Retraining — Before vs After",
                  "All 3 models retrained with Chartmetric data integrated")

    # Before / After comparison
    cols = [
        ("BEFORE", LGRAY, BLACK, [
            ("Training songs",  "231"),
            ("Training lines",  "12,282"),
            ("Line model F1",   "0.845"),
            ("Song model F1",   "0.690"),
            ("Weight range",    "0.80 – 2.00"),
            ("TikTok signal",   "None (chart tiers only)"),
        ]),
        ("AFTER", RED, WHITE, [
            ("Training songs",  "1,218"),
            ("Training lines",  "61,991"),
            ("Line model F1",   "0.843  (stable)"),
            ("Song model F1",   "0.956  ↑"),
            ("Weight range",    "0.80 – 8.42"),
            ("TikTok signal",   "Real video counts (Chartmetric)"),
        ]),
    ]
    for c, (label, hdr_color, hdr_txt, items) in enumerate(cols):
        x = 0.55 + c * 4.7
        _box(s, x, 1.28, 4.5, 0.42, hdr_color)
        _t(s, label, x + 0.2, 1.33, 4.1, 0.32, size=13, bold=True, color=hdr_txt)
        for i, (key, val) in enumerate(items):
            y = 1.72 + i * 0.72
            _box(s, x, y, 4.5, 0.66, FLGRAY if i % 2 == 0 else WHITE)
            _t(s, key, x + 0.15, y + 0.1,  2.1, 0.3, size=9,  color=GRAY)
            _t(s, val, x + 0.15, y + 0.36, 4.0, 0.26, size=11, bold=True,
               color=RED if c == 1 and "↑" in val else BLACK)

    # Why F1 jumped
    _rule(s, 0.6, 6.14, 8.8)
    _t(s, "Why song model F1 jumped 0.690 → 0.956:",
       0.6, 6.22, 4.0, 0.28, size=9, bold=True, color=GRAY)
    _t(s, "Real TikTok video counts replace chart-tier guesses as ground truth. "
          "Songs with 98M+ TikTok videos get weight 8.42× — the model learns from the strongest signal first.",
       0.6, 6.5, 9.0, 0.26, size=9, color=GRAY, italic=True)

    _footer(s, "GradientBoostingClassifier  ·  300 estimators  ·  max_depth=4  ·  StandardScaler")


# ── Slide 9: The Problem ──────────────────────────────────────────────────────

def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Problem — Model Too Optimistic",
                  "Why all of Tatiana's songs are scoring 93–100%")

    # Red alert box
    _box(s, 0.55, 1.28, 8.95, 1.1, RGBColor(0xFF, 0xED, 0xED))
    _box(s, 0.55, 1.28, 0.06, 1.1, RED)
    _t(s, "Class Imbalance: 92% Viral vs 8% Non-Viral",
       0.75, 1.36, 8.3, 0.42, size=16, bold=True, color=RED)
    _t(s, "The model barely knows what a non-viral song looks like — so it predicts high for almost everything.",
       0.75, 1.8,  8.3, 0.3,  size=10, color=GRAY)

    # Visual imbalance bar
    _t(s, "TRAINING DATA COMPOSITION", 0.6, 2.55, 8.8, 0.26, size=9, bold=True, color=GRAY)
    _box(s, 0.6, 2.85, 8.55, 0.52, LGRAY)
    _box(s, 0.6, 2.85, 8.55 * 0.92, 0.52, RED)
    _t(s, "Viral  1,077 songs  (92%)", 0.75, 2.9,  4.0, 0.38, size=10, bold=True, color=WHITE)
    _t(s, "Non-viral  92  (8%)",       8.3,  2.9,  1.0, 0.38, size=9,  color=BLACK)

    # Before / after imbalance
    _rule(s, 0.6, 3.55, 8.8)
    _t(s, "HOW WE GOT HERE", 0.6, 3.65, 8.8, 0.26, size=9, bold=True, color=GRAY)
    phases = [
        ("Old model  (balanced)",
         "95 viral  vs  92 non-viral  →  51% / 49%  ✓ Healthy split",
         GREEN, "0.690 F1"),
        ("After adding 987 Chartmetric songs  (all viral)",
         "1,077 viral  vs  92 non-viral  →  92% / 8%  ✗ Severely skewed",
         RED, "0.956 F1 (misleading)"),
    ]
    for i, (head, body, color, f1) in enumerate(phases):
        y = 3.95 + i * 1.1
        _box(s, 0.55, y, 8.95, 1.0, FLGRAY)
        _box(s, 0.55, y, 0.04, 1.0, color)
        _t(s, head, 0.72, y + 0.08, 7.0, 0.32, size=11, bold=True)
        _t(s, body, 0.72, y + 0.48, 6.5, 0.38, size=9, color=GRAY)
        _t(s, f1,   7.8,  y + 0.25, 1.4, 0.4,  size=11, bold=True, color=color, align=PP_ALIGN.RIGHT)

    _rule(s, 0.6, 6.18, 8.8)
    _t(s, "Result: Tatiana's songs all score 93–100% — the model can't distinguish between her stronger and weaker tracks.",
       0.6, 6.26, 9.0, 0.28, size=9, color=BLACK, italic=True)

    _footer(s, "F1 = 0.956 looks great but is inflated — model predicts viral for almost everything")


# ── Slide 10: Tatiana Current Rankings ───────────────────────────────────────

def slide_tatiana(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Tatiana DeMaria — Current Rankings",
                  "Scores from the retrained model  ·  Note: all high due to class imbalance (will be fixed next week)")

    _box(s, 0.55, 1.26, 8.95, 0.38, RGBColor(0xFF, 0xF0, 0xF0))
    _t(s, "⚠  Scores are currently inflated — next week's fix will give a meaningful spread.",
       0.72, 1.32, 8.5, 0.26, size=9, bold=True, color=RED)

    # Table header
    _box(s, 0.55, 1.66, 9.1, 0.36, BLACK)
    _t(s, "Song",        0.7,  1.7, 3.5, 0.26, size=9, bold=True, color=WHITE)
    _t(s, "Score",       4.3,  1.7, 1.0, 0.26, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _t(s, "Viral Line",  5.4,  1.7, 4.0, 0.26, size=9, bold=True, color=WHITE)

    songs = [
        ("Make Me Feel",         "100%", "Caught here"),
        ("You Make Me",          "100%", "You, You Make Me"),
        ("Save Me",              "100%", "God only knows"),
        ("Born Again Sinner",    "100%", "I'm a born again sinner"),
        ("Fresh Meat",           "100%", "Rotting here down in the belly of the beast"),
        ("What It Is About You", "99%",  "With your guilt free heart"),
        ("Without You",          "99%",  "Chasing the good life"),
        ("AWS",                  "98%",  "tired of running"),
        ("Gallery",              "96%",  "when you're afraid and all alone?"),
        ("London Don't Lie",     "94%",  "London don't lie"),
        ("Anxiety - Acoustic",   "94%",  "I hear the voices saying"),
        ("Too Much",             "93%",  "Too much it ties me up"),
    ]
    for i, (song, score, line) in enumerate(songs):
        y      = 2.04 + i * 0.37
        row_bg = FLGRAY if i % 2 == 0 else WHITE
        _box(s, 0.55, y, 9.1, 0.35, row_bg)
        _t(s, f"{i+1}. {song}", 0.68, y + 0.05, 3.5, 0.26, size=9)
        _t(s, score, 4.3, y + 0.05, 1.0, 0.26, size=9, bold=True,
           color=RED, align=PP_ALIGN.CENTER)
        _t(s, f'"{line}"', 5.4, y + 0.05, 4.0, 0.26, size=8, color=GRAY, italic=True)

    _footer(s, "13 songs analysed  ·  Scores will spread meaningfully after next week's rebalance")


# ── Slide 11: Next Week Plan ──────────────────────────────────────────────────

def slide_next_week(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _slide_header(s, "Next Week — Fix the Model",
                  "Add non-viral songs → rebalance → re-score Tatiana")

    steps = [
        ("01", "Source Non-Hit Songs",
         "Identify songs confirmed to have low TikTok traction — "
         "tracks that charted on Billboard but never went viral on TikTok, "
         "or songs with very low Chartmetric video counts (< 1,000 videos).",
         "Target: ~800–1,000 non-viral songs"),
        ("02", "Fetch Their Lyrics",
         "Run Genius lyrics fetch on the non-viral dataset. "
         "These are mostly English-language chart songs so Genius coverage should be high.",
         "Expected: ~70–80% lyrics hit rate"),
        ("03", "Retrain With Balanced Data",
         "Merge non-viral songs into training set. "
         "Target 50/50 split: ~1,000 viral (Chartmetric) vs ~1,000 non-viral. "
         "Song model should produce a proper spread of scores 10–90%.",
         "Expected F1: ~0.80–0.85 (honest)"),
        ("04", "Re-Score Tatiana's 13 Songs",
         "Run the retrained model on Tatiana's full catalog. "
         "Scores will now meaningfully separate stronger songs from weaker ones "
         "and give actionable recommendations.",
         "Deliverable: updated ranked report + slide"),
    ]

    for i, (num, head, body, target) in enumerate(steps):
        y = 1.28 + i * 1.3
        _box(s, 0.55, y, 8.95, 1.2, FLGRAY)
        _box(s, 0.55, y, 0.04, 1.2, GREEN)
        _t(s, num,    0.72, y + 0.12, 0.55, 0.42, size=13, bold=True, color=GREEN)
        _t(s, head,   1.35, y + 0.08, 5.8,  0.38, size=13, bold=True)
        _t(s, body,   1.35, y + 0.5,  5.8,  0.55, size=9,  color=GRAY)
        _box(s, 7.3, y + 0.3, 2.05, 0.55, WHITE)
        _t(s, target, 7.4, y + 0.38, 1.9, 0.38, size=8, color=GREEN, italic=True)

    _footer(s, "Goal: balanced 50/50 training set  ·  Meaningful Tatiana score spread  ·  Honest model performance")


# ── Slide 12: Close ───────────────────────────────────────────────────────────

def slide_close(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _box(s, 0.6, 1.6, 0.06, 2.6, RED)
    _t(s, "See you next week.", 0.9, 1.6,  8.5, 0.9, size=38, bold=True)
    _t(s, "October 14, 2026",   0.9, 2.6,  8.5, 0.6, size=20, color=RED, bold=True)

    _t(s, "What's coming:", 0.9, 3.4, 8.5, 0.32, size=11, bold=True, color=GRAY)
    coming = [
        "Non-viral song dataset sourced and lyrics fetched",
        "Models retrained on balanced 50/50 split",
        "Tatiana's 13 songs re-scored with meaningful spread",
    ]
    for i, item in enumerate(coming):
        _t(s, f"→  {item}", 0.9, 3.78 + i * 0.38, 8.5, 0.32, size=10, color=BLACK)

    summary = "2,737 songs  ·  995 lyrics  ·  590 Billboard  ·  40 weeks of 2026 data  ·  F1: 0.956 (needs rebalance)"
    _footer(s, summary)


# ── Build ─────────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_agenda(prs)
    slide_chartmetric_scrape(prs)
    slide_lyrics_fetch(prs)
    slide_bb_overview(prs)
    slide_bb_results(prs)
    slide_tiktok_formula(prs)
    slide_retraining(prs)
    slide_problem(prs)
    slide_tatiana(prs)
    slide_next_week(prs)
    slide_close(prs)

    prs.save(PPTX_OUT)
    print(f"Saved → {PPTX_OUT}  ({prs.slides.__len__()} slides)")


if __name__ == "__main__":
    build()

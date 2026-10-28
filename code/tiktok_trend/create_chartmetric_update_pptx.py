"""
TikTok Viral Detector — Chartmetric Update
Minimal white design matching the project's existing style.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path
from config import DATA_DIR

PPTX_OUT = DATA_DIR / "TikTok_Viral_Detector_Chartmetric.pptx"

# ── Colors ──────────────────────────────────────────────────────────────────
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x11, 0x11, 0x11)
GRAY  = RGBColor(0x88, 0x88, 0x88)
LGRAY = RGBColor(0xE8, 0xE8, 0xE8)
FLGRAY= RGBColor(0xF7, 0xF7, 0xF7)
RED   = RGBColor(0xE5, 0x20, 0x3A)


# ── Primitive helpers ────────────────────────────────────────────────────────

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


# ── Slides ───────────────────────────────────────────────────────────────────

def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _rule(s, 0.6, 2.0, 0.6, RED)                          # vertical accent bar (thin horizontal here)
    _box(s, 0.6, 2.05, 0.06, 1.8, RED)                    # thick left accent bar
    _t(s, "TikTok Viral Detector", 0.9, 1.0, 8.5, 0.9, size=38, bold=True, color=BLACK)
    _t(s, "Chartmetric Integration + Algorithm Update", 0.9, 2.05, 8.5, 0.6,
       size=16, color=GRAY)
    _t(s, "Dhesel  ·  2026", 0.9, 2.75, 8.5, 0.4, size=12, color=GRAY, italic=True)
    _footer(s, "3 models  ·  Chartmetric API  ·  Genius lyrics  ·  Streamlit dashboard")


def slide_overview(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "What was built", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _rule(s, 0.6, 0.98, 8.8)

    items = [
        ("01", "Line-Level Viral Detector  (Model 1)",
         "Scores every lyric line 0–100% for TikTok viral potential using "
         "hook structure, repetition, energy words, and sentiment."),
        ("02", "Snippet Detector  (Model 2)",
         "Finds the best 3-line clip window in a song — the exact segment "
         "most likely to get used on TikTok."),
        ("03", "Song-Level Classifier  (Model 3)",
         "Aggregates line scores into 13 song-level features and outputs a "
         "final TikTok viral probability for the whole track."),
        ("04", "Chartmetric Integration  (new)",
         "Fetches real TikTok video counts from Chartmetric API — "
         "sounds and tracks charts — to provide verified ground-truth labels."),
    ]
    for i, (num, head, body) in enumerate(items):
        y = 1.1 + i * 1.35
        _box(s, 0.55, y, 8.95, 1.2, FLGRAY)
        _t(s, num, 0.72, y + 0.18, 0.6, 0.5, size=11, bold=True, color=RED)
        _t(s, head, 1.35, y + 0.12, 7.8, 0.42, size=13, bold=True)
        _t(s, body, 1.35, y + 0.58, 7.8, 0.55, size=10, color=GRAY)

    _footer(s, "Pipeline: fetch → enrich → train → predict → Streamlit app")


def slide_chartmetric(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Chartmetric Data", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _rule(s, 0.6, 0.98, 8.8)

    # Left column — sounds
    _t(s, "TIKTOK SOUNDS", 0.6, 1.15, 4.0, 0.3, size=9, bold=True, color=GRAY)
    sounds = [
        ("I'll Never Let You Go",    "BCD Studio",          "6.8M"),
        ("Final Game Showdown",       "ProdByLucK",          "1.2M"),
        ("Sunset in Girona",          "Noah Preminger",      "4.3M"),
        ("we fell in love in october","girl in red",         "226K"),
        ("Voidness (Ultra Slowed)",   "Canneliys et al.",    "253K"),
        ("BbY WOW",                   "KAROL G",             "270K"),
        ("Don't Let Me Down",         "The Chainsmokers",    "54K"),
        ("Freak",                     "Doja Cat",            "159K"),
    ]
    for i, (song, artist, vc) in enumerate(sounds):
        y = 1.5 + i * 0.6
        if i % 2 == 0:
            _box(s, 0.55, y - 0.03, 4.1, 0.58, FLGRAY)
        _t(s, f"{song}  —  {artist}", 0.65, y + 0.05, 3.2, 0.38, size=9)
        _t(s, vc, 3.9, y + 0.05, 0.7, 0.38, size=9, bold=True, color=RED,
           align=PP_ALIGN.RIGHT)

    # Right column — tracks
    _t(s, "TIKTOK TRACKS  (video count)", 5.1, 1.15, 4.5, 0.3, size=9, bold=True, color=GRAY)
    tracks = [
        ("I'll Never Let You Go",    "BCD Studio",       "98M"),
        ("Beat Automotivo",          "WZ Beat",          "45M"),
        ("Laxed – Siren Beat",       "Jawsh 685",        "30M"),
        ("Love You So",              "King Khan & BBQ",  "27M"),
        ("Aesthetic",                "Tollan Kim",       "23M"),
        ("Banana",                   "Conkarah ft. Shaggy","21M"),
        ("Savage",                   "Megan Thee Stallion","9.4M"),
        ("Closer",                   "The Chainsmokers", "7.5M"),
    ]
    for i, (song, artist, vc) in enumerate(tracks):
        y = 1.5 + i * 0.6
        if i % 2 == 0:
            _box(s, 5.05, y - 0.03, 4.6, 0.58, FLGRAY)
        _t(s, f"{song}  —  {artist}", 5.15, y + 0.05, 3.7, 0.38, size=9)
        _t(s, vc, 9.0, y + 0.05, 0.7, 0.38, size=9, bold=True, color=BLACK,
           align=PP_ALIGN.RIGHT)

    _footer(s, "100 sounds + 100 tracks fetched  ·  Lyrics enriched via Genius API  ·  187 songs total, 66 with lyrics")


def slide_algorithm(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Algorithm Improvement", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _rule(s, 0.6, 0.98, 8.8)

    _t(s, "BEFORE  —  Chart-based labels only", 0.6, 1.1, 8.8, 0.32, size=9, bold=True, color=GRAY)
    before = [
        ("Tier 3/4  (chart hits)", "viral = 1", "Weight: equal 1.0"),
        ("Tier 0/1  (non-hits)",   "viral = 0", "Weight: equal 1.0"),
        ("No TikTok signal",       "—",         "Treats Billboard hit = TikTok hit"),
    ]
    for i, (cat, label, note) in enumerate(before):
        y = 1.45 + i * 0.52
        _box(s, 0.55, y - 0.03, 8.95, 0.5, RGBColor(0xFF, 0xF0, 0xF0))
        _t(s, cat,   0.7,  y + 0.05, 3.0, 0.38, size=10)
        _t(s, label, 3.75, y + 0.05, 1.5, 0.38, size=10, bold=True, color=GRAY)
        _t(s, note,  5.3,  y + 0.05, 4.0, 0.38, size=10, color=GRAY, italic=True)

    _rule(s, 0.6, 3.15, 8.8)
    _t(s, "AFTER  —  Chartmetric video count weights", 0.6, 3.25, 8.8, 0.32, size=9, bold=True, color=GRAY)
    after = [
        ("Tier -1  (Chartmetric TikTok song)", "viral = 1", "Weight: 2.0 + log(video_count / 50K)"),
        ("Tier 3/4  (chart hits)",             "viral = 1", "Weight: 0.8  — probable but unverified"),
        ("Tier 0/1  (non-hits)",               "viral = 0", "Weight: 1.0  — solid negatives"),
    ]
    for i, (cat, label, note) in enumerate(after):
        y = 3.6 + i * 0.65
        _box(s, 0.55, y - 0.03, 8.95, 0.6, FLGRAY)
        _t(s, cat,   0.7,  y + 0.08, 3.5, 0.42, size=10)
        _t(s, label, 4.25, y + 0.08, 1.2, 0.42, size=10, bold=True, color=RED)
        _t(s, note,  5.5,  y + 0.08, 4.0, 0.42, size=10, color=GRAY)

    _footer(s, "Weight range: 0.80× – 8.42×  ·  Higher video counts = stronger training signal")


def slide_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Model Results", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _rule(s, 0.6, 0.98, 8.8)

    stats = [
        ("0.851", "Line Model  5-fold CV F1",
         "Classifies individual lyric lines as hook-worthy or not.\n"
         "Features: energy, sentiment, brevity, section type, repetition."),
        ("0.792", "Song Model  5-fold CV F1",
         "Aggregates 13 song-level features to predict full-song viral potential.\n"
         "Trained with Chartmetric video-count sample weights."),
        ("8.42×", "Max sample weight",
         "Songs with 98M+ TikTok videos receive the highest confidence weight,\n"
         "anchoring the model on verified real-world signals."),
    ]
    for i, (stat, label, body) in enumerate(stats):
        y = 1.1 + i * 1.75
        _t(s, stat, 0.6, y, 3.0, 1.1, size=52, bold=True, color=RED)
        _t(s, label, 3.8, y + 0.1, 5.7, 0.45, size=13, bold=True)
        _t(s, body,  3.8, y + 0.6, 5.7, 0.9,  size=10, color=GRAY)
        _rule(s, 0.6, y + 1.55, 8.8)

    _footer(s, "GradientBoostingClassifier  ·  200 estimators  ·  max_depth=3  ·  StandardScaler")


def slide_features(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "13 Song-Level Features", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _rule(s, 0.6, 0.98, 8.8)

    features = [
        ("hook_line_ratio",      "% of all lines in chorus / hook / bridge sections"),
        ("repetition_rate",      "Fraction of lines that repeat (1 − unique / total)"),
        ("max_repeats_norm",     "Most-repeated line count ÷ 10, capped at 1"),
        ("avg_energy",           "Mean energy-word density across all lines"),
        ("avg_positive",         "Mean positive-sentiment word density"),
        ("avg_negative",         "Mean negative-sentiment word density"),
        ("avg_brevity",          "Mean brevity score — optimal range 4–9 words"),
        ("avg_viral_prob",       "Mean line-level viral probability from Model 1"),
        ("top_viral_prob",       "Single highest line-level viral probability"),
        ("strong_hook_density",  "Fraction of lines with viral probability ≥ 0.55"),
        ("viral_line_density",   "Fraction of lines with viral probability ≥ 0.75"),
        ("chorus_repeat_ratio",  "Fraction of section types that are hook sections"),
        ("line_count_norm",      "Total lyric lines ÷ 80, capped at 1"),
    ]

    col_split = 7
    for i, (name, desc) in enumerate(features[:col_split]):
        y = 1.08 + i * 0.78
        if i % 2 == 0:
            _box(s, 0.55, y - 0.05, 4.35, 0.73, FLGRAY)
        _t(s, name, 0.68, y + 0.02, 3.5, 0.3,  size=9, bold=True, color=BLACK)
        _t(s, desc, 0.68, y + 0.32, 3.5, 0.38, size=8, color=GRAY)

    for i, (name, desc) in enumerate(features[col_split:]):
        y = 1.08 + i * 0.78
        if i % 2 == 0:
            _box(s, 5.05, y - 0.05, 4.55, 0.73, FLGRAY)
        _t(s, name, 5.18, y + 0.02, 3.8, 0.3,  size=9, bold=True, color=BLACK)
        _t(s, desc, 5.18, y + 0.32, 3.8, 0.38, size=8, color=GRAY)

    _footer(s, "Features engineered from lyrics + ViralLineDetector predictions")


def slide_song_examples(prs):
    """
    'Too Much' by Tatiana DeMaria — every lyric line scored for hook potential.
    Color-coded: red rows = hook, gray = supporting, white = filler.
    """
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Too Much  —  Tatiana DeMaria", 0.6, 0.25, 7.0, 0.52, size=24, bold=True)
    _t(s, "Hook detection: every lyric line scored for TikTok / IG viral potential",
       0.6, 0.78, 8.8, 0.28, size=10, color=GRAY, italic=True)
    _rule(s, 0.6, 1.06, 8.8)

    # ── Column header ─────────────────────────────────────────────────────
    _box(s, 0.6, 1.08, 9.1, 0.34, BLACK)
    _t(s, "Lyric Line",    0.68, 1.11, 6.3, 0.26, size=9, bold=True, color=WHITE)
    _t(s, "Hook Score",    7.0,  1.11, 1.3, 0.26, size=9, bold=True, color=WHITE,
       align=PP_ALIGN.CENTER)
    _t(s, "Label",         8.4,  1.11, 1.0, 0.26, size=9, bold=True, color=WHITE,
       align=PP_ALIGN.RIGHT)

    # (score, lyric_line)  — unique lines in song order
    lines = [
        (0.97, "Too much it ties me up and I can't sleep"),
        (0.06, "Got a little bit older / Got a little bit bolder"),
        (0.14, "Got a little more drugs than I should"),
        (0.29, "But my heart burns still"),
        (0.99, "'Cos I am what I am"),
        (0.38, "We'll remain who we are"),
        (0.89, "You created a monster"),
        (0.84, "I'm out to get loud"),
        (0.86, "If there's a cure for this"),
        (0.25, "Hell, I don't want it"),
        (0.97, "If there's sanity in peace well I'll just keep running"),
        (0.86, "Still"),
        (0.54, "And my blood runs ill"),
        (0.26, "They tried to censor a pair of my lips"),
    ]

    BAR_W   = 1.25   # max bar width in inches
    ROW_H   = 0.36
    Y_START = 1.44

    for i, (score, text) in enumerate(lines):
        y = Y_START + i * ROW_H

        # Row background
        if score >= 0.80:
            row_bg   = RGBColor(0xFF, 0xED, 0xED)   # light pink — hook
            txt_col  = BLACK
        elif score >= 0.45:
            row_bg   = FLGRAY                         # light gray — supporting
            txt_col  = BLACK
        else:
            row_bg   = WHITE                          # white — filler
            txt_col  = GRAY

        _box(s, 0.6, y, 9.1, ROW_H - 0.02, row_bg)

        # Lyric text
        _t(s, text, 0.68, y + 0.04, 6.2, ROW_H - 0.06,
           size=9, color=txt_col)

        # Score bar — gray track + red fill
        bar_x = 7.05
        bar_y = y + ROW_H / 2 - 0.07
        _box(s, bar_x, bar_y, BAR_W, 0.14, LGRAY)
        if score > 0:
            _box(s, bar_x, bar_y, BAR_W * score, 0.14, RED)

        # Label: HOOK / score%
        if score >= 0.80:
            lbl      = "HOOK"
            lbl_col  = RED
            lbl_bold = True
        else:
            lbl      = f"{round(score*100)}%"
            lbl_col  = GRAY
            lbl_bold = False
        _t(s, lbl, 8.4, y + 0.04, 1.0, ROW_H - 0.06,
           size=9, bold=lbl_bold, color=lbl_col, align=PP_ALIGN.RIGHT)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_y = Y_START + len(lines) * ROW_H + 0.05
    _box(s, 0.6, legend_y, 0.14, 0.14, RGBColor(0xFF, 0xED, 0xED))
    _t(s, "Hook (≥80%)", 0.8, legend_y, 1.4, 0.18, size=8, color=GRAY)
    _box(s, 2.35, legend_y, 0.14, 0.14, FLGRAY)
    _t(s, "Supporting (45–79%)", 2.55, legend_y, 2.0, 0.18, size=8, color=GRAY)
    _box(s, 4.75, legend_y, 0.14, 0.14, WHITE)
    _t(s, "Filler (<45%)", 4.95, legend_y, 1.5, 0.18, size=8, color=GRAY)

    _footer(s, "Model 3 line-level scores  ·  Red bar = viral hook probability  ·  Tatiana DeMaria — Too Much")


def slide_weekly_update(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)

    # Header accent bar
    _box(s, 0.6, 0.35, 0.06, 0.55, RED)
    _t(s, "This Week's Update", 0.82, 0.35, 8.5, 0.55, size=28, bold=True)
    _t(s, "October 7, 2026  ·  What was built, what was found, what's next",
       0.82, 0.92, 8.5, 0.28, size=10, color=GRAY, italic=True)
    _rule(s, 0.6, 1.22, 8.8)

    # ── Section 1: What we did ────────────────────────────────────────────
    _t(s, "WHAT WE DID", 0.6, 1.3, 8.8, 0.28, size=9, bold=True, color=GRAY)

    did = [
        ("Scraped all 40 weeks of 2026 Chartmetric TikTok charts",
         "Fetched top 100 sounds + top 100 tracks per week → 7,000 raw rows → "
         "2,737 unique songs after deduplication."),
        ("Fetched Genius lyrics for every song",
         "929 new lyrics fetched. Total: 995 songs with lyrics "
         "(1,742 are instrumentals / non-English sounds — no lyrics available on Genius)."),
        ("Compared against Billboard Hot 100 2026",
         "590 Billboard songs vs 995 Chartmetric TikTok songs analysed across "
         "10 lyric features."),
    ]
    for i, (head, body) in enumerate(did):
        y = 1.6 + i * 0.88
        _box(s, 0.55, y, 8.95, 0.82, FLGRAY)
        _box(s, 0.55, y, 0.04, 0.82, RED)
        _t(s, head, 0.72, y + 0.06, 8.1, 0.3, size=11, bold=True)
        _t(s, body, 0.72, y + 0.42, 8.1, 0.36, size=9, color=GRAY)

    # ── Section 2: Key finding ────────────────────────────────────────────
    _rule(s, 0.6, 4.28, 8.8)
    _t(s, "KEY FINDING — BILLBOARD vs TIKTOK", 0.6, 4.36, 8.8, 0.28, size=9, bold=True, color=GRAY)

    findings = [
        ("↑ +28%", "Hook Strength",   "TikTok songs repeat their best line far more often"),
        ("↑ +18%", "Rhyme Density",   "TikTok lyrics rhyme almost every couplet"),
        ("↑ +17%", "Repetition Rate", "More lines repeat — earworms in 15 seconds"),
        ("↓ −20%", "Chorus Density",  "Less of the song is chorus — hook does the work"),
        ("↓ −30%", "Energy Score",    "Lower energy-word density than Billboard"),
    ]
    col_w = 1.73
    for i, (pct, feat, note) in enumerate(findings):
        x     = 0.55 + i * (col_w + 0.07)
        y     = 4.66
        color = RED if pct.startswith("↑") else GRAY
        _box(s, x, y, col_w, 1.72, FLGRAY)
        _t(s, pct,  x + 0.1, y + 0.1,  col_w - 0.2, 0.42, size=18, bold=True, color=color)
        _t(s, feat, x + 0.1, y + 0.55, col_w - 0.2, 0.32, size=10, bold=True)
        _t(s, note, x + 0.1, y + 0.9,  col_w - 0.2, 0.72, size=8,  color=GRAY)

    # ── Section 3: Next week ──────────────────────────────────────────────
    _rule(s, 0.6, 6.45, 8.8)
    _t(s, "NEXT WEEK", 0.6, 6.52, 1.2, 0.25, size=9, bold=True, color=GRAY)
    _t(s,
       "Add non-hit songs (confirmed TikTok failures) to balance the training data. "
       "Currently 92% of training songs are viral — the model needs more non-viral examples "
       "to distinguish between Tatiana's stronger and weaker songs accurately.",
       1.9, 6.52, 7.5, 0.4, size=9, color=BLACK)
    _box(s, 0.6, 6.52, 1.22, 0.36, RGBColor(0xFF, 0xF0, 0xF0))
    _t(s, "NEXT WEEK", 0.66, 6.57, 1.1, 0.25, size=9, bold=True, color=RED)

    _footer(s, "2,737 CM songs  ·  995 with lyrics  ·  590 Billboard  ·  40 weeks of 2026 data")


def slide_billboard_comparison(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Billboard Hot 100 vs TikTok Trending", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _t(s, "590 Billboard songs  ·  995 Chartmetric TikTok songs  ·  2026",
       0.6, 0.92, 8.8, 0.28, size=10, color=GRAY, italic=True)
    _rule(s, 0.6, 1.22, 8.8)

    # Column headers
    _box(s, 0.6, 1.26, 9.1, 0.34, BLACK)
    _t(s, "Feature",       0.68, 1.29, 3.5, 0.26, size=9, bold=True, color=WHITE)
    _t(s, "Billboard",     4.2,  1.29, 1.3, 0.26, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _t(s, "TikTok",        5.6,  1.29, 1.3, 0.26, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _t(s, "Difference",    7.3,  1.29, 2.0, 0.26, size=9, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    rows = [
        ("Hook Strength",        0.0817, 0.1044, +27.8),
        ("Rhyme Density",        0.1297, 0.1535, +18.4),
        ("Repetition Rate",      0.2415, 0.2834, +17.4),
        ("Avg Line Length (wds)",7.857,  7.062,  -10.1),
        ("Chorus/Hook Density",  0.4887, 0.3897, -20.3),
        ("Emotional Punch",      0.0367, 0.0278, -24.4),
        ("Energy Score",         0.0223, 0.0157, -29.6),
    ]

    BAR_MAX = 1.4   # max bar width in inches
    ROW_H   = 0.62
    Y_START = 1.62

    for i, (feat, bb, tt, pct) in enumerate(rows):
        y      = Y_START + i * ROW_H
        higher = pct > 0
        row_bg = FLGRAY if i % 2 == 0 else WHITE
        _box(s, 0.6, y, 9.1, ROW_H - 0.04, row_bg)

        # Feature name
        _t(s, feat, 0.68, y + 0.14, 3.4, 0.32, size=9, color=BLACK)

        # Billboard bar (gray)
        norm_bb = min(abs(bb) / max(abs(tt), abs(bb), 0.001), 1.0)
        _box(s, 4.2, y + 0.18, BAR_MAX * norm_bb, 0.18, LGRAY)
        _t(s, f"{bb:.3f}", 4.2, y + 0.38, 1.3, 0.2, size=8, color=GRAY, align=PP_ALIGN.CENTER)

        # TikTok bar (red)
        norm_tt = min(abs(tt) / max(abs(tt), abs(bb), 0.001), 1.0)
        _box(s, 5.6, y + 0.18, BAR_MAX * norm_tt, 0.18, RED)
        _t(s, f"{tt:.3f}", 5.6, y + 0.38, 1.3, 0.2, size=8, color=RED, align=PP_ALIGN.CENTER)

        # Difference pill
        arrow  = "↑" if higher else "↓"
        diff_c = RED if higher else GRAY
        _t(s, f"{arrow} {abs(pct):.1f}%", 7.3, y + 0.14, 2.0, 0.32,
           size=10, bold=True, color=diff_c, align=PP_ALIGN.CENTER)

    # Legend
    leg_y = Y_START + len(rows) * ROW_H + 0.05
    _box(s, 0.6, leg_y, 0.18, 0.14, LGRAY)
    _t(s, "Billboard", 0.85, leg_y, 1.2, 0.18, size=8, color=GRAY)
    _box(s, 2.3, leg_y, 0.18, 0.14, RED)
    _t(s, "TikTok / Chartmetric", 2.55, leg_y, 2.0, 0.18, size=8, color=GRAY)

    _footer(s, "TikTok wins on: hook strength, rhyme, repetition  ·  Billboard wins on: emotion, energy, chorus density, line length")


def slide_next_steps(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Next Steps", 0.6, 0.35, 8.8, 0.55, size=24, bold=True)
    _rule(s, 0.6, 0.98, 8.8)

    steps = [
        ("01", "Expand Chartmetric dataset",
         "Schedule weekly fetches to grow the CM training set beyond the current 66-lyric subset."),
        ("02", "Add audio features",
         "Integrate BPM, key, danceability from Spotify/Chartmetric to complement lyric signals."),
        ("03", "Instagram trend layer",
         "Use the IG sounds endpoint already in chartmetric_fetcher.py to add cross-platform signal."),
        ("04", "Fine-tune thresholds",
         "Calibrate the 0.65 / 0.45 probability cutoffs using precision-recall curves on held-out data."),
        ("05", "Artist-specific model",
         "Train per-genre or per-artist variants — viral signals differ between pop, hip-hop, K-pop."),
    ]
    for i, (num, head, body) in enumerate(steps):
        y = 1.1 + i * 1.12
        _t(s, num, 0.6,  y + 0.1,  0.55, 0.5, size=11, bold=True, color=RED)
        _t(s, head, 1.25, y + 0.05, 8.2, 0.42, size=13, bold=True)
        _t(s, body, 1.25, y + 0.5,  8.2, 0.5,  size=10, color=GRAY)
        if i < len(steps) - 1:
            _rule(s, 1.2, y + 1.0, 8.2)

    _footer(s, "Priority: expand training data first — model quality is bounded by label quantity")


def slide_close(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _box(s, 0.6, 1.8, 0.06, 2.2, RED)
    _t(s, "TikTok Viral Detector", 0.9, 1.8, 8.5, 0.8, size=34, bold=True)
    _t(s, "Chartmetric  ·  3 ML models  ·  Streamlit", 0.9, 2.7, 8.5, 0.5,
       size=16, color=GRAY)
    _t(s, "Line F1: 0.851   Song F1: 0.792   Weights: 0.80× – 8.42×",
       0.9, 3.35, 8.5, 0.45, size=13, color=BLACK)
    _footer(s, "github  ·  2026")


# ── Build ────────────────────────────────────────────────────────────────────

def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_overview(prs)
    slide_chartmetric(prs)
    slide_algorithm(prs)
    slide_results(prs)
    slide_features(prs)
    slide_song_examples(prs)
    slide_weekly_update(prs)
    slide_billboard_comparison(prs)
    slide_next_steps(prs)
    slide_close(prs)

    prs.save(PPTX_OUT)
    print(f"Saved → {PPTX_OUT}")


if __name__ == "__main__":
    build()

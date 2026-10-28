"""
TikTok Viral Predictor — Oct 14 2026 Update
Minimal white presentation. Covers lyrics fetch, evaluation, and retraining.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
EXTERNAL_DIR = Path("/Volumes/Dhesel Driv /Work")
OUT_DIR      = EXTERNAL_DIR if EXTERNAL_DIR.exists() else Path(__file__).parent / "data"
OUT_FILE     = OUT_DIR / "TikTok_Model_Update_Oct14_2026.pptx"

# ── Palette ───────────────────────────────────────────────────────────────────
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BLACK  = RGBColor(0x11, 0x11, 0x11)
GRAY   = RGBColor(0x88, 0x88, 0x88)
LGRAY  = RGBColor(0xF0, 0xF0, 0xF0)
MGRAY  = RGBColor(0xDD, 0xDD, 0xDD)
ACCENT = RGBColor(0x1A, 0x1A, 0x2E)   # near-black navy — minimal accent
GREEN  = RGBColor(0x22, 0xAA, 0x66)   # improvement indicator


# ── Primitives ────────────────────────────────────────────────────────────────

def _bg(slide):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = WHITE


def _box(slide, left, top, w, h, color=LGRAY):
    from pptx.util import Emu
    s = slide.shapes.add_shape(1,
        Inches(left), Inches(top), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    return s


def _t(slide, text, left, top, w, h,
       size=12, bold=False, color=BLACK,
       align=PP_ALIGN.LEFT, italic=False):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    r   = p.add_run()
    r.text = text
    r.font.size  = Pt(size)
    r.font.bold  = bold
    r.font.color.rgb = color
    r.font.italic    = italic
    return txb


def _rule(slide, left, top, w, color=MGRAY, h=0.018):
    _box(slide, left, top, w, h, color)


def _footer(slide, text="TikTok Viral Predictor  ·  Oct 14, 2026"):
    _rule(slide, 0.5, 7.1, 9.0)
    _t(slide, text, 0.5, 7.18, 9.0, 0.3, size=8, color=GRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ══════════════════════════════════════════════════════════════════════════════

def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    # Left accent bar
    _box(s, 0.5, 1.8, 0.06, 2.6, ACCENT)

    _t(s, "TikTok Viral Predictor", 0.75, 1.8, 8.5, 0.9,
       size=38, bold=True, color=BLACK)
    _t(s, "Model Update", 0.75, 2.7, 8.5, 0.65,
       size=28, bold=False, color=GRAY)

    _rule(s, 0.75, 3.55, 5.0, ACCENT)

    _t(s, "October 14, 2026", 0.75, 3.75, 5.0, 0.4,
       size=13, color=GRAY)

    # Three pill stats at bottom
    pills = [
        ("4,785", "songs analyzed"),
        ("95.6%", "lyrics coverage"),
        ("5,319", "training songs"),
    ]
    for i, (num, lbl) in enumerate(pills):
        x = 0.5 + i * 3.15
        _box(s, x, 5.4, 2.85, 1.1, LGRAY)
        _t(s, num, x + 0.18, 5.48, 2.5, 0.6,
           size=28, bold=True, color=ACCENT)
        _t(s, lbl, x + 0.18, 6.02, 2.5, 0.35,
           size=10, color=GRAY)

    _footer(s)


def slide_what_we_did(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "What we did today", 0.5, 0.35, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0)

    steps = [
        ("01", "Fetched lyrics for Summer Dataset",
         "4,785 songs via Genius API — went from 14% to 95.6% coverage."),
        ("02", "Ran full model evaluation",
         "Tested old models on 4,612 songs. Found they called 93-96% of non-hits viral. "
         "Old model scored everything at 0.92+ - no discrimination."),
        ("03", "Identified the root cause",
         "Original training set had only ~200 songs. Model learned 'good lyrics = viral' "
         "and could not distinguish TikTok virality from mainstream quality."),
        ("04", "Retrained all 3 models",
         "New training set: 5,319 songs with proper negatives (Summer tiers 0/1 + "
         "Billboard flops). Models now separate viral from non-viral."),
    ]

    for i, (num, title, body) in enumerate(steps):
        y = 1.2 + i * 1.42
        _box(s, 0.5, y, 0.55, 0.55, ACCENT)
        _t(s, num, 0.5, y + 0.06, 0.55, 0.45,
           size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _t(s, title, 1.2, y, 8.1, 0.45, size=13, bold=True)
        _t(s, body,  1.2, y + 0.48, 8.1, 0.75, size=10, color=GRAY)

    _footer(s)


def slide_data(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "Training data — before vs. after", 0.5, 0.35, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0)

    # Before column
    _t(s, "BEFORE", 0.5, 1.2, 4.3, 0.35, size=9, bold=True, color=GRAY)
    _rule(s, 0.5, 1.6, 4.3, MGRAY)
    before = [
        ("~200",  "songs in training set"),
        ("~200",  "with lyrics (catalog only)"),
        ("0",     "confirmed non-TikTok negatives"),
        ("?",     "Billboard songs as negatives"),
    ]
    for i, (num, lbl) in enumerate(before):
        y = 1.7 + i * 0.9
        _t(s, num, 0.5, y, 1.5, 0.55, size=30, bold=True, color=GRAY)
        _t(s, lbl, 2.1, y + 0.15, 2.7, 0.35, size=10, color=GRAY)

    # Divider
    _box(s, 4.95, 1.2, 0.06, 5.0, MGRAY)

    # After column
    _t(s, "AFTER", 5.15, 1.2, 4.3, 0.35, size=9, bold=True, color=ACCENT)
    _rule(s, 5.15, 1.6, 4.3, ACCENT, h=0.025)
    after = [
        ("5,319",  "songs in training set"),
        ("4,576",  "Summer Dataset lyrics (Genius)"),
        ("3,396",  "non-viral songs (proper negatives)"),
        ("526",    "Billboard flops as hard negatives"),
    ]
    for i, (num, lbl) in enumerate(after):
        y = 1.7 + i * 0.9
        _t(s, num, 5.15, y, 1.8, 0.55, size=30, bold=True, color=ACCENT)
        _t(s, lbl, 7.0,  y + 0.15, 2.5, 0.35, size=10, color=BLACK)

    _footer(s)


def slide_old_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "Old model — the problem", 0.5, 0.35, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0)

    _t(s, "Evaluated on 3,916 Summer Dataset songs with true tier labels",
       0.5, 1.15, 9.0, 0.35, size=11, color=GRAY)

    # Big stat row
    stats = [
        ("30.1%", "Accuracy",        GRAY,  "should be 65%+"),
        ("93%",   "Non-hits called viral",  RGBColor(0xCC,0x22,0x22), "should be <35%"),
        ("0.92",  "Avg viral score",  GRAY,  "everything near max"),
    ]
    for i, (num, lbl, c, note) in enumerate(stats):
        x = 0.5 + i * 3.1
        _box(s, x, 1.65, 2.85, 1.6, LGRAY)
        _t(s, num, x + 0.15, 1.75, 2.6, 0.8,
           size=38, bold=True, color=c)
        _t(s, lbl, x + 0.15, 2.52, 2.6, 0.35, size=10, bold=True, color=BLACK)
        _t(s, note, x + 0.15, 2.88, 2.6, 0.28, size=9, color=GRAY, italic=True)

    # Tier table
    _t(s, "Tier-level breakdown", 0.5, 3.55, 9.0, 0.35, size=12, bold=True)
    _rule(s, 0.5, 3.95, 9.0)

    cols = ["Tier", "Meaning", "Songs", "Predicted viral", "Avg score"]
    widths = [0.9, 2.2, 1.2, 2.2, 1.8]
    xs = [0.5, 1.5, 3.8, 5.1, 7.4]

    for j, (h, x, w) in enumerate(zip(cols, xs, widths)):
        _t(s, h, x, 4.05, w, 0.3, size=9, bold=True, color=GRAY,
           align=PP_ALIGN.LEFT)

    _rule(s, 0.5, 4.38, 9.0, BLACK, 0.012)

    rows = [
        ("0", "Non-hit",   "1,468", "93%  ✗", "0.892"),
        ("1", "Minor hit", "1,418", "96%  ✗", "0.928"),
        ("3", "Major hit",  "651",  "99%  ✓", "0.971"),
        ("4", "Super hit",  "379",  "99%  ✓", "0.969"),
    ]
    for i, row in enumerate(rows):
        y = 4.45 + i * 0.52
        if i % 2 == 0:
            _box(s, 0.48, y - 0.04, 9.1, 0.5, LGRAY)
        bad = row[0] in ("0", "1")
        for j, (val, x, w) in enumerate(zip(row, xs, widths)):
            c = RGBColor(0xCC,0x22,0x22) if (j == 3 and bad) else (GREEN if (j == 3 and not bad) else BLACK)
            _t(s, val, x, y + 0.05, w, 0.38, size=10,
               color=c, bold=(j == 3))

    _footer(s)


def slide_new_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "New model — after retraining", 0.5, 0.35, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0)

    _t(s, "Held-out test set: 1,064 songs  ·  80/20 stratified split",
       0.5, 1.15, 9.0, 0.35, size=11, color=GRAY)

    # Big stats
    stats = [
        ("67.2%", "Test accuracy",    GREEN, "+37pts vs before"),
        ("0.694", "AUC-ROC",          GREEN, "was ~0.5 (random)"),
        ("+0.142","Separation gap",   ACCENT, "viral vs non-viral"),
    ]
    for i, (num, lbl, c, note) in enumerate(stats):
        x = 0.5 + i * 3.1
        _box(s, x, 1.65, 2.85, 1.6, LGRAY)
        _t(s, num, x + 0.15, 1.75, 2.6, 0.8,
           size=38, bold=True, color=c)
        _t(s, lbl, x + 0.15, 2.52, 2.6, 0.35, size=10, bold=True, color=BLACK)
        _t(s, note, x + 0.15, 2.88, 2.6, 0.28, size=9, color=GRAY, italic=True)

    # Score distribution
    _t(s, "Score distribution — the key fix", 0.5, 3.55, 9.0, 0.35, size=12, bold=True)
    _rule(s, 0.5, 3.95, 9.0)

    for i, (label, score, median, c, w) in enumerate([
        ("True viral songs",     "0.498 avg", "0.491 median", GREEN,  4.3),
        ("True non-viral songs", "0.356 avg", "0.349 median", GRAY,   4.3),
    ]):
        y = 4.1 + i * 1.3
        _box(s, 0.5, y, w, 0.9, LGRAY)
        _t(s, label,  0.7, y + 0.08, 2.8, 0.35, size=10, bold=True, color=c)
        _t(s, score,  0.7, y + 0.45, 1.5, 0.35, size=12, bold=True, color=c)
        _t(s, median, 2.3, y + 0.45, 2.5, 0.35, size=10, color=GRAY)

        # Visual bar
        bar_w = score_val = float(score.split()[0])
        _box(s, 5.0, y + 0.3, bar_w * 3.5, 0.3, RGBColor(0xDD,0xDD,0xDD))
        _box(s, 5.0, y + 0.3, bar_w * 3.5, 0.3, c)

    _t(s, "Threshold: 0.50  ·  Viral songs now cluster above it, non-viral below it.",
       0.5, 6.75, 9.0, 0.3, size=9, color=GRAY, italic=True)

    _footer(s)


def slide_model1(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "Model 1 — Line Classifier", 0.5, 0.35, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0)

    _t(s, "Scores every lyric line for TikTok viral probability  ·  214,917 training lines",
       0.5, 1.15, 9.0, 0.35, size=11, color=GRAY)

    # Stats
    for i, (num, lbl, c) in enumerate([
        ("0.832", "5-fold CV F1",   GREEN),
        ("85.2%", "Accuracy",       GREEN),
        ("214k",  "Training lines", ACCENT),
    ]):
        x = 0.5 + i * 3.1
        _box(s, x, 1.6, 2.85, 1.0, LGRAY)
        _t(s, num, x + 0.15, 1.68, 2.6, 0.55, size=30, bold=True, color=c)
        _t(s, lbl, x + 0.15, 2.22, 2.6, 0.3, size=10, color=GRAY)

    # Feature importance
    _t(s, "Feature importances", 0.5, 2.85, 9.0, 0.35, size=12, bold=True)
    _rule(s, 0.5, 3.25, 9.0)

    features = [
        ("repetition_score",  0.936),
        ("position_ratio",    0.046),
        ("word_count_norm",   0.009),
        ("positive_density",  0.003),
        ("negative_density",  0.002),
        ("energy_density",    0.001),
    ]
    for i, (name, imp) in enumerate(features):
        y = 3.38 + i * 0.55
        _t(s, name, 0.5, y, 2.8, 0.38, size=10, color=BLACK)
        _t(s, f"{imp:.3f}", 3.4, y, 0.8, 0.38, size=10,
           color=GRAY, align=PP_ALIGN.RIGHT)
        bar_w = max(imp * 5.5, 0.05)
        c = ACCENT if imp > 0.1 else MGRAY
        _box(s, 4.3, y + 0.08, bar_w, 0.22, c)

    _t(s, "Repetition dominates (93.6%) — chorus lines repeat, verses don't. "
         "Position in song is the next most informative signal.",
       0.5, 6.65, 9.0, 0.4, size=9, color=GRAY, italic=True)

    _footer(s)


TATIANA_DATA = [
    # (song, song_prob, viral_line, why, runner_up)
    ("Save Me",            0.894, "God only knows",
     "repeats 4x · chorus/hook",
     "God only knows"),
    ("You Make Me",        0.832, "My heart",
     "rhymes · chorus/hook",
     "I can breathe"),
    ("Born Again Sinner",  0.638, "I'm a born again sinner",
     "repeats 8x · perfect length · chorus",
     "I'm a born again sinner"),
    ("Make Me Feel",       0.572, "Caught in the thrill of the chase",
     "repeats 3x · perfect length · chorus",
     "Still man I can't stand the taste"),
    ("Fresh Meat",         0.504, "Rotting here down in the belly of the beast",
     "repeats 6x · high energy · perfect length",
     "Rotting here down in the belly of the beast"),
    ("Too Much",           0.277, "Too much it ties me up",
     "repeats 6x · perfect length",
     "Too much it ties me up"),
    ("What It Is About You", 0.270, "With your guilt free heart",
     "repeats 7x · uplifting",
     "With your guilt free heart"),
    ("Gallery",            0.262, "Sketching with blood and darts",
     "repeats 2x · emotional",
     "Sketching with blood and darts"),
    ("Without You",        0.248, "Chasing the good life",
     "repeats 6x · rhymes · uplifting",
     "Chasing the good life"),
    ("AWS",                0.240, "I finally found my way into my mind",
     "repeats 3x · perfect length",
     "now I need a way out"),
    ("Anxiety - Acoustic", 0.222, "I hear the voices saying",
     "repeats 3x · perfect length",
     "You're not the one who'll be saved"),
    ("London Don't Lie",   0.126, "London don't lie",
     "repeats 13x · rhymes · emotional",
     "London don't lie"),
]


def slide_tatiana_rankings(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "Tatiana DeMaria — TikTok Viral Rankings", 0.5, 0.35, 9.0, 0.6,
       size=24, bold=True)
    _t(s, "Retrained models  ·  12 songs analyzed  ·  threshold 0.55",
       0.5, 0.95, 9.0, 0.32, size=10, color=GRAY)
    _rule(s, 0.5, 1.3, 9.0)

    # Column headers
    _t(s, "RANK", 0.5,  1.4, 0.55, 0.3, size=8, bold=True, color=GRAY)
    _t(s, "SONG", 1.1,  1.4, 3.3,  0.3, size=8, bold=True, color=GRAY)
    _t(s, "SCORE", 4.5, 1.4, 1.1,  0.3, size=8, bold=True, color=GRAY)
    _t(s, "VIRAL POTENTIAL", 5.7, 1.4, 3.7, 0.3, size=8, bold=True, color=GRAY)
    _rule(s, 0.5, 1.72, 9.0, BLACK, 0.010)

    for i, (song, prob, *_) in enumerate(TATIANA_DATA):
        y = 1.82 + i * 0.44

        if i % 2 == 0:
            _box(s, 0.48, y - 0.03, 9.1, 0.42, LGRAY)

        # Verdict color
        if prob >= 0.55:
            vc, label = GREEN,  "VIRAL"
        elif prob >= 0.40:
            vc, label = RGBColor(0xE6, 0x8A, 0x00), "BORDERLINE"
        else:
            vc, label = GRAY,   "not viral"

        _t(s, str(i + 1), 0.5, y + 0.03, 0.55, 0.35,
           size=10, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
        _t(s, song, 1.1, y + 0.03, 3.3, 0.35, size=10, bold=(prob >= 0.55))

        # Score bar
        bar_w = prob * 3.5
        _box(s, 4.5, y + 0.1, 3.5, 0.22, MGRAY)
        _box(s, 4.5, y + 0.1, max(bar_w, 0.04), 0.22, vc)
        _t(s, f"{prob:.0%}", 8.1, y + 0.03, 0.7, 0.35,
           size=10, bold=True, color=vc)
        _t(s, label, 8.85, y + 0.03, 0.9, 0.35, size=8, color=vc)

    _footer(s)


def slide_tatiana_hooks(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "Key viral hooks — what makes them stick", 0.5, 0.35, 9.0, 0.6,
       size=24, bold=True)
    _t(s, "Top 6 songs by viral score  ·  line that anchors each hook",
       0.5, 0.95, 9.0, 0.32, size=10, color=GRAY)
    _rule(s, 0.5, 1.3, 9.0)

    top6 = TATIANA_DATA[:6]

    for i, (song, prob, viral_line, why, runner_up) in enumerate(top6):
        col = i % 2
        row = i // 2
        x = 0.5 + col * 4.75
        y = 1.45 + row * 1.95

        if prob >= 0.55:
            bar_c = GREEN
        elif prob >= 0.40:
            bar_c = RGBColor(0xE6, 0x8A, 0x00)
        else:
            bar_c = GRAY

        # Card background
        _box(s, x, y, 4.4, 1.75, LGRAY)
        # Top accent bar
        _box(s, x, y, 4.4, 0.045, bar_c)

        _t(s, song, x + 0.15, y + 0.07, 3.1, 0.38,
           size=11, bold=True, color=BLACK)
        _t(s, f"{prob:.0%}", x + 3.3, y + 0.07, 0.9, 0.38,
           size=14, bold=True, color=bar_c, align=PP_ALIGN.RIGHT)

        # Viral line (quoted)
        line_display = f'"{viral_line}"' if len(viral_line) <= 44 else f'"{viral_line[:42]}..."'
        _t(s, line_display, x + 0.15, y + 0.52, 4.1, 0.55,
           size=10, bold=True, color=ACCENT)

        # Why
        _t(s, why, x + 0.15, y + 1.12, 4.1, 0.42,
           size=8, color=GRAY, italic=True)

    _footer(s)


def slide_next_steps(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)

    _t(s, "Next steps", 0.5, 0.35, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0)

    steps = [
        ("Re-evaluate",
         "Run model_tester.py with the new models to get updated metrics "
         "across all 4,612 songs and compare directly to the old baseline."),
        ("Add TikTok-specific features",
         "Current features capture lyric quality, not TikTok patterns. "
         "Add: short-line density, dance/movement vocabulary, challenge language, "
         "trending slang detection."),
        ("Retrain with stronger negatives",
         "Billboard flops (526 songs) are good hard negatives. "
         "Add more by scraping songs that peaked low on charts "
         "and have confirmed zero TikTok presence."),
        ("Calibrate probabilities",
         "Apply Platt scaling so 0.7 really means 70% likely viral, "
         "not just 'ranked in the top 30%'. Helps set better thresholds."),
    ]

    for i, (title, body) in enumerate(steps):
        y = 1.2 + i * 1.35
        _box(s, 0.5, y, 0.06, 0.85, ACCENT)
        _t(s, title, 0.75, y, 8.6, 0.42, size=13, bold=True)
        _t(s, body,  0.75, y + 0.45, 8.6, 0.7, size=10, color=GRAY)

    _footer(s)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_what_we_did(prs)
    slide_data(prs)
    slide_old_results(prs)
    slide_new_results(prs)
    slide_model1(prs)
    slide_tatiana_rankings(prs)
    slide_tatiana_hooks(prs)
    slide_next_steps(prs)

    prs.save(OUT_FILE)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    build()

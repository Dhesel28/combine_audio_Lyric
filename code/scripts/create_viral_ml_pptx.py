"""
Generate PowerPoint + PDF for the ML Viral Lyric Section Predictor project.
Updated: v2 — 38 features, 10 models, LogisticReg_L1 best model (71.7% P@1)
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import subprocess, os

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Palette ────────────────────────────────────────────────────────────────────
PURPLE   = RGBColor(0x6C, 0x5C, 0xE7)
VIOLET   = RGBColor(0xA2, 0x9B, 0xFA)
TEAL     = RGBColor(0x00, 0xB8, 0x94)
CORAL    = RGBColor(0xFF, 0x6B, 0x6B)
GOLD     = RGBColor(0xFD, 0xCB, 0x6E)
DARK     = RGBColor(0x1E, 0x1E, 0x2E)
DARK2    = RGBColor(0x2D, 0x2D, 0x44)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
LGRAY    = RGBColor(0xF0, 0xF0, 0xF8)
MID      = RGBColor(0xAA, 0xA8, 0xC8)
GREEN    = RGBColor(0x55, 0xEF, 0xC4)

W = prs.slide_width
H = prs.slide_height


# ── Helpers ────────────────────────────────────────────────────────────────────

def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def rect(slide, l, t, w, h, color, radius=False):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        l, t, w, h
    )
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp

def txt(slide, text, l, t, w, h, size=18, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, wrap=True, italic=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb

def bullet_box(slide, items, l, t, w, h, size=14, color=WHITE, dot_color=None, spacing=0.38):
    """Draw a vertical list of bullet items."""
    dot_color = dot_color or TEAL
    line_h = Inches(spacing)
    for i, item in enumerate(items):
        y = t + i * line_h
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, l, y + Inches(0.09), Inches(0.12), Inches(0.12))
        d.fill.solid(); d.fill.fore_color.rgb = dot_color
        d.line.fill.background()
        txt(slide, item, l + Inches(0.2), y, w - Inches(0.2), line_h,
            size=size, color=color)

def bar(slide, l, t, full_w, h, val, max_val, fill_color, bg_color=DARK2, label="", pct_label=""):
    """Horizontal progress bar."""
    rect(slide, l, t, full_w, h, bg_color, radius=True)
    filled = int(full_w * val / max_val)
    if filled > 0:
        rect(slide, l, t, filled, h, fill_color, radius=True)
    if label:
        txt(slide, label, l - Inches(2.2), t - Inches(0.02), Inches(2.1), h + Inches(0.05),
            size=13, color=WHITE, align=PP_ALIGN.RIGHT)
    if pct_label:
        txt(slide, pct_label, l + full_w + Inches(0.1), t - Inches(0.02),
            Inches(0.9), h + Inches(0.05), size=13, bold=True, color=fill_color)

def accent_line(slide, l, t, w, color=TEAL):
    r = rect(slide, l, t, w, Inches(0.04), color)
    return r


# ── SLIDE 1 — Title ────────────────────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)

rect(s, Inches(-0.5), Inches(-0.5), Inches(7), H + Inches(1), DARK2)

for cx, cy, r_in, col in [
    (Inches(12.5), Inches(0.3),  Inches(1.4), PURPLE),
    (Inches(11.2), Inches(6.8),  Inches(1.0), TEAL),
    (Inches(0.2),  Inches(6.5),  Inches(0.7), CORAL),
]:
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, cx - r_in/2, cy - r_in/2, r_in, r_in)
    c.fill.solid(); c.fill.fore_color.rgb = col
    c.line.fill.background()

txt(s, "ML Viral Lyric Section Predictor",
    Inches(0.6), Inches(1.5), Inches(9), Inches(1.4),
    size=40, bold=True, color=WHITE)

accent_line(s, Inches(0.6), Inches(3.0), Inches(5), TEAL)

txt(s, "Which part of a song goes viral on TikTok?",
    Inches(0.6), Inches(3.15), Inches(9), Inches(0.7),
    size=22, color=VIOLET)

txt(s, "Machine Learning  ·  180 Songs  ·  38 Features  ·  10 Models  ·  Leave-One-Song-Out CV",
    Inches(0.6), Inches(3.85), Inches(11), Inches(0.5),
    size=15, color=MID, italic=True)

for i, (val, label, col) in enumerate([
    ("180",   "Songs",        PURPLE),
    ("979",   "Sections",     TEAL),
    ("38",    "Features",     CORAL),
    ("71.7%", "P@1 Accuracy", GOLD),
]):
    lx = Inches(0.6 + i * 2.2)
    rect(s, lx, Inches(5.0), Inches(1.9), Inches(1.1), col, radius=True)
    txt(s, val,   lx, Inches(5.05), Inches(1.9), Inches(0.55),
        size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s, label, lx, Inches(5.6),  Inches(1.9), Inches(0.4),
        size=12, color=WHITE, align=PP_ALIGN.CENTER)

txt(s, "SUPA Spring 2026", Inches(9.5), Inches(6.8), Inches(3.5), Inches(0.5),
    size=12, color=MID, align=PP_ALIGN.RIGHT)


# ── SLIDE 2 — Project Overview ─────────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Project Overview", Inches(0.5), Inches(0.2), Inches(10), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), TEAL)

rect(s, Inches(0.4), Inches(1.3), Inches(5.8), Inches(5.7), DARK2, radius=True)
txt(s, "The Problem", Inches(0.6), Inches(1.45), Inches(5.4), Inches(0.5),
    size=18, bold=True, color=TEAL)
bullet_box(s, [
    "Songs go viral on TikTok via specific lyric sections",
    "Creators use 10–30 sec clips of the 'hook'",
    "No tool predicts WHICH section will trend",
    "Manual scoring formulas miss complex patterns",
    "Goal: train ML to learn viral patterns from data",
], Inches(0.55), Inches(2.05), Inches(5.5), Inches(4.5), size=13)

rect(s, Inches(6.6), Inches(1.3), Inches(6.3), Inches(5.7), DARK2, radius=True)
txt(s, "Our Solution", Inches(6.8), Inches(1.45), Inches(5.9), Inches(0.5),
    size=18, bold=True, color=CORAL)
bullet_box(s, [
    "Parse any song's lyrics into sections via Genius API",
    "Extract 38 engineered features per section",
    "Train 10 ML models with leave-one-song-out CV",
    "Best: LogisticReg_L1 — 71.7% P@1 (L1 regularisation)",
    "Live Streamlit dashboard for any song",
    "AUC-ROC of 0.897 — excellent class separation",
], Inches(6.75), Inches(2.05), Inches(6.0), Inches(4.5), size=13,
    dot_color=CORAL)


# ── SLIDE 3 — Data Pipeline ────────────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Data Pipeline  —  2 Sources → 979 Labeled Sections",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), PURPLE)

rect(s, Inches(0.4), Inches(1.3), Inches(5.8), Inches(2.7), DARK2, radius=True)
txt(s, "Source A — Valentine Songs Dataset",
    Inches(0.6), Inches(1.4), Inches(5.6), Inches(0.45),
    size=15, bold=True, color=PURPLE)
bullet_box(s, [
    "173 songs · valentine_songs_full_dataset.csv",
    "Full lyrics column (Genius-scraped)",
    "Label = section with highest text_virality_score",
    "Yields 906 labeled sections",
], Inches(0.55), Inches(1.9), Inches(5.6), Inches(2.0), size=12, dot_color=PURPLE)

rect(s, Inches(0.4), Inches(4.2), Inches(5.8), Inches(2.7), DARK2, radius=True)
txt(s, "Source B — Cross-Platform Annotated Songs",
    Inches(0.6), Inches(4.3), Inches(5.6), Inches(0.45),
    size=15, bold=True, color=TEAL)
bullet_box(s, [
    "10 songs · cross_platform_analysis.csv",
    "Genius annotation scores + YouTube Shorts data",
    "Label = section with highest combined_viral_score",
    "Yields 73 labeled sections",
], Inches(0.55), Inches(4.8), Inches(5.6), Inches(2.0), size=12, dot_color=TEAL)

for ay in [Inches(2.35), Inches(5.5)]:
    arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
        Inches(6.4), ay, Inches(0.6), Inches(0.35))
    arrow.fill.solid(); arrow.fill.fore_color.rgb = GOLD
    arrow.line.fill.background()

rect(s, Inches(7.2), Inches(3.0), Inches(5.7), Inches(3.8), DARK2, radius=True)
txt(s, "Combined Training Set",
    Inches(7.4), Inches(3.1), Inches(5.3), Inches(0.45),
    size=15, bold=True, color=GOLD)
bullet_box(s, [
    "979  total sections",
    "180  unique songs",
    "185  positive labels (one viral section per song)",
    "18.9% positive rate — class imbalance handled",
    "GroupKFold keeps song sections together (no leakage)",
], Inches(7.35), Inches(3.65), Inches(5.4), Inches(3.0), size=13, dot_color=GOLD)


# ── SLIDE 4 — Feature Engineering ─────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Feature Engineering  —  38 Features per Section  (+6 New)",
    Inches(0.5), Inches(0.2), Inches(12.5), Inches(0.75),
    size=26, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), CORAL)

groups = [
    ("Text Scores",      4,  CORAL,  ["repeat_score", "short_line_score", "rhyme_score", "hook_keyword_score"]),
    ("Position",         4,  PURPLE, ["section_index", "normalized_position", "is_first", "is_last"]),
    ("Section Type",     7,  TEAL,   ["chorus/hook · verse · bridge · pre-chorus · intro · outro · other"]),
    ("VADER Sentiment",  4,  GOLD,   ["compound", "positive", "negative", "neutral"]),
    ("TextBlob",         2,  VIOLET, ["polarity", "subjectivity"]),
    ("Linguistic",       4,  GREEN,  ["pronoun_density", "question_count", "exclamation_count", "cap_word_ratio"]),
    ("Singability",      2,  CORAL,  ["vowel_density", "avg_syllables_per_word"]),
    ("Complexity",       2,  MID,    ["unique_word_ratio", "avg_word_length"]),
    ("Structural",       2,  PURPLE, ["has_repetition_pattern", "call_response_pattern"]),
    ("Readability",      1,  TEAL,   ["flesch_reading_ease"]),
    ("NEW: Structure",   2,  GREEN,  ["word_count", "line_count"]),
    ("NEW: Hook & Emotion", 4, GOLD, ["emotional_intensity", "ends_strong", "starts_strong", "hook_word_variety"]),
]

cols = 2
rows_per_col = 6
box_w, box_h = Inches(5.9), Inches(1.02)
start_x = [Inches(0.35), Inches(6.55)]
start_y = Inches(1.25)

for idx, (grp, n, col, features) in enumerate(groups):
    cx = idx // rows_per_col
    cy = idx % rows_per_col
    lx = start_x[cx]
    ty = start_y + cy * (box_h + Inches(0.04))

    bg_col = DARK2
    if "NEW" in grp:
        bg_col = RGBColor(0x1A, 0x2A, 0x1E)  # slightly green tint for new
    rect(s, lx, ty, box_w, box_h, bg_col, radius=True)
    rect(s, lx, ty, Inches(0.18), box_h, col, radius=True)

    label = f"{grp}  ({n})"
    if "NEW" in grp:
        label = f"★ {grp}  ({n})"
    txt(s, label, lx + Inches(0.28), ty + Inches(0.05),
        Inches(2.8), Inches(0.35), size=12, bold=True, color=col)
    feature_str = "  ·  ".join(features) if isinstance(features, list) else features[0]
    txt(s, feature_str, lx + Inches(0.28), ty + Inches(0.45),
        box_w - Inches(0.35), Inches(0.48), size=9.5, color=MID)


# ── SLIDE 5 — Top Features by Importance ──────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Top Feature Importances  (Random Forest, 38 Features)",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), GOLD)

top_features = [
    ("hook_keyword_score",     0.0879, False),
    ("unique_word_ratio",      0.0843, False),
    ("repeat_score",           0.0788, False),
    ("has_repetition_pattern", 0.0531, False),
    ("line_count",             0.0489, True),   # NEW
    ("emotional_intensity",    0.0472, True),   # NEW
    ("rhyme_score",            0.0441, False),
    ("short_line_score",       0.0424, False),
    ("vowel_density",          0.0383, False),
    ("word_count",             0.0365, True),   # NEW
    ("flesch_reading_ease",    0.0365, False),
    ("avg_word_length",        0.0317, False),
]

bar_colors = [GOLD, GOLD, CORAL, PURPLE, GREEN,
              GREEN, TEAL, CORAL, TEAL, GREEN, TEAL, MID]
max_val = 0.095
bar_l = Inches(3.2)
bar_full_w = Inches(8.0)
bar_h = Inches(0.31)
start_bar_y = Inches(1.25)
gap = Inches(0.43)

for i, ((feat, val, is_new), col) in enumerate(zip(top_features, bar_colors)):
    ty = start_bar_y + i * gap
    bar(s, bar_l, ty, bar_full_w, bar_h, val, max_val,
        fill_color=col, label=feat,
        pct_label=f"{val*100:.1f}%")
    if is_new:
        txt(s, "NEW", bar_l + bar_full_w + Inches(1.0), ty - Inches(0.02),
            Inches(0.6), Inches(0.33), size=9, bold=True, color=GREEN)

txt(s, "3 new features enter the top-10: line_count, emotional_intensity, word_count  —  "
       "confirming section length and absolute emotional strength are key viral signals.",
    Inches(0.5), Inches(6.55), Inches(12), Inches(0.7),
    size=12, color=MID, italic=True)


# ── SLIDE 6 — Models & Validation ─────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Models & Validation Strategy  —  10 Models",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), VIOLET)

rect(s, Inches(0.4), Inches(1.25), Inches(4.8), Inches(5.8), DARK2, radius=True)
txt(s, "Leave-One-Song-Out CV",
    Inches(0.6), Inches(1.35), Inches(4.4), Inches(0.45),
    size=16, bold=True, color=VIOLET)
bullet_box(s, [
    "GroupKFold(n_splits=180)",
    "All sections from one song → test set",
    "Remaining 179 songs → training set",
    "No data leakage: song sections never split",
    "180 evaluation folds total",
    "Mirrors real-world: predict a brand-new song",
], Inches(0.55), Inches(1.9), Inches(4.5), Inches(4.8), size=13, dot_color=VIOLET)

rect(s, Inches(5.5), Inches(1.25), Inches(7.4), Inches(5.8), DARK2, radius=True)
txt(s, "10 Models Evaluated",
    Inches(5.7), Inches(1.35), Inches(7.0), Inches(0.45),
    size=16, bold=True, color=TEAL)

model_info = [
    ("LogisticReg_L1  ★ BEST",         "L1 penalty — auto feature selection",       TEAL),
    ("Logistic Regression",             "L2 baseline — linear boundary",              VIOLET),
    ("Stacking (XGB+LGBM+ET → LR)",    "ensemble meta-learner, 5-fold inner CV",     GOLD),
    ("Voting (LR + L1 + HGB)",         "soft-vote ensemble of top linear models",    GREEN),
    ("XGBoost (300 est.)",              "scale_pos_weight + gamma regularisation",    GOLD),
    ("LightGBM (300 est.)",             "num_leaves=20 + reg_alpha/lambda",           GREEN),
    ("HGB (HistGradientBoosting)",      "sklearn native GBDT — small data friendly",  TEAL),
    ("Random Forest (500 trees)",       "balanced class_weight, sqrt features",       PURPLE),
    ("Extra Trees (500 trees)",         "more randomisation than RF",                 VIOLET),
    ("MLP 128→64→32",                   "deep neural net + early stopping",           CORAL),
]
for i, (name, note, col) in enumerate(model_info):
    ty = Inches(1.88) + i * Inches(0.49)
    rect(s, Inches(5.6), ty, Inches(7.1), Inches(0.42), DARK, radius=True)
    p2 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(5.65), ty + Inches(0.06), Inches(0.15), Inches(0.30))
    p2.fill.solid(); p2.fill.fore_color.rgb = col
    p2.line.fill.background()
    txt(s, name, Inches(5.92), ty + Inches(0.03), Inches(4.7), Inches(0.22),
        size=11, bold=("BEST" in name), color=WHITE if "BEST" not in name else TEAL)
    txt(s, note, Inches(5.92), ty + Inches(0.24), Inches(6.5), Inches(0.18),
        size=9, color=MID, italic=True)


# ── SLIDE 7 — Model Results ────────────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Model Results  —  Leave-One-Song-Out CV  (180 songs)",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), CORAL)

results = [
    ("LogisticReg_L1",       0.7167, 0.8367, 0.8966, TEAL,   True),
    ("LogisticReg",          0.7111, 0.8332, 0.8942, VIOLET, False),
    ("Stacking",             0.6944, 0.8200, 0.8821, GOLD,   False),
    ("Voting(LR+L1+HGB)",   0.6889, 0.8237, 0.8863, GREEN,  False),
    ("XGBoost",              0.6556, 0.8032, 0.8702, GOLD,   False),
    ("LightGBM",             0.6556, 0.7994, 0.8686, GREEN,  False),
    ("HGB",                  0.6556, 0.8020, 0.8642, TEAL,   False),
    ("ExtraTrees",           0.6389, 0.7883, 0.8568, VIOLET, False),
    ("RandomForest",         0.6333, 0.7903, 0.8665, PURPLE, False),
    ("MLP(128-64-32)",       0.5722, 0.7313, 0.7907, CORAL,  False),
]

for lx, label in [(Inches(3.4), "P@1"), (Inches(7.0), "MRR"), (Inches(10.3), "AUC-ROC")]:
    txt(s, label, lx, Inches(1.2), Inches(1.5), Inches(0.4),
        size=14, bold=True, color=MID, align=PP_ALIGN.CENTER)
txt(s, "higher is better →", Inches(10.5), Inches(1.25), Inches(2.5), Inches(0.35),
    size=10, color=MID, italic=True)

row_h = Inches(0.575)
bar_w = Inches(2.8)
for i, (name, p1, mrr, auc, col, best) in enumerate(results):
    ty = Inches(1.62) + i * row_h
    bg_col = DARK2 if not best else RGBColor(0x1A, 0x2A, 0x1A)
    rect(s, Inches(0.3), ty, Inches(12.7), row_h - Inches(0.04), bg_col, radius=True)

    if best:
        rect(s, Inches(0.3), ty, Inches(0.22), row_h - Inches(0.04), TEAL, radius=True)
        txt(s, "★ BEST", Inches(0.56), ty + Inches(0.19), Inches(0.9), Inches(0.26),
            size=8, bold=True, color=TEAL)

    txt(s, name, Inches(1.55), ty + Inches(0.13), Inches(1.7), Inches(0.34),
        size=12, bold=best, color=WHITE if not best else TEAL)

    bx = Inches(3.2)
    rect(s, bx, ty + Inches(0.13), bar_w, Inches(0.27), DARK, radius=True)
    rect(s, bx, ty + Inches(0.13), int(bar_w * p1), Inches(0.27), col, radius=True)
    txt(s, f"{p1:.1%}", bx + bar_w + Inches(0.1), ty + Inches(0.1),
        Inches(0.7), Inches(0.3), size=11, bold=best, color=col)

    bx2 = Inches(6.8)
    rect(s, bx2, ty + Inches(0.13), bar_w, Inches(0.27), DARK, radius=True)
    rect(s, bx2, ty + Inches(0.13), int(bar_w * mrr), Inches(0.27), col, radius=True)
    txt(s, f"{mrr:.3f}", bx2 + bar_w + Inches(0.1), ty + Inches(0.1),
        Inches(0.8), Inches(0.3), size=11, bold=best, color=col)

    bx3 = Inches(10.1)
    rect(s, bx3, ty + Inches(0.13), Inches(2.0), Inches(0.27), DARK, radius=True)
    rect(s, bx3, ty + Inches(0.13), int(Inches(2.0) * auc), Inches(0.27), col, radius=True)
    txt(s, f"{auc:.3f}", bx3 + Inches(2.1), ty + Inches(0.1),
        Inches(0.75), Inches(0.3), size=11, bold=best, color=col)


# ── SLIDE 8 — Metrics Explained ───────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Understanding the Metrics",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), GOLD)

metrics = [
    ("P@1\nPrecision at Rank 1", TEAL, GOLD,
     "Did the model pick the correct viral section as #1?",
     "Score = 1 if top prediction is the true viral section, else 0.\n"
     "Our best: 71.7% → right ~5 out of 7 songs.",
     "Random baseline ≈ 18.9%  →  3.8× better than random."),
    ("MRR\nMean Reciprocal Rank", PURPLE, VIOLET,
     "How high up the ranked list is the correct section?",
     "MRR = 1/rank of first correct prediction.\n"
     "Our best: 0.837 → correct section usually #1 or #2.",
     "High MRR confirms model confidently ranks viral sections near the top."),
    ("AUC-ROC\nArea Under ROC Curve", CORAL, GOLD,
     "Can the model separate viral from non-viral sections?",
     "AUC=1.0 perfect · AUC=0.5 random.\n"
     "Our best: 0.897 → excellent discrimination.",
     "Robust to class imbalance · meaningful across all 180 songs."),
]

for i, (title, col, acc, q, body, insight) in enumerate(metrics):
    lx = Inches(0.35 + i * 4.33)
    rect(s, lx, Inches(1.25), Inches(4.15), Inches(5.9), DARK2, radius=True)
    rect(s, lx, Inches(1.25), Inches(4.15), Inches(0.65), col, radius=True)
    txt(s, title, lx + Inches(0.15), Inches(1.28), Inches(3.9), Inches(0.6),
        size=14, bold=True, color=WHITE)
    txt(s, q, lx + Inches(0.15), Inches(2.0), Inches(3.9), Inches(0.6),
        size=12, bold=True, color=acc)
    txt(s, body, lx + Inches(0.15), Inches(2.65), Inches(3.9), Inches(1.5),
        size=11, color=WHITE)
    rect(s, lx + Inches(0.1), Inches(4.3), Inches(3.95), Inches(0.04), col)
    txt(s, insight, lx + Inches(0.15), Inches(4.4), Inches(3.9), Inches(1.6),
        size=11, color=MID, italic=True)


# ── SLIDE 9 — Why LogisticReg_L1 Wins ─────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Why Does LogisticReg_L1 Win?",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), TEAL)

rect(s, Inches(0.4), Inches(1.25), Inches(6.0), Inches(5.8), DARK2, radius=True)
txt(s, "The Bias-Variance Tradeoff", Inches(0.6), Inches(1.35), Inches(5.6), Inches(0.45),
    size=17, bold=True, color=TEAL)
bullet_box(s, [
    "180 songs is a small dataset for complex models",
    "Tree ensembles overfit to training song patterns",
    "Neural nets need far more data to generalise",
    "LR + StandardScaler finds a robust linear boundary",
    "L1 (Lasso) penalty zeroes out noisy features — acts as",
    "  built-in feature selector across 38 dimensions",
    "class_weight='balanced' corrects 18.9% imbalance cleanly",
], Inches(0.55), Inches(1.9), Inches(5.7), Inches(4.5), size=13)

rect(s, Inches(6.7), Inches(1.25), Inches(6.2), Inches(2.7), DARK2, radius=True)
txt(s, "L2 LogisticReg close behind at 71.1%", Inches(6.9), Inches(1.35),
    Inches(5.8), Inches(0.45), size=15, bold=True, color=GOLD)
bullet_box(s, [
    "Only 0.6% gap confirms L1 vs L2 is a minor difference",
    "Both beat all tree / ensemble models by wide margin",
    "Linear signal dominates — features are well-engineered",
    "Stacking (69.4%) and Voting (68.9%) trail by ~2.5%",
], Inches(6.85), Inches(1.9), Inches(5.9), Inches(1.9), size=13, dot_color=GOLD)

rect(s, Inches(6.7), Inches(4.1), Inches(6.2), Inches(2.95), DARK2, radius=True)
txt(s, "Next steps to improve further", Inches(6.9), Inches(4.2),
    Inches(5.8), Inches(0.45), size=17, bold=True, color=CORAL)
bullet_box(s, [
    "More data — annotate 500+ songs with real TikTok views",
    "Audio features: BPM, energy, danceability (Spotify API)",
    "BERT/sentence embeddings for semantic meaning",
    "Fine-tune on platform-specific virality signals",
], Inches(6.85), Inches(4.75), Inches(5.9), Inches(2.0), size=13, dot_color=CORAL)


# ── SLIDE 10 — The Streamlit Dashboard ────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "The Streamlit Dashboard  —  5 Tabs",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), GREEN)

tabs = [
    ("TikTok Trending",    CORAL,  "Live TikTok trending songs via RapidAPI. Bar chart of plays & likes."),
    ("Lyric Sections",     PURPLE, "Viral score breakdown by section for saved songs. Progress bars per section."),
    ("Cross-Platform",     TEAL,   "TikTok vs YouTube Shorts scatter. Genius annotations vs YT matches."),
    ("Analyze Any Song",   GOLD,   "Live Genius fetch. Radar chart + feature breakdown + lyrics preview."),
    ("Model Prediction",   GREEN,  "ML viral probability bars + feature importances + predicted lyrics."),
]
for i, (name, col, desc) in enumerate(tabs):
    lx = Inches(0.35 + i * 2.57)
    rect(s, lx, Inches(1.25), Inches(2.45), Inches(0.9), col, radius=True)
    txt(s, name, lx, Inches(1.32), Inches(2.45), Inches(0.55),
        size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(s, lx, Inches(2.28), Inches(2.45), Inches(2.35), DARK2, radius=True)
    txt(s, desc, lx + Inches(0.1), Inches(2.38), Inches(2.25), Inches(2.15),
        size=11, color=WHITE)

rect(s, Inches(0.4), Inches(4.78), Inches(12.5), Inches(2.4), DARK2, radius=True)
txt(s, "Tab 5 — Model Prediction Detail:",
    Inches(0.6), Inches(4.88), Inches(4.0), Inches(0.4),
    size=15, bold=True, color=GREEN)
bullet_box(s, [
    "Enter any song → Genius API fetches & parses sections automatically",
    "38 features extracted per section → LogisticReg_L1 model scores each",
    "Bar chart: viral probability per section  |  Top-15 feature importances (RF)",
    "Scatter: ML probability vs Genius annotation score  |  Predicted lyrics shown",
], Inches(0.55), Inches(5.33), Inches(12.0), Inches(1.6),
   size=12, dot_color=GREEN, spacing=0.30)


# ── SLIDE 11 — Example: Espresso ──────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, 'Example Prediction — "Espresso" by Sabrina Carpenter',
    Inches(0.5), Inches(0.2), Inches(12.5), Inches(0.75),
    size=26, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), CORAL)

rect(s, Inches(0.4), Inches(1.25), Inches(5.6), Inches(5.8), DARK2, radius=True)
txt(s, "Model scores each section",
    Inches(0.6), Inches(1.35), Inches(5.2), Inches(0.45),
    size=16, bold=True, color=CORAL)
sections_eg = [
    ("[Intro]",          0.12, MID),
    ("[Verse 1]",        0.28, TEAL),
    ("[Pre-Chorus]",     0.51, VIOLET),
    ("[Chorus]  <- #1",  0.88, CORAL),
    ("[Post-Chorus]",    0.62, GOLD),
    ("[Verse 2]",        0.30, TEAL),
    ("[Bridge]",         0.43, PURPLE),
    ("[Outro]",          0.19, MID),
]
bar_l2 = Inches(3.2); bar_fw2 = Inches(2.3)
for i, (lbl, prob, col) in enumerate(sections_eg):
    ty = Inches(1.9) + i * Inches(0.6)
    txt(s, lbl, Inches(0.55), ty + Inches(0.04), Inches(2.5), Inches(0.3),
        size=11, bold=(prob > 0.7), color=WHITE if prob < 0.7 else CORAL)
    rect(s, bar_l2, ty + Inches(0.07), bar_fw2, Inches(0.24), DARK, radius=True)
    rect(s, bar_l2, ty + Inches(0.07), int(bar_fw2 * prob), Inches(0.24), col, radius=True)
    txt(s, f"{prob:.0%}", bar_l2 + bar_fw2 + Inches(0.1), ty + Inches(0.04),
        Inches(0.5), Inches(0.28), size=10, bold=(prob > 0.7), color=col)

rect(s, Inches(6.3), Inches(1.25), Inches(6.6), Inches(5.8), DARK2, radius=True)
txt(s, "Why [Chorus] scores highest",
    Inches(6.5), Inches(1.35), Inches(6.2), Inches(0.45),
    size=16, bold=True, color=TEAL)
bullet_box(s, [
    "hook_keyword_score HIGH: 'sweet', 'baby', 'feel', 'yeah'",
    "hook_word_variety HIGH: 5 unique hook words",
    "repeat_score HIGH: chorus lines repeat 4x across song",
    "stype_chorus_hook = 1 (strongest categorical signal)",
    "vader_compound: positive upbeat emotional sentiment",
    "emotional_intensity HIGH: strong absolute sentiment",
    "short_line_score: punchy 6-8 word lines",
    "has_repetition_pattern = 1 — memorable structure",
    "flesch_reading_ease: very easy — sing-along friendly",
], Inches(6.45), Inches(1.9), Inches(6.25), Inches(4.2), size=12, dot_color=TEAL)

txt(s, '"And now he\'s thinkin\' \'bout me every night, oh\n'
       'Is it that sweet? I guess so..."',
    Inches(6.5), Inches(6.1), Inches(6.2), Inches(0.9),
    size=12, color=MID, italic=True)


# ── SLIDE 12 — Key Takeaways ───────────────────────────────────────────────────
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
rect(s, 0, 0, W, Inches(1.1), DARK2)
txt(s, "Key Takeaways",
    Inches(0.5), Inches(0.2), Inches(12), Inches(0.75),
    size=28, bold=True, color=WHITE)
accent_line(s, Inches(0.5), Inches(1.1), Inches(12.3), PURPLE)

takeaways = [
    (TEAL,   "71.7% P@1",
     "LogisticReg_L1 correctly identifies the viral section 71.7%\nof the time — 3.8× better than random (18.9%)"),
    (CORAL,  "38 Features Work",
     "6 new features added: line_count, emotional_intensity,\nword_count, ends_strong, starts_strong, hook_word_variety"),
    (PURPLE, "Linear > Complex",
     "With 180 songs, L1 Logistic Regression beats XGBoost,\nLightGBM, Neural Nets — features matter more than model"),
    (GOLD,   "AUC 0.897",
     "Near-perfect discrimination across all songs —\nhook keywords + repetition capture real viral structure"),
    (GREEN,  "Live & Deployable",
     "Streamlit dashboard runs real-time inference on any song\nvia Genius API — usable as songwriter / A&R tool"),
    (VIOLET, "Next: Better Labels",
     "Real TikTok view counts + Spotify audio features\ncould push P@1 above 80% with the same architecture"),
]

for i, (col, title, body) in enumerate(takeaways):
    cx = i % 3; cy = i // 3
    lx = Inches(0.35 + cx * 4.33)
    ty = Inches(1.25 + cy * 2.8)
    rect(s, lx, ty, Inches(4.15), Inches(2.55), DARK2, radius=True)
    rect(s, lx, ty, Inches(4.15), Inches(0.5), col, radius=True)
    txt(s, title, lx + Inches(0.15), ty + Inches(0.08),
        Inches(3.9), Inches(0.38), size=16, bold=True, color=WHITE)
    txt(s, body, lx + Inches(0.15), ty + Inches(0.62),
        Inches(3.9), Inches(1.8), size=12, color=WHITE)


# ── Save PPTX ─────────────────────────────────────────────────────────────────
out_pptx = "/Users/dhekha/Desktop/SUPA_Spring/ML_Viral_Lyric_Predictor.pptx"
prs.save(out_pptx)
print(f"Saved PPTX -> {out_pptx}")

# ── Convert to PDF ─────────────────────────────────────────────────────────────
import shutil
lo = shutil.which("libreoffice") or shutil.which("soffice")
if lo:
    subprocess.run([
        lo, "--headless", "--convert-to", "pdf",
        "--outdir", "/Users/dhekha/Desktop/SUPA_Spring/", out_pptx
    ], check=True)
    print("Converted to PDF via LibreOffice")
else:
    result = subprocess.run([
        "osascript", "-e",
        f'''tell application "Microsoft PowerPoint"
            open POSIX file "{out_pptx}"
            save active presentation in POSIX file "{out_pptx.replace('.pptx','.pdf')}" as save as PDF
            close active presentation
        end tell'''
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print("Converted via PowerPoint AppleScript")
    else:
        print("Note: PDF conversion requires LibreOffice or Microsoft PowerPoint.")
        print(f"PPTX is ready at: {out_pptx}")

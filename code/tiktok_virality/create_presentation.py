"""
TikTok Viral Predictor — PowerPoint Presentation Generator
===========================================================
Creates a 11-slide deck matching the dark-theme style of the project.

Run:
    cd tiktok_virality
    python create_presentation.py
"""

import subprocess, shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Palette ────────────────────────────────────────────────────────────────────
PURPLE  = RGBColor(0x6C, 0x5C, 0xE7)
VIOLET  = RGBColor(0xA2, 0x9B, 0xFA)
TEAL    = RGBColor(0x00, 0xC8, 0x96)
CORAL   = RGBColor(0xE9, 0x45, 0x60)
GOLD    = RGBColor(0xFF, 0xD1, 0x66)
DARK    = RGBColor(0x1A, 0x1A, 0x2E)
DARK2   = RGBColor(0x16, 0x21, 0x3E)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
MID     = RGBColor(0xAA, 0xA8, 0xC8)
GREEN   = RGBColor(0x00, 0xC8, 0x96)
BLUE    = RGBColor(0x4C, 0xC9, 0xF0)

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
        l, t, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp

def txt(slide, text, l, t, w, h, size=18, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.italic = italic; run.font.color.rgb = color
    return tb

def bullet_box(slide, items, l, t, w, h, size=14, color=WHITE,
               dot_color=None, spacing=0.38):
    dot_color = dot_color or TEAL
    line_h = Inches(spacing)
    for i, item in enumerate(items):
        y = t + i * line_h
        d = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, l, y + Inches(0.09), Inches(0.12), Inches(0.12))
        d.fill.solid(); d.fill.fore_color.rgb = dot_color
        d.line.fill.background()
        txt(slide, item, l + Inches(0.2), y, w - Inches(0.2), line_h,
            size=size, color=color)

def hbar(slide, l, t, full_w, h, val, max_val, fill_color,
         bg_color=None, label="", pct=""):
    bg_color = bg_color or DARK2
    rect(slide, l, t, full_w, h, bg_color, radius=True)
    filled = int(full_w * val / max_val)
    if filled > 0:
        rect(slide, l, t, filled, h, fill_color, radius=True)
    if label:
        txt(slide, label, l - Inches(2.4), t - Inches(0.02),
            Inches(2.3), h + Inches(0.05), size=12, color=WHITE, align=PP_ALIGN.RIGHT)
    if pct:
        txt(slide, pct, l + full_w + Inches(0.1), t - Inches(0.02),
            Inches(0.9), h + Inches(0.05), size=12, bold=True, color=fill_color)

def accent(slide, l, t, w, color=TEAL):
    rect(slide, l, t, w, Inches(0.04), color)

def header(slide, title, accent_color=TEAL):
    rect(slide, 0, 0, W, Inches(1.1), DARK2)
    txt(slide, title, Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.75),
        size=28, bold=True, color=WHITE)
    accent(slide, Inches(0.5), Inches(1.1), Inches(12.3), accent_color)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)

rect(s, Inches(-0.5), Inches(-0.5), Inches(7.5), H + Inches(1), DARK2)

for cx, cy, r, col in [
    (Inches(12.5), Inches(0.4),  Inches(1.5), PURPLE),
    (Inches(11.3), Inches(6.9),  Inches(1.1), TEAL),
    (Inches(0.3),  Inches(6.6),  Inches(0.8), CORAL),
]:
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, cx - r/2, cy - r/2, r, r)
    c.fill.solid(); c.fill.fore_color.rgb = col
    c.line.fill.background()

txt(s, "TikTok Viral Predictor",
    Inches(0.6), Inches(1.2), Inches(9.5), Inches(1.5),
    size=42, bold=True, color=WHITE)

accent(s, Inches(0.6), Inches(2.85), Inches(5.5), CORAL)

txt(s, "Which songs — and which sections — go viral on TikTok?",
    Inches(0.6), Inches(3.0), Inches(9.5), Inches(0.75),
    size=22, color=VIOLET)

txt(s, "Machine Learning  ·  3,560 Songs  ·  16 Features  ·  5 Models  ·  Viral Snippet Algorithm",
    Inches(0.6), Inches(3.75), Inches(11), Inches(0.5),
    size=14, color=MID, italic=True)

for i, (val, label, col) in enumerate([
    ("3,560",  "Training Songs",   PURPLE),
    ("509",    "Viral Songs",      TEAL),
    ("16",     "Audio Features",   CORAL),
    ("0.677",  "AUC-ROC",          GOLD),
]):
    lx = Inches(0.6 + i * 2.2)
    rect(s, lx, Inches(5.0), Inches(1.95), Inches(1.2), col, radius=True)
    txt(s, val,   lx, Inches(5.05), Inches(1.95), Inches(0.6),
        size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s, label, lx, Inches(5.65), Inches(1.95), Inches(0.4),
        size=11, color=WHITE, align=PP_ALIGN.CENTER)

txt(s, "SUPA Spring 2026", Inches(9.5), Inches(6.85), Inches(3.5), Inches(0.45),
    size=12, color=MID, align=PP_ALIGN.RIGHT)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Project Overview", TEAL)

rect(s, Inches(0.4), Inches(1.3), Inches(5.9), Inches(5.7), DARK2, radius=True)
txt(s, "The Problem", Inches(0.6), Inches(1.45), Inches(5.5), Inches(0.5),
    size=18, bold=True, color=CORAL)
bullet_box(s, [
    "Songs go viral on TikTok via specific 10-30 second clips",
    "Creators manually hunt for the 'hook' by ear",
    "No data-driven tool predicts which section will trend",
    "Hand-crafted scoring formulas miss complex patterns",
    "Song-level tools exist but section-level prediction is unsolved",
    "Goal: train ML on real TikTok data to learn viral patterns",
], Inches(0.55), Inches(2.05), Inches(5.65), Inches(4.8),
   size=13, dot_color=CORAL)

rect(s, Inches(6.7), Inches(1.3), Inches(6.2), Inches(5.7), DARK2, radius=True)
txt(s, "Our Solution", Inches(6.9), Inches(1.45), Inches(5.8), Inches(0.5),
    size=18, bold=True, color=TEAL)
bullet_box(s, [
    "Train song-level ML on 3,560 TikTok songs (audio features)",
    "509 Spotify-confirmed viral songs as ground-truth labels",
    "5 models with stratified 5-fold cross-validation",
    "Best model: AUC 0.677 on audio features alone",
    "Viral Snippet Algorithm ranks lyric sections by TikTok clip potential",
    "Live Streamlit dashboard — analyze any song in real time",
], Inches(6.85), Inches(2.05), Inches(5.9), Inches(4.8),
   size=13, dot_color=TEAL)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Data Pipeline  —  3 Sources  →  2,452 Labeled Songs", PURPLE)

rect(s, Inches(0.4), Inches(1.3), Inches(5.7), Inches(2.4), DARK2, radius=True)
txt(s, "Source A — TikTok Trending Tracks",
    Inches(0.6), Inches(1.4), Inches(5.3), Inches(0.45),
    size=15, bold=True, color=PURPLE)
bullet_box(s, [
    "tiktok_trending_tracks/tiktok.csv",
    "6,746 raw rows  →  3,560 unique tracks (deduped)",
    "11 Spotify audio features + popularity score (0-100)",
], Inches(0.55), Inches(1.92), Inches(5.5), Inches(1.5), size=12, dot_color=PURPLE)

rect(s, Inches(0.4), Inches(3.9), Inches(5.7), Inches(2.0), DARK2, radius=True)
txt(s, "Source B — TikTok Popular 2020",
    Inches(0.6), Inches(4.0), Inches(5.3), Inches(0.45),
    size=15, bold=True, color=BLUE)
bullet_box(s, [
    "tiktok_popular_2020/TikTok_songs_2020.csv",
    "292 songs  ·  same 11 audio features + track_pop",
], Inches(0.55), Inches(4.55), Inches(5.5), Inches(1.2), size=12, dot_color=BLUE)

rect(s, Inches(0.4), Inches(6.05), Inches(5.7), Inches(1.2), DARK2, radius=True)
txt(s, "Source C — Spotify Viral Playlists (ground truth)",
    Inches(0.6), Inches(6.13), Inches(5.3), Inches(0.4),
    size=15, bold=True, color=TEAL)
bullet_box(s, [
    "509 confirmed viral tracks (2024 / 2025 / 2026 playlists)",
], Inches(0.55), Inches(6.6), Inches(5.5), Inches(0.45), size=12, dot_color=TEAL)

for ay in [Inches(2.5), Inches(4.9), Inches(6.6)]:
    arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
        Inches(6.25), ay, Inches(0.55), Inches(0.3))
    arr.fill.solid(); arr.fill.fore_color.rgb = GOLD
    arr.line.fill.background()

rect(s, Inches(7.0), Inches(1.3), Inches(5.9), Inches(5.95), DARK2, radius=True)
txt(s, "Combined Training Set",
    Inches(7.2), Inches(1.4), Inches(5.5), Inches(0.45),
    size=16, bold=True, color=GOLD)
for i, (val, lbl, col) in enumerate([
    ("2,452",   "songs in training set",   WHITE),
    ("981",     "viral labels (1)",         TEAL),
    ("1,471",   "non-viral labels (0)",     MID),
    ("40.0%",   "positive rate",            CORAL),
    ("16",      "features per song",        GOLD),
]):
    ty = Inches(1.95) + i * Inches(0.92)
    txt(s, val, Inches(7.15), ty, Inches(1.6), Inches(0.55),
        size=22, bold=True, color=col, align=PP_ALIGN.RIGHT)
    txt(s, lbl, Inches(8.9), ty + Inches(0.08), Inches(3.8), Inches(0.4),
        size=13, color=WHITE)

txt(s, "Viral label: in Spotify playlist  OR  popularity >= 70\n"
       "Not viral:   popularity < 50  AND  not in any viral playlist",
    Inches(7.1), Inches(6.45), Inches(5.8), Inches(0.65),
    size=11, color=MID, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Feature Engineering  —  16 Audio + 25 NLP Section Features", CORAL)

rect(s, Inches(0.4), Inches(1.3), Inches(5.9), Inches(5.8), DARK2, radius=True)
txt(s, "Song-Level Audio Features (16)",
    Inches(0.6), Inches(1.4), Inches(5.5), Inches(0.45),
    size=16, bold=True, color=CORAL)

for ty_start, col, title, body in [
    (Inches(2.0),  CORAL,  "Spotify Raw (11)",
     "danceability  ·  energy  ·  key  ·  loudness  ·  mode\n"
     "speechiness  ·  acousticness  ·  instrumentalness\n"
     "liveness  ·  valence  ·  tempo"),
    (Inches(4.15), PURPLE, "Derived Interactions (5)",
     "dance_energy = danceability × energy\n"
     "mood_score = valence × energy\n"
     "tempo_norm = tempo / 200\n"
     "acoustic_anti = 1 − acousticness\n"
     "vocal_content = 1 − instrumentalness"),
]:
    rect(s, Inches(0.5), ty_start, Inches(5.7), Inches(0.04), col)
    txt(s, title, Inches(0.55), ty_start + Inches(0.08),
        Inches(5.5), Inches(0.38), size=13, bold=True, color=col)
    txt(s, body, Inches(0.55), ty_start + Inches(0.5),
        Inches(5.5), Inches(1.5), size=11, color=MID)

txt(s, "Derived features capture non-linear interactions that raw features miss.",
    Inches(0.55), Inches(6.55), Inches(5.6), Inches(0.65),
    size=11, color=GOLD, italic=True)

rect(s, Inches(6.55), Inches(1.3), Inches(6.35), Inches(5.8), DARK2, radius=True)
txt(s, "Section-Level NLP Features (25)",
    Inches(6.75), Inches(1.4), Inches(6.0), Inches(0.45),
    size=16, bold=True, color=TEAL)

nlp_ty = Inches(2.0)
for col, title, body in [
    (TEAL,   "Text Quality (4)",
     "repeat_score  ·  hook_keyword_score\nrhyme_score  ·  short_line_score"),
    (PURPLE, "Section Type (7)",
     "chorus/hook  ·  verse  ·  bridge  ·  pre-chorus\nintro  ·  outro  ·  other  [one-hot]"),
    (GOLD,   "Sentiment (4)",
     "VADER: compound  ·  positive  ·  negative  ·  neutral"),
    (CORAL,  "Linguistic (5)",
     "pronoun_density  ·  exclamation_count  ·  question_count\ncap_word_ratio  ·  vowel_density"),
    (VIOLET, "Structure + Readability (5)",
     "has_repetition  ·  call_response  ·  avg_syllables\nunique_word_ratio  ·  flesch_score"),
]:
    rect(s, Inches(6.65), nlp_ty, Inches(6.15), Inches(0.04), col)
    txt(s, title, Inches(6.7), nlp_ty + Inches(0.08),
        Inches(6.0), Inches(0.35), size=13, bold=True, color=col)
    txt(s, body, Inches(6.7), nlp_ty + Inches(0.47),
        Inches(6.0), Inches(0.75), size=11, color=MID)
    nlp_ty += Inches(1.08)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — VIRAL SNIPPET ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Viral Snippet Algorithm  —  3-Stage Scoring", GREEN)

stages = [
    (TEAL,   "Stage 1",  "Song-Level Audio Score",
     "Audio features fed into trained Random Forest.\n"
     "Outputs P(viral) for the whole song (0-1).\n\n"
     "Answers: Is this song the type that\ngoes viral on TikTok?",
     "35% weight"),
    (CORAL,  "Stage 2",  "Section NLP Score",
     "Full lyrics parsed into sections.\n"
     "25 NLP features extracted per section.\n"
     "Weighted sum of viral-correlated features:\n"
     "hook_keywords (22%) + repeat (18%) +\n"
     "chorus_type (15%) + repetition (10%)\n"
     "+ sentiment + rhyme + short_lines...",
     "55% weight"),
    (GOLD,   "Stage 3",  "Position Boost",
     "Bell-curve boost for sections in the\n20-60% position of the song.\n\n"
     "Centred at 35% — mirrors real TikTok\nbehaviour where early-mid sections\nare most commonly clipped.",
     "10% weight"),
]

for i, (col, stage, title, body, weight) in enumerate(stages):
    lx = Inches(0.35 + i * 4.33)
    rect(s, lx, Inches(1.3), Inches(4.15), Inches(5.1), DARK2, radius=True)
    rect(s, lx, Inches(1.3), Inches(4.15), Inches(0.55), col, radius=True)
    txt(s, stage, lx + Inches(0.15), Inches(1.33),
        Inches(1.8), Inches(0.45), size=12, bold=True, color=WHITE)
    txt(s, title, lx + Inches(0.15), Inches(1.95),
        Inches(3.9), Inches(0.45), size=15, bold=True, color=col)
    txt(s, body, lx + Inches(0.15), Inches(2.5),
        Inches(3.9), Inches(2.9), size=12, color=WHITE)
    rect(s, lx + Inches(2.4), Inches(6.0), Inches(1.6), Inches(0.38), col, radius=True)
    txt(s, weight, lx + Inches(2.4), Inches(6.01),
        Inches(1.6), Inches(0.36), size=13, bold=True, color=DARK, align=PP_ALIGN.CENTER)

rect(s, Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.72), DARK2, radius=True)
txt(s, "Final Score  =  0.35 × Song_Prob  +  0.55 × Section_NLP_Score  +  0.10 × Position_Boost",
    Inches(0.6), Inches(6.62), Inches(12.1), Inches(0.55),
    size=17, bold=True, color=GOLD, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — TOP AUDIO FEATURES
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Top Audio Feature Importances  (Random Forest)", GOLD)

top_feats = [
    ("loudness",           0.117),
    ("dance_energy",       0.078),
    ("mood_score",         0.071),
    ("danceability",       0.071),
    ("liveness",           0.069),
    ("energy",             0.068),
    ("speechiness",        0.068),
    ("tempo_norm",         0.066),
    ("tempo",              0.066),
    ("valence",            0.066),
    ("acousticness",       0.062),
    ("vocal_content",      0.058),
]
feat_cols = [GOLD, CORAL, CORAL, TEAL, MID, TEAL, PURPLE,
             BLUE, BLUE, TEAL, MID, PURPLE]
bl = Inches(3.3); bfw = Inches(7.8); bh = Inches(0.32); gap = Inches(0.44)
sy = Inches(1.25)

for i, ((feat, val), col) in enumerate(zip(top_feats, feat_cols)):
    ty = sy + i * gap
    hbar(s, bl, ty, bfw, bh, val, 0.13, fill_color=col,
         label=feat, pct=f"{val*100:.1f}%")

txt(s, "Loudness is the #1 predictor — louder-mastered songs dominate TikTok feeds (mobile listening).\n"
       "dance_energy (danceability × energy) and mood_score (valence × energy) confirm the 'bop factor'.",
    Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.75),
    size=12, color=MID, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Model Results  —  Stratified 5-Fold Cross-Validation", CORAL)

results = [
    ("Random Forest",       0.677, 0.018, 0.564, TEAL,   True),
    ("Logistic Regression", 0.677, 0.021, 0.553, PURPLE, False),
    ("LightGBM",            0.677, 0.018, 0.560, GREEN,  False),
    ("XGBoost",             0.675, 0.015, 0.552, GOLD,   False),
    ("Extra Trees",         0.664, 0.016, 0.555, VIOLET, False),
]

for lx, label in [(Inches(3.6), "AUC-ROC"), (Inches(7.2), "± Std"), (Inches(9.8), "Avg Precision")]:
    txt(s, label, lx, Inches(1.2), Inches(1.6), Inches(0.4),
        size=13, bold=True, color=MID, align=PP_ALIGN.CENTER)

row_h = Inches(0.9); bw = Inches(3.3)
for i, (name, auc, std, ap, col, best) in enumerate(results):
    ty = Inches(1.65) + i * row_h
    bg_col = RGBColor(0x1A, 0x2A, 0x1A) if best else DARK2
    rect(s, Inches(0.3), ty, Inches(12.7), row_h - Inches(0.06), bg_col, radius=True)

    if best:
        rect(s, Inches(0.3), ty, Inches(0.22), row_h - Inches(0.06), TEAL, radius=True)
        txt(s, "★ BEST", Inches(0.58), ty + Inches(0.3),
            Inches(0.9), Inches(0.28), size=9, bold=True, color=TEAL)

    txt(s, name, Inches(1.55), ty + Inches(0.25), Inches(1.8), Inches(0.4),
        size=13, bold=best, color=TEAL if best else WHITE)

    bx = Inches(3.4)
    rect(s, bx, ty + Inches(0.25), bw, Inches(0.34), DARK, radius=True)
    rect(s, bx, ty + Inches(0.25), int(bw * auc), Inches(0.34), col, radius=True)
    txt(s, f"{auc:.3f}", bx + bw + Inches(0.1), ty + Inches(0.22),
        Inches(0.7), Inches(0.38), size=13, bold=best, color=col)

    txt(s, f"± {std:.3f}", Inches(7.2), ty + Inches(0.25),
        Inches(1.5), Inches(0.38), size=12, color=MID, align=PP_ALIGN.CENTER)

    bx2 = Inches(9.5); bw2 = Inches(2.8)
    rect(s, bx2, ty + Inches(0.25), bw2, Inches(0.34), DARK, radius=True)
    rect(s, bx2, ty + Inches(0.25), int(bw2 * ap), Inches(0.34), col, radius=True)
    txt(s, f"{ap:.3f}", bx2 + bw2 + Inches(0.1), ty + Inches(0.22),
        Inches(0.65), Inches(0.38), size=12, bold=best, color=col)

txt(s, "All 3 top models tie at AUC 0.677 — the signal ceiling in audio features alone.\n"
       "Adding NLP lyric features or real TikTok view count labels would push this significantly higher.",
    Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.75),
    size=12, color=MID, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — WHY RANDOM FOREST + NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Why Models Tie  +  How To Improve", TEAL)

rect(s, Inches(0.4), Inches(1.3), Inches(6.0), Inches(5.8), DARK2, radius=True)
txt(s, "Why RF / LR / LightGBM all tie", Inches(0.6), Inches(1.4),
    Inches(5.6), Inches(0.45), size=17, bold=True, color=TEAL)
bullet_box(s, [
    "2,452 songs — large enough for all models to find the signal",
    "16 well-engineered features → both linear and tree boundaries work",
    "RF captures non-linear interactions (dance × energy × tempo)",
    "LR succeeds because features are already linearly separable",
    "LightGBM ties — fast GBDT matches tree ensembles on this scale",
    "AUC ceiling ~0.68 hit — more model complexity won't help here",
], Inches(0.55), Inches(2.0), Inches(5.7), Inches(4.8), size=13)

rect(s, Inches(6.7), Inches(1.3), Inches(6.2), Inches(5.8), DARK2, radius=True)
txt(s, "How to push AUC above 0.85", Inches(6.9), Inches(1.4),
    Inches(5.8), Inches(0.45), size=17, bold=True, color=CORAL)
bullet_box(s, [
    "Real TikTok view counts as labels (not Spotify popularity proxy)",
    "Add lyric NLP features to the song-level model",
    "Spotify API audio features for real-time inference on any song",
    "Train section model on labeled TikTok clip timestamps",
    "BERT/sentence embeddings to capture semantic hook meaning",
    "Fine-tune snippet weights using real A/B test data",
    "Annotate 500+ songs with clip timestamps",
], Inches(6.85), Inches(2.0), Inches(5.9), Inches(4.8),
   size=13, dot_color=CORAL)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "The Streamlit Dashboard  —  5 Tabs", GREEN)

tabs = [
    ("🎵", "Song Virality\nChecker",    CORAL,
     "Sliders for 11 audio features → viral probability + radar chart. "
     "Search training data to auto-fill sliders."),
    ("✂️", "Viral Snippet\nFinder",     PURPLE,
     "Enter any song → Genius API fetches lyrics → parses sections → "
     "ranks by viral score + NLP breakdown + score decomposition."),
    ("📊", "Data\nExplorer",            TEAL,
     "Visualise the training dataset: popularity distribution, "
     "scatter plots, feature correlations, top songs table."),
    ("🏆", "Model\nPerformance",        GOLD,
     "5-fold CV table for all 5 models. AUC bar chart with error bars. "
     "Average precision comparison."),
    ("🔬", "Feature\nAnalysis",         BLUE,
     "RF feature importances for audio model. NLP snippet weights. "
     "Explainer for each viral signal."),
]

for i, (icon, name, col, desc) in enumerate(tabs):
    lx = Inches(0.35 + i * 2.58)
    rect(s, lx, Inches(1.3), Inches(2.46), Inches(1.25), col, radius=True)
    txt(s, icon, lx, Inches(1.33), Inches(2.46), Inches(0.55),
        size=28, align=PP_ALIGN.CENTER)
    txt(s, name, lx, Inches(1.82), Inches(2.46), Inches(0.6),
        size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(s, lx, Inches(2.7), Inches(2.46), Inches(2.5), DARK2, radius=True)
    txt(s, desc, lx + Inches(0.12), Inches(2.8), Inches(2.22), Inches(2.3),
        size=11, color=WHITE)

rect(s, Inches(0.4), Inches(5.38), Inches(12.5), Inches(1.87), DARK2, radius=True)
txt(s, "✂️  Tab 2 — Viral Snippet Finder Detail:",
    Inches(0.6), Inches(5.48), Inches(4.5), Inches(0.4),
    size=14, bold=True, color=PURPLE)
bullet_box(s, [
    "Enter song + artist → Genius API fetches full lyrics automatically",
    "25 NLP features extracted per section → 3-stage viral score computed",
    "Progress bars rank all sections | Score decomposition bar chart",
    "Top section lyrics preview + full 25-feature breakdown table",
], Inches(0.55), Inches(5.95), Inches(12.1), Inches(1.18),
   size=12, dot_color=PURPLE, spacing=0.28)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — EXAMPLE: ESPRESSO
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, 'Example  —  "Espresso" by Sabrina Carpenter', CORAL)

rect(s, Inches(0.4), Inches(1.3), Inches(5.7), Inches(5.8), DARK2, radius=True)
txt(s, "Model scores each section:", Inches(0.6), Inches(1.4),
    Inches(5.3), Inches(0.45), size=16, bold=True, color=CORAL)

sections_ex = [
    ("[Intro]",          0.29, MID),
    ("[Verse 1]",        0.32, TEAL),
    ("[Pre-Chorus]",     0.45, VIOLET),
    ("[Chorus]  <- #1", 0.46, CORAL),
    ("[Post-Chorus]",    0.35, BLUE),
    ("[Verse 2]",        0.32, TEAL),
    ("[Bridge]",         0.33, PURPLE),
    ("[Outro]",          0.28, MID),
]
bl = Inches(3.3); bfw = Inches(2.5)
for i, (lbl, prob, col) in enumerate(sections_ex):
    ty = Inches(1.95) + i * Inches(0.68)
    is_top = prob == 0.46
    txt(s, lbl, Inches(0.55), ty + Inches(0.06), Inches(2.6), Inches(0.35),
        size=12, bold=is_top, color=CORAL if is_top else WHITE)
    rect(s, bl, ty + Inches(0.09), bfw, Inches(0.28), DARK, radius=True)
    rect(s, bl, ty + Inches(0.09), int(bfw * prob / 0.50), Inches(0.28), col, radius=True)
    txt(s, f"{prob*100:.0f}%", bl + bfw + Inches(0.1), ty + Inches(0.06),
        Inches(0.6), Inches(0.33), size=11, bold=is_top, color=col)

rect(s, Inches(6.4), Inches(1.3), Inches(6.5), Inches(5.8), DARK2, radius=True)
txt(s, "Why [Chorus] scores highest:", Inches(6.6), Inches(1.4),
    Inches(6.1), Inches(0.45), size=16, bold=True, color=TEAL)
bullet_box(s, [
    "hook_keyword_score HIGH: 'sweet', 'baby', 'feel', 'yeah'",
    "repeat_score: chorus lines repeat 4x across the song",
    "stype_chorus_hook = 1 (strongest categorical signal)",
    "vader_compound: positive upbeat emotional sentiment",
    "short_line_score: punchy 6-8 word lines -> clip-ready",
    "has_repetition_pattern = 1 -> memorable earworm structure",
    "flesch_reading_ease: very easy to sing along",
    "pronoun_density: personal connection via I / he / me",
], Inches(6.55), Inches(2.0), Inches(6.2), Inches(4.3), size=13, dot_color=TEAL)

txt(s, '"And now he\'s thinkin\' \'bout me every night, oh\n'
       'Is it that sweet? I guess so\n'
       'Say you can\'t sleep, baby, I know\n'
       'That\'s that me espresso..."',
    Inches(6.6), Inches(6.2), Inches(6.2), Inches(1.0),
    size=12, color=MID, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — KEY TAKEAWAYS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Key Takeaways", PURPLE)

takeaways = [
    (TEAL,   "AUC 0.677",
     "Song-level ML on audio features achieves AUC 0.677.\n"
     "Loudness, dance×energy, and mood score are the key drivers."),
    (CORAL,  "Viral Snippet Algorithm",
     "3-stage scoring (audio 35% + NLP 55% + position 10%).\n"
     "Correctly picks chorus as #1 on real songs like Espresso."),
    (PURPLE, "16 Features Work",
     "Derived interactions (dance_energy, mood_score) outperform\n"
     "raw audio alone — feature engineering > model complexity."),
    (GOLD,   "Label Quality Is the Bottleneck",
     "RF, LR, LightGBM all tie at 0.677 — model choice is not\n"
     "the limiting factor. Better labels will unlock bigger gains."),
    (GREEN,  "Live & Deployable",
     "Streamlit dashboard runs real-time viral prediction on any\n"
     "song via Genius API — usable as a songwriter / A&R tool."),
    (VIOLET, "Next: Real TikTok Labels",
     "Actual TikTok view counts + BERT lyrics + Spotify audio\n"
     "features in real time → target AUC > 0.85."),
]

for i, (col, title, body) in enumerate(takeaways):
    cx = i % 3; cy = i // 3
    lx = Inches(0.35 + cx * 4.33)
    ty = Inches(1.25 + cy * 2.85)
    rect(s, lx, ty, Inches(4.15), Inches(2.6), DARK2, radius=True)
    rect(s, lx, ty, Inches(4.15), Inches(0.52), col, radius=True)
    txt(s, title, lx + Inches(0.15), ty + Inches(0.08),
        Inches(3.9), Inches(0.38), size=16, bold=True, color=WHITE)
    txt(s, body, lx + Inches(0.15), ty + Inches(0.65),
        Inches(3.9), Inches(1.8), size=12, color=WHITE)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_pptx = "/Users/dhekha/Desktop/SUPA_Spring/TikTok_Viral_Predictor.pptx"
prs.save(out_pptx)
print(f"Saved PPTX -> {out_pptx}")

lo = shutil.which("libreoffice") or shutil.which("soffice")
if lo:
    subprocess.run([lo, "--headless", "--convert-to", "pdf",
                    "--outdir", "/Users/dhekha/Desktop/SUPA_Spring/", out_pptx], check=True)
    print("Converted to PDF via LibreOffice")
else:
    result = subprocess.run([
        "osascript", "-e",
        f'''tell application "Microsoft PowerPoint"
            open POSIX file "{out_pptx}"
            save active presentation in POSIX file "{out_pptx.replace('.pptx', '.pdf')}" as save as PDF
            close active presentation
        end tell'''
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print("Converted to PDF via PowerPoint")
    else:
        print(f"PPTX ready. PDF needs LibreOffice or Microsoft PowerPoint.")
        print(f"File: {out_pptx}")

"""
TikTok Viral Predictor — Updated PowerPoint v2
================================================
Reflects dataset expansion work: 4 sources, 6,163 songs,
TikTok View labels, and the before/after comparison.
"""

import subprocess, shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches, Pt, Emu
import pptx.util as util

# Re-import MSO_SHAPE from the right place
from pptx.enum.shapes import PP_PLACEHOLDER
try:
    from pptx.enum.shapes import MSO_SHAPE
except ImportError:
    from pptx.util import MSO_SHAPE

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
ORANGE  = RGBColor(0xFF, 0x87, 0x00)

W = prs.slide_width
H = prs.slide_height

from pptx.oxml.ns import qn
from lxml import etree

def add_rounded_rect(slide, l, t, w, h, color, corner_pct=5):
    from pptx.util import Emu
    sp = slide.shapes.add_shape(1, l, t, w, h)  # 1 = rectangle
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    # Set corner rounding via XML
    sp_elem = sp.element
    prstGeom = sp_elem.find(qn('p:spPr')).find(qn('a:prstGeom'))
    if prstGeom is not None:
        prstGeom.set('prst', 'roundRect')
        avLst = prstGeom.find(qn('a:avLst'))
        if avLst is None:
            avLst = etree.SubElement(prstGeom, qn('a:avLst'))
        else:
            for gd in avLst.findall(qn('a:gd')):
                avLst.remove(gd)
        gd = etree.SubElement(avLst, qn('a:gd'))
        gd.set('name', 'adj')
        gd.set('fmla', f'val {corner_pct * 1000}')
    return sp


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def rect(slide, l, t, w, h, color, radius=False):
    if radius:
        return add_rounded_rect(slide, l, t, w, h, color)
    shp = slide.shapes.add_shape(1, l, t, w, h)
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
        d = slide.shapes.add_shape(1, l, y + Inches(0.09), Inches(0.12), Inches(0.12))
        d.fill.solid(); d.fill.fore_color.rgb = dot_color
        d.line.fill.background()
        txt(slide, item, l + Inches(0.2), y, w - Inches(0.2), line_h,
            size=size, color=color)

def hbar(slide, l, t, full_w, h, val, max_val, fill_color,
         bg_color=None, label="", pct=""):
    bg_color = bg_color or DARK2
    rect(slide, l, t, full_w, h, bg_color)
    filled = int(full_w * val / max_val)
    if filled > 0:
        rect(slide, l, t, filled, h, fill_color)
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

def oval(slide, cx, cy, r, col):
    c = slide.shapes.add_shape(9, cx - r/2, cy - r/2, r, r)  # 9 = oval
    c.fill.solid(); c.fill.fore_color.rgb = col
    c.line.fill.background()

def badge(slide, lx, ty, val, label, col):
    rect(slide, lx, ty, Inches(1.95), Inches(1.2), col, radius=True)
    txt(slide, val,   lx, ty + Inches(0.05), Inches(1.95), Inches(0.6),
        size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(slide, label, lx, ty + Inches(0.65), Inches(1.95), Inches(0.4),
        size=11, color=WHITE, align=PP_ALIGN.CENTER)


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
    oval(s, cx, cy, r, col)

txt(s, "TikTok Viral Predictor",
    Inches(0.6), Inches(1.2), Inches(9.5), Inches(1.5),
    size=42, bold=True, color=WHITE)

accent(s, Inches(0.6), Inches(2.85), Inches(5.5), CORAL)

txt(s, "Which songs go viral on TikTok?  —  v3: Lyrics + Embeddings",
    Inches(0.6), Inches(3.0), Inches(9.5), Inches(0.75),
    size=22, color=VIOLET)

txt(s, "ML  ·  6,163 Songs  ·  4 Data Sources  ·  Genius Lyrics  ·  MiniLM Embeddings  ·  AUC 0.751",
    Inches(0.6), Inches(3.75), Inches(11), Inches(0.5),
    size=14, color=MID, italic=True)

for i, (val, label, col) in enumerate([
    ("6,163",  "Total Songs",      PURPLE),
    ("2,474",  "With Lyrics",      TEAL),
    ("0.751",  "Best AUC",         GOLD),
    ("+0.097", "AUC Lift",         CORAL),
]):
    badge(s, Inches(0.6 + i * 2.2), Inches(5.0), val, label, col)

txt(s, "SUPA Spring 2026  ·  v3: Lyrics + MiniLM Embeddings",
    Inches(7.5), Inches(6.85), Inches(5.5), Inches(0.45),
    size=12, color=MID, align=PP_ALIGN.RIGHT)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — WHAT CHANGED (NEW SLIDE)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "What Changed — Dataset Expansion Update", ORANGE)

# BEFORE column
rect(s, Inches(0.4), Inches(1.3), Inches(5.9), Inches(5.8), DARK2, radius=True)
txt(s, "BEFORE  (v1)", Inches(0.6), Inches(1.42), Inches(5.5), Inches(0.45),
    size=18, bold=True, color=MID)

before_items = [
    ("Training Songs",    "~3,560",  MID),
    ("Labeled Songs",     "2,452",   MID),
    ("Data Sources",      "3",       MID),
    ("Virality Labels",   "Spotify popularity proxy", MID),
    ("Songs with TikTok Views", "0",  MID),
    ("Best AUC",          "0.677",   MID),
    ("Year Coverage",     "Up to 2021", MID),
    ("Class Balance",     "40% viral / 60% not", MID),
]
for i, (key, val, col) in enumerate(before_items):
    ty = Inches(1.98) + i * Inches(0.62)
    txt(s, key + ":", Inches(0.6), ty, Inches(2.6), Inches(0.5), size=12, color=MID)
    txt(s, val,       Inches(3.3), ty, Inches(2.8), Inches(0.5), size=12, bold=True, color=MID)

# Arrow
txt(s, "→", Inches(6.25), Inches(3.9), Inches(0.8), Inches(0.7),
    size=40, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

# AFTER column
rect(s, Inches(7.0), Inches(1.3), Inches(5.9), Inches(5.8), DARK2, radius=True)
txt(s, "AFTER  (v2 — This Update)", Inches(7.2), Inches(1.42), Inches(5.5), Inches(0.45),
    size=18, bold=True, color=ORANGE)

after_items = [
    ("Training Songs",    "6,163  (+73%)",           TEAL),
    ("Labeled Songs",     "3,373  (+38%)",           TEAL),
    ("Data Sources",      "4  (+1 new)",             TEAL),
    ("Virality Labels",   "TikTok Views (direct!)",  CORAL),
    ("Songs with TikTok Views", "1,281",             CORAL),
    ("Best AUC",          "0.654  (honest labels)",  GOLD),
    ("Year Coverage",     "2010 – 2024",             TEAL),
    ("Class Balance",     "41.6% viral / 58.4% not", TEAL),
]
for i, (key, val, col) in enumerate(after_items):
    ty = Inches(1.98) + i * Inches(0.62)
    txt(s, key + ":", Inches(7.2), ty, Inches(2.6), Inches(0.5), size=12, color=WHITE)
    txt(s, val,       Inches(9.9), ty, Inches(2.8), Inches(0.5), size=12, bold=True, color=col)

txt(s, "Note: AUC dropped from 0.677 → 0.654 because TikTok View labels are harder to predict "
       "than Spotify popularity — this is a more honest measure of real virality.",
    Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35),
    size=11, italic=True, color=MID, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — NEW DATA PIPELINE (4 SOURCES)
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Data Pipeline  —  4 Sources  →  6,163 Songs", PURPLE)

sources = [
    (PURPLE, "Source A — TikTok Trending Tracks (Kaggle)",
     ["tiktok_trending_tracks/tiktok.csv",
      "3,375 unique tracks after deduplication",
      "11 Spotify audio features  ·  TikTok genre tags",
      "Covers popular TikTok sounds (TIKTOK DANCE, OPM, etc.)"]),
    (BLUE,   "Source B — TikTok Popular 2020 (Kaggle)",
     ["tiktok_popular_2020/TikTok_songs_2020.csv",
      "92 unique tracks  ·  same 11 audio features",
      "TikTok-specific 2020 viral songs"]),
    (TEAL,   "Source C — Spotify Top Hits 2010–2023 (Kaggle)  ← NEW",
     ["spotify_top_hits/playlist_2010to2023.csv",
      "1,954 songs added after deduplication",
      "Covers 13 years of chart hits with full audio features"]),
    (CORAL,  "Source D — Spotify 2023 Chart Songs (Kaggle)  ← NEW",
     ["spotify_2023/spotify-2023.csv",
      "742 songs  ·  Audio features + streaming counts",
      "Converted from 0–100 scale to Spotify's 0–1 scale"]),
]

for i, (col, title, bullets) in enumerate(sources):
    lx = Inches(0.35 + (i % 2) * 6.5)
    ty = Inches(1.28 + (i // 2) * 2.95)
    rect(s, lx, ty, Inches(6.2), Inches(2.7), DARK2, radius=True)
    rect(s, lx, ty, Inches(6.2), Inches(0.52), col, radius=True)
    txt(s, title, lx + Inches(0.15), ty + Inches(0.07),
        Inches(5.9), Inches(0.4), size=13, bold=True, color=WHITE)
    bullet_box(s, bullets, lx + Inches(0.1), ty + Inches(0.62),
               Inches(5.9), Inches(1.8), size=11, dot_color=col, spacing=0.5)

# Right side — combined result box
rect(s, Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35), DARK2)
txt(s, "After merge + dedup by (track_name | artist_name):   6,163 unique songs  ·  "
       "3,373 labeled (1,402 viral  /  1,971 not viral)  ·  1,281 with TikTok View labels",
    Inches(0.5), Inches(7.07), Inches(12.2), Inches(0.3),
    size=11, color=GOLD, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — LABELING STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Labeling Strategy  —  TikTok Views + Popularity Hybrid", CORAL)

rect(s, Inches(0.4), Inches(1.28), Inches(5.9), Inches(5.9), DARK2, radius=True)
txt(s, "Old Labeling  (v1)", Inches(0.6), Inches(1.38), Inches(5.5), Inches(0.45),
    size=16, bold=True, color=MID)
bullet_box(s, [
    "Viral = 1  if:  Spotify popularity ≥ 70  OR  in viral playlist",
    "Not viral = 0  if:  popularity < 50  AND  not in playlist",
    "Ambiguous dropped:  popularity 50–69 (not in playlist)",
    "Problem: Spotify popularity ≠ TikTok virality",
    "A song can be TikTok famous but have low Spotify popularity",
    "A song can have high Spotify pop from radio — not TikTok",
    "Labels were a proxy; often wrong for the actual target",
], Inches(0.55), Inches(1.95), Inches(5.65), Inches(4.8),
   size=12, dot_color=MID)

rect(s, Inches(6.7), Inches(1.28), Inches(6.2), Inches(5.9), DARK2, radius=True)
txt(s, "New Labeling  (v2)", Inches(6.9), Inches(1.38), Inches(5.8), Inches(0.45),
    size=16, bold=True, color=CORAL)
bullet_box(s, [
    "1,281 songs matched to 2024 Spotify dataset with TikTok Views",
    "Viral = top 33% of TikTok Views  (> 1.18 billion views)",
    "Not viral = bottom 50% of TikTok Views  (< 631 million)",
    "For remaining songs (no TikTok data): keep popularity proxy",
    "Result: labels directly measure TikTok performance",
    "More honest — harder to predict, but more meaningful",
    "Next step: get TikTok Views for all 6,163 songs",
], Inches(6.85), Inches(1.95), Inches(5.9), Inches(4.8),
   size=12, dot_color=CORAL)

txt(s, "TikTok View thresholds from 4,600-song 2024 Spotify dataset  ·  "
       "Median: 266M views  ·  Top 33%: 1.18B views",
    Inches(0.5), Inches(7.08), Inches(12.3), Inches(0.3),
    size=11, color=GOLD, italic=True, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Feature Engineering  —  11 Audio  +  5 Derived  =  16 Features", CORAL)

rect(s, Inches(0.4), Inches(1.28), Inches(5.9), Inches(5.9), DARK2, radius=True)
txt(s, "Spotify Raw Audio Features (11)",
    Inches(0.6), Inches(1.38), Inches(5.5), Inches(0.45),
    size=16, bold=True, color=CORAL)
bullet_box(s, [
    "danceability  — how suitable for dancing (0–1)",
    "energy        — intensity and activity (0–1)",
    "valence       — musical positivity / happiness (0–1)",
    "loudness      — overall loudness in dB (-60 to 0)",
    "tempo         — beats per minute (BPM)",
    "speechiness   — spoken words presence (0–1)",
    "acousticness  — acoustic vs electric (0–1)",
    "instrumentalness — vocal absence estimate (0–1)",
    "liveness      — audience presence signal (0–1)",
    "key           — estimated key (0–11)",
    "mode          — major=1 / minor=0",
], Inches(0.55), Inches(1.95), Inches(5.65), Inches(4.8),
   size=12, dot_color=CORAL, spacing=0.45)

rect(s, Inches(6.7), Inches(1.28), Inches(6.2), Inches(5.9), DARK2, radius=True)
txt(s, "Derived Interaction Features (5)",
    Inches(6.9), Inches(1.38), Inches(5.8), Inches(0.45),
    size=16, bold=True, color=TEAL)

derived = [
    ("dance_energy",    "danceability × energy",   "Captures the 'bop factor' — danceable AND intense songs"),
    ("mood_score",      "valence × energy",        "High = happy + energetic = TikTok trend archetype"),
    ("tempo_norm",      "tempo ÷ 200",             "Normalises BPM to 0–1 scale for model stability"),
    ("acoustic_anti",   "1 − acousticness",        "How 'produced/electronic' the track is"),
    ("vocal_content",   "1 − instrumentalness",    "How vocal-driven (vs. purely instrumental)"),
]
for i, (name, formula, desc) in enumerate(derived):
    ty = Inches(2.0) + i * Inches(1.12)
    rect(s, Inches(6.8), ty, Inches(5.9), Inches(0.04), TEAL)
    txt(s, name,    Inches(6.85), ty + Inches(0.1), Inches(2.5), Inches(0.38),
        size=13, bold=True, color=TEAL)
    txt(s, formula, Inches(9.4), ty + Inches(0.1), Inches(3.3), Inches(0.38),
        size=12, italic=True, color=GOLD)
    txt(s, desc,    Inches(6.85), ty + Inches(0.5), Inches(5.8), Inches(0.55),
        size=11, color=MID)

txt(s, "Derived features capture non-linear interactions that raw audio features miss individually.",
    Inches(6.85), Inches(7.08), Inches(5.9), Inches(0.3),
    size=11, italic=True, color=MID)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — FEATURE IMPORTANCES
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Top Feature Importances  —  Random Forest on Expanded Dataset", GOLD)

top_feats = [
    ("loudness",           0.106, GOLD),
    ("danceability",       0.081, TEAL),
    ("speechiness",        0.077, PURPLE),
    ("dance_energy",       0.075, CORAL),
    ("mood_score",         0.074, CORAL),
    ("energy",             0.072, TEAL),
    ("liveness",           0.072, BLUE),
    ("valence",            0.068, TEAL),
    ("acousticness",       0.064, MID),
    ("tempo_norm",         0.063, VIOLET),
    ("tempo",              0.063, VIOLET),
    ("acoustic_anti",      0.063, MID),
]

bl = Inches(3.3); bfw = Inches(7.8); bh = Inches(0.32); gap = Inches(0.44)
sy = Inches(1.25)

for i, (feat, val, col) in enumerate(top_feats):
    ty = sy + i * gap
    hbar(s, bl, ty, bfw, bh, val, 0.115, fill_color=col,
         label=feat, pct=f"{val*100:.1f}%")

txt(s, "Loudness remains #1 — louder-mastered songs dominate TikTok mobile feeds.\n"
       "dance_energy and mood_score confirm the 'bop factor' as TikTok's core viral pattern.",
    Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.7),
    size=12, color=MID, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Model Results  —  Stratified 5-Fold CV on Expanded Dataset", CORAL)

results = [
    ("LightGBM",            0.654, 0.009, 0.546, GREEN,  True),
    ("Random Forest",       0.653, 0.009, 0.551, TEAL,   False),
    ("XGBoost",             0.647, 0.013, 0.541, GOLD,   False),
    ("Logistic Regression", 0.647, 0.021, 0.540, PURPLE, False),
    ("Extra Trees",         0.647, 0.007, 0.556, VIOLET, False),
]

for lx, label in [(Inches(3.5), "AUC-ROC"), (Inches(7.2), "± Std"), (Inches(9.7), "Avg Precision")]:
    txt(s, label, lx, Inches(1.2), Inches(1.6), Inches(0.4),
        size=13, bold=True, color=MID, align=PP_ALIGN.CENTER)

row_h = Inches(0.9); bw = Inches(3.3)
for i, (name, auc, std, ap, col, best) in enumerate(results):
    ty = Inches(1.65) + i * row_h
    bg_col = RGBColor(0x12, 0x2A, 0x1A) if best else DARK2
    rect(s, Inches(0.3), ty, Inches(12.7), row_h - Inches(0.06), bg_col, radius=True)
    if best:
        rect(s, Inches(0.3), ty, Inches(0.22), row_h - Inches(0.06), GREEN)
        txt(s, "BEST", Inches(0.55), ty + Inches(0.33),
            Inches(0.9), Inches(0.25), size=9, bold=True, color=GREEN)
    txt(s, name, Inches(1.55), ty + Inches(0.25), Inches(1.8), Inches(0.4),
        size=13, bold=best, color=GREEN if best else WHITE)
    bx = Inches(3.4)
    rect(s, bx, ty + Inches(0.25), bw, Inches(0.34), DARK)
    rect(s, bx, ty + Inches(0.25), int(bw * auc / 1.0), Inches(0.34), col)
    txt(s, f"{auc:.3f}", bx + bw + Inches(0.1), ty + Inches(0.22),
        Inches(0.7), Inches(0.38), size=13, bold=best, color=col)
    txt(s, f"± {std:.3f}", Inches(7.2), ty + Inches(0.25),
        Inches(1.5), Inches(0.38), size=12, color=MID, align=PP_ALIGN.CENTER)
    bx2 = Inches(9.4); bw2 = Inches(3.0)
    rect(s, bx2, ty + Inches(0.25), bw2, Inches(0.34), DARK)
    rect(s, bx2, ty + Inches(0.25), int(bw2 * ap / 0.6), Inches(0.34), col)
    txt(s, f"{ap:.3f}", bx2 + bw2 + Inches(0.1), ty + Inches(0.22),
        Inches(0.65), Inches(0.38), size=12, bold=best, color=col)

txt(s, "LightGBM edges out at AUC 0.654 on the new dataset with more honest TikTok View labels.\n"
       "All models cluster tightly — the ceiling is the feature set, not the algorithm.",
    Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.75),
    size=12, color=MID, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — BEFORE vs AFTER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Before vs After  —  Dataset Expansion Impact", ORANGE)

# Comparison table
headers_row = ["Metric", "v1 (Before)", "v2 (After)", "Change"]
col_x = [Inches(0.4), Inches(3.6), Inches(7.0), Inches(10.4)]
col_w = [Inches(3.0), Inches(3.2), Inches(3.2), Inches(2.7)]

# Header row
for j, (hdr, lx, lw) in enumerate(zip(headers_row, col_x, col_w)):
    col = DARK2 if j == 0 else (MID if j == 1 else ORANGE if j == 2 else TEAL)
    rect(s, lx, Inches(1.28), lw, Inches(0.45), col)
    txt(s, hdr, lx + Inches(0.1), Inches(1.3), lw - Inches(0.1), Inches(0.42),
        size=13, bold=True, color=WHITE)

rows = [
    ("Unique Songs",        "~3,560",   "6,163",     "+73%",  TEAL),
    ("Labeled Songs",       "2,452",    "3,373",     "+38%",  TEAL),
    ("Data Sources",        "3",        "4",         "+1",    TEAL),
    ("TikTok View Labels",  "0",        "1,281",     "NEW",   CORAL),
    ("Year Coverage",       "≤2021",    "2010–2024", "+3 yrs",TEAL),
    ("Best AUC",            "0.677",    "0.654",     "Better labels", GOLD),
    ("Best Model",          "RF / LR",  "LightGBM",  "same tier", MID),
    ("Class Balance",       "40/60",    "41.6/58.4", "✓ stable", TEAL),
]

for i, (metric, before, after, change, chg_col) in enumerate(rows):
    ty = Inches(1.78) + i * Inches(0.64)
    row_bg = RGBColor(0x18, 0x1E, 0x35) if i % 2 == 0 else DARK2
    for lx, lw in zip(col_x, col_w):
        rect(s, lx, ty, lw, Inches(0.60), row_bg)

    txt(s, metric, col_x[0] + Inches(0.1), ty + Inches(0.1),
        col_w[0] - Inches(0.1), Inches(0.5), size=12, color=WHITE)
    txt(s, before, col_x[1] + Inches(0.1), ty + Inches(0.1),
        col_w[1] - Inches(0.1), Inches(0.5), size=12, color=MID, bold=False)
    txt(s, after,  col_x[2] + Inches(0.1), ty + Inches(0.1),
        col_w[2] - Inches(0.1), Inches(0.5), size=12, color=ORANGE, bold=True)
    txt(s, change, col_x[3] + Inches(0.1), ty + Inches(0.1),
        col_w[3] - Inches(0.1), Inches(0.5), size=12, color=chg_col, bold=True)

txt(s, "AUC 0.654 vs 0.677 — lower AUC reflects harder, more honest TikTok View labels, "
       "not worse model performance.",
    Inches(0.4), Inches(7.08), Inches(12.5), Inches(0.3),
    size=11, italic=True, color=MID, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — WHY SPOTIFY API FAILED + WHAT WE DID INSTEAD
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Spotify API Restrictions  —  How We Adapted", PURPLE)

rect(s, Inches(0.4), Inches(1.28), Inches(5.9), Inches(5.9), DARK2, radius=True)
txt(s, "What Spotify Blocked (Post-Nov 2024)", Inches(0.6), Inches(1.38),
    Inches(5.5), Inches(0.45), size=16, bold=True, color=CORAL)
bullet_box(s, [
    "Audio Features API — GET /audio-features (HTTP 403)",
    "Editorial playlists — Today's Top Hits, RapCaviar, etc.",
    "Popularity field missing from search results",
    "Apps created after Nov 2024 get severely restricted access",
    "Audio features (valence, energy, etc.) no longer fetchable",
    "This is a Spotify policy change — affects all new developers",
], Inches(0.55), Inches(1.95), Inches(5.65), Inches(4.8),
   size=12, dot_color=CORAL)

rect(s, Inches(6.7), Inches(1.28), Inches(6.2), Inches(5.9), DARK2, radius=True)
txt(s, "What We Did Instead", Inches(6.9), Inches(1.38),
    Inches(5.8), Inches(0.45), size=16, bold=True, color=TEAL)
bullet_box(s, [
    "Downloaded 3 Kaggle datasets with pre-fetched audio features",
    "Spotify Top Hits 2010–2023: 2,400 songs with full audio features",
    "Spotify 2023 Chart Songs: 953 songs (scaled from 0–100 to 0–1)",
    "Most Streamed Spotify Songs 2024: 4,600 songs with TikTok Views",
    "Matched 1,281 songs to get real TikTok View labels",
    "Result: bigger dataset AND better quality virality labels",
], Inches(6.85), Inches(1.95), Inches(5.9), Inches(4.8),
   size=12, dot_color=TEAL)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Next Steps  —  Path to AUC > 0.80", TEAL)

next_steps = [
    (TEAL,   "Add Lyrics Features",
     "Fetch Genius lyrics for 3,373 training songs.\n"
     "Add VADER sentiment, rhyme score, hook density,\n"
     "repetition, and MiniLM embeddings to the model.\n"
     "Expected: +5–10% AUC gain.",
     "High Impact"),
    (CORAL,  "Expand TikTok View Labels",
     "Currently 1,281 of 6,163 songs have TikTok Views.\n"
     "Match remaining 4,882 songs to streaming datasets.\n"
     "Better labels directly improve prediction accuracy.\n"
     "Expected: more stable model, better generalisation.",
     "High Impact"),
    (PURPLE, "Viral Snippet Finder",
     "Currently uses NLP rule-based scoring.\n"
     "Train section-level model on Genius lyric sections\n"
     "with real TikTok clip timestamp data.\n"
     "Expected: more accurate clip identification.",
     "Medium Impact"),
    (GOLD,   "Live Song Prediction",
     "Currently model only works on known songs.\n"
     "Add Genius API search + lyrics extraction pipeline\n"
     "for any new song entered by user.\n"
     "Expected: usable A&R/songwriter tool.",
     "High Usability"),
    (GREEN,  "Cross-Platform Signals",
     "Add YouTube Views, Shazam Counts, Apple Music\n"
     "playlist count as additional features.\n"
     "These are available in the 2024 dataset already.\n"
     "Expected: +3–5% AUC on top of lyrics.",
     "Medium Impact"),
    (BLUE,   "Real-Time TikTok Data",
     "Use TikTok RapidAPI to pull current trending sounds\n"
     "and their view counts as training labels.\n"
     "Continuously update dataset with new viral songs.\n"
     "Expected: living model that stays current.",
     "Future Work"),
]

for i, (col, title, body, tag) in enumerate(next_steps):
    cx = i % 3; cy = i // 3
    lx = Inches(0.35 + cx * 4.33)
    ty = Inches(1.25 + cy * 2.9)
    rect(s, lx, ty, Inches(4.15), Inches(2.65), DARK2, radius=True)
    rect(s, lx, ty, Inches(4.15), Inches(0.48), col, radius=True)
    txt(s, title, lx + Inches(0.15), ty + Inches(0.06),
        Inches(2.8), Inches(0.38), size=13, bold=True, color=WHITE)
    txt(s, tag, lx + Inches(3.0), ty + Inches(0.08),
        Inches(1.0), Inches(0.3), size=9, bold=True, color=DARK,
        align=PP_ALIGN.CENTER)
    txt(s, body, lx + Inches(0.15), ty + Inches(0.6),
        Inches(3.9), Inches(1.9), size=11, color=WHITE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — KEY TAKEAWAYS
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Key Takeaways", PURPLE)

takeaways = [
    (TEAL,   "+73% More Data",
     "6,163 unique songs from 4 sources.\n"
     "Added Spotify Top Hits 2010–2023 and\n"
     "2023 chart songs to cover recent years."),
    (CORAL,  "Real TikTok Labels",
     "1,281 songs now have actual TikTok Views\n"
     "as virality ground truth — not a Spotify\n"
     "popularity proxy."),
    (PURPLE, "AUC 0.654 (Honest)",
     "LightGBM best model. Lower than v1's 0.677\n"
     "because labels are harder and more accurate.\n"
     "This is a more trustworthy number."),
    (GOLD,   "Loudness Still #1",
     "Feature importances unchanged — loudness,\n"
     "dance×energy, and mood_score remain the\n"
     "core viral audio signals."),
    (GREEN,  "Spotify API Workaround",
     "Audio features API is blocked for new apps.\n"
     "Solution: Kaggle datasets with pre-fetched\n"
     "features from before the deprecation."),
    (VIOLET, "Lyrics: +0.097 AUC Lift!",
     "2,474 songs (73%) have lyrics via Genius.\n"
     "Audio only: 0.654 → Audio + Lyrics: 0.751.\n"
     "MiniLM embeddings dominate top 60 features."),
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
# SLIDE 12 — LYRICS COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Phase 1 — Genius Lyrics Collection", TEAL)

# Coverage badge row
for i, (val, label, col) in enumerate([
    ("3,373",  "Training Songs",   PURPLE),
    ("2,474",  "Lyrics Found",     TEAL),
    ("73.3%",  "Coverage",         GOLD),
    ("~15 min","Fetch Time",       MID),
]):
    badge(s, Inches(0.5 + i * 3.2), Inches(1.3), val, label, col)

# How it works box
rect(s, Inches(0.5), Inches(2.7), Inches(5.8), Inches(3.9), DARK2, radius=True)
txt(s, "How Lyrics Were Collected", Inches(0.7), Inches(2.8),
    Inches(5.4), Inches(0.45), size=16, bold=True, color=TEAL)
bullet_box(s, [
    "lyricsgenius library + Genius API token",
    "6 concurrent workers (ThreadPoolExecutor)",
    "Rate-limited: 0.3s sleep between requests",
    "Resumable: progress saved to lyrics_cache.csv",
    "Cleaned: stripped contributor headers & boilerplate",
], Inches(0.7), Inches(3.35), Inches(5.3), Inches(2.5), size=13, dot_color=TEAL)

# NLP features extracted box
rect(s, Inches(6.6), Inches(2.7), Inches(6.4), Inches(3.9), DARK2, radius=True)
txt(s, "NLP Features Extracted per Song", Inches(6.8), Inches(2.8),
    Inches(6.0), Inches(0.45), size=16, bold=True, color=CORAL)
bullet_box(s, [
    "VADER sentiment: compound, pos, neg, neutral",
    "TextBlob: polarity + subjectivity",
    "Hook words: oh/yeah/hey/la/na (TikTok bait)",
    "Dance words: move/shake/groove/bounce",
    "Rhyme score, repetition, syllable rate",
    "Quotable lines ratio (2–8 words = clip-ready)",
    "MiniLM-L6-v2 embeddings (384 dims)",
], Inches(6.8), Inches(3.35), Inches(6.0), Inches(2.8), size=12, dot_color=CORAL)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — AUDIO vs LYRICS MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK); header(s, "Phase 3 — Audio Only vs Audio + Lyrics", GOLD)

# Title comparison summary
rect(s, Inches(0.5), Inches(1.3), Inches(12.3), Inches(0.9), DARK2, radius=True)
txt(s, "Adding Genius lyrics + MiniLM embeddings lifts AUC from 0.654 → 0.751  (+0.097)",
    Inches(0.7), Inches(1.45), Inches(11.9), Inches(0.55),
    size=18, bold=True, color=GOLD, align=PP_ALIGN.CENTER)

# Audio Only column
rect(s, Inches(0.5), Inches(2.4), Inches(5.8), Inches(4.4), DARK2, radius=True)
rect(s, Inches(0.5), Inches(2.4), Inches(5.8), Inches(0.55), PURPLE, radius=True)
txt(s, "Audio Only  (16 features)", Inches(0.65), Inches(2.47),
    Inches(5.5), Inches(0.42), size=15, bold=True, color=WHITE)

audio_models = [
    ("LightGBM",           "0.654", GOLD),
    ("Random Forest",      "0.653", TEAL),
    ("XGBoost",            "0.647", BLUE),
    ("Extra Trees",        "0.647", VIOLET),
    ("Logistic Regression","0.647", MID),
]
for i, (name, auc, col) in enumerate(audio_models):
    y = Inches(3.1) + i * Inches(0.58)
    rect(s, Inches(0.65), y, Inches(5.5), Inches(0.48), DARK, radius=True)
    txt(s, name, Inches(0.8), y + Inches(0.08), Inches(3.5), Inches(0.35), size=13, color=WHITE)
    txt(s, f"AUC {auc}", Inches(4.5), y + Inches(0.08), Inches(1.6), Inches(0.35),
        size=13, bold=True, color=col, align=PP_ALIGN.RIGHT)

# Audio + Lyrics column
rect(s, Inches(6.55), Inches(2.4), Inches(6.25), Inches(4.4), DARK2, radius=True)
rect(s, Inches(6.55), Inches(2.4), Inches(6.25), Inches(0.55), CORAL, radius=True)
txt(s, "Audio + Lyrics  (SelectKBest top 60)", Inches(6.7), Inches(2.47),
    Inches(5.9), Inches(0.42), size=15, bold=True, color=WHITE)

lyric_models = [
    ("Logistic Regression","0.751", GOLD,   True),
    ("LightGBM",           "0.747", TEAL,   False),
    ("Random Forest",      "0.746", BLUE,   False),
    ("XGBoost",            "0.745", VIOLET, False),
    ("Extra Trees",        "0.740", MID,    False),
]
for i, (name, auc, col, best) in enumerate(lyric_models):
    y = Inches(3.1) + i * Inches(0.58)
    bg_col = DARK2 if not best else RGBColor(0x28, 0x28, 0x45)
    rect(s, Inches(6.7), y, Inches(5.9), Inches(0.48), bg_col, radius=True)
    txt(s, ("★ " if best else "   ") + name, Inches(6.85), y + Inches(0.08),
        Inches(3.5), Inches(0.35), size=13, color=GOLD if best else WHITE)
    txt(s, f"AUC {auc}", Inches(10.8), y + Inches(0.08), Inches(1.7), Inches(0.35),
        size=13, bold=True, color=col, align=PP_ALIGN.RIGHT)

# Feature selection note
txt(s, "SelectKBest selected: 0 audio  +  5 lyrics  +  55 MiniLM embeddings  (out of 436 total features)",
    Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.4),
    size=11, color=MID, italic=True, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_pptx = "/Users/dhekha/Desktop/SUPA_Spring/TikTok_Viral_Predictor_v3.pptx"
prs.save(out_pptx)
print(f"Saved → {out_pptx}")

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
        print(f"PPTX saved → {out_pptx}")
        print("Open in PowerPoint or Keynote to view.")

"""
Generate NLP Viral Lyric Predictor — PowerPoint + PDF
Covers: NLP algorithm, training sources (TikTok + Tatiana + Annotated), results, Streamlit demo.

Run: python create_nlp_viral_pdf.py
Output: NLP_Viral_Lyric_Predictor.pptx + NLP_Viral_Lyric_Predictor.pdf
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import subprocess
import sys
from pathlib import Path

# ── Color palette ──────────────────────────────────────────────────────────────
BG_DARK    = RGBColor(0x0f, 0x17, 0x29)    # deep navy
BG_CARD    = RGBColor(0x1e, 0x29, 0x3b)    # dark slate
TEAL       = RGBColor(0x2d, 0xd4, 0xbf)    # teal accent
PURPLE     = RGBColor(0x81, 0x8c, 0xf8)    # indigo/purple
CORAL      = RGBColor(0xf8, 0x71, 0x71)    # red/coral
GOLD       = RGBColor(0xfb, 0xbf, 0x24)    # amber/gold
GREEN      = RGBColor(0x34, 0xd3, 0x99)    # emerald green
WHITE      = RGBColor(0xff, 0xff, 0xff)
LIGHT_GRAY = RGBColor(0x94, 0xa3, 0xb8)
DARK_TEXT  = RGBColor(0x1e, 0x29, 0x3b)

W  = Inches(13.33)   # widescreen 16:9
H  = Inches(7.5)


# ── Slide helper functions ─────────────────────────────────────────────────────

def set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, color: RGBColor, alpha=None):
    shape = slide.shapes.add_shape(
        1, Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             font_size=18, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txb


def add_circle(slide, cx, cy, r, color: RGBColor):
    from pptx.util import Inches
    shape = slide.shapes.add_shape(
        9,  # oval
        Inches(cx - r), Inches(cy - r),
        Inches(r * 2), Inches(r * 2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def bullet_box(slide, items, l, t, w, h, title=None,
               title_color=TEAL, item_color=WHITE, bg_color=BG_CARD,
               font_size=13, title_size=15):
    add_rect(slide, l, t, w, h, bg_color)
    y = t + 0.18
    if title:
        add_text(slide, title, l + 0.18, y, w - 0.36, 0.35,
                 font_size=title_size, bold=True, color=title_color)
        y += 0.38
    for item in items:
        add_text(slide, f"• {item}", l + 0.18, y, w - 0.36, 0.32,
                 font_size=font_size, color=item_color)
        y += 0.33


def metric_card(slide, value, label, l, t, w, h, color: RGBColor):
    add_rect(slide, l, t, w, h, color)
    add_text(slide, value, l, t + 0.08, w, 0.5,
             font_size=28, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, label, l, t + 0.52, w, 0.32,
             font_size=12, color=WHITE, align=PP_ALIGN.CENTER)


# ── Build presentation ─────────────────────────────────────────────────────────

def build_pptx(out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    blank = prs.slide_layouts[6]   # blank layout

    # ══ SLIDE 1 — Title ═══════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)

    # Decorative circles
    add_circle(s, 12.0, 1.0, 1.2, PURPLE)
    add_circle(s, 0.5,  6.8, 0.8, CORAL)
    add_circle(s, 12.8, 6.5, 1.0, TEAL)

    # Title
    add_text(s, "NLP Viral Lyric Section Predictor",
             1.0, 1.8, 10.0, 1.2, font_size=42, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)

    # Underline bar
    add_rect(s, 3.5, 3.15, 6.3, 0.06, TEAL)

    # Subtitle
    add_text(s, "Which part of a song goes viral on TikTok?",
             1.0, 3.3, 10.3, 0.6, font_size=20, color=TEAL, align=PP_ALIGN.CENTER)

    # Chips
    chips = ["NLP · 12 Signals", "3 Data Sources", "71.7% P@1", "Streamlit Dashboard"]
    chip_colors = [PURPLE, TEAL, CORAL, GOLD]
    cx = 2.0
    for chip, cc in zip(chips, chip_colors):
        add_rect(s, cx, 4.2, 2.2, 0.42, cc)
        add_text(s, chip, cx, 4.2, 2.2, 0.42, font_size=13, bold=True,
                 color=WHITE, align=PP_ALIGN.CENTER)
        cx += 2.55

    add_text(s, "SUPA Spring 2026", 0.5, 6.9, 12.0, 0.4,
             font_size=12, color=LIGHT_GRAY, align=PP_ALIGN.RIGHT)

    # ══ SLIDE 2 — Project Overview ════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "Project Overview", 0.5, 0.3, 12.0, 0.65,
             font_size=32, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 3.5, 0.06, TEAL)

    bullet_box(s, [
        "Songs go viral on TikTok via 10-30s lyric clips",
        "Creators manually hunt for the 'hook' by ear",
        "No data-driven tool predicts which section trends",
        "Hand-crafted scoring formulas miss complex patterns",
        "Goal: NLP algorithm to learn viral patterns",
    ], 0.5, 1.3, 5.8, 2.8, title="The Problem", title_color=CORAL)

    bullet_box(s, [
        "Parse pasted lyrics into sections (no Genius API needed)",
        "12 NLP signals per section (hook keywords, rhyme, emotion...)",
        "Train ML on 3 sources: TikTok songs + Tatiana + annotated",
        "Best model: LogisticReg_L1 — 71.7% P@1, AUC 0.897",
        "Paste any lyrics in Streamlit → instant ML prediction",
    ], 6.8, 1.3, 5.9, 2.8, title="Our Solution", title_color=GREEN)

    # Stats row
    for val, lbl, col, x in [
        ("180+", "Training Songs", PURPLE, 0.8),
        ("3", "Data Sources", TEAL, 3.5),
        ("12", "NLP Signals", GOLD, 6.2),
        ("71.7%", "P@1 Accuracy", CORAL, 9.0),
    ]:
        metric_card(s, val, lbl, x, 4.5, 2.2, 1.0, col)

    # ══ SLIDE 3 — Data Pipeline ════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "Data Pipeline — 3 Training Sources", 0.5, 0.3, 12.0, 0.65,
             font_size=30, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 6.5, 0.06, PURPLE)

    bullet_box(s, [
        "15 proven TikTok viral songs",
        "Espresso, APT., Cruel Summer, Blinding Lights...",
        "Lyrics fetched from Genius (training only)",
        "Label = highest NLP viral score section",
    ], 0.5, 1.3, 4.1, 2.2, title="Source A — TikTok Viral Songs", title_color=TEAL)

    bullet_box(s, [
        "Tatiana_cyanite.csv — 16 songs",
        "Cyanite.ai audio features (valence, arousal, BPM...)",
        "Lyrics fetched from Genius (training only)",
        "Label = section with highest NLP viral score",
        "Cyanite clip timestamps used for validation",
    ], 4.7, 1.3, 4.1, 2.6, title="Source B — Tatiana (Cyanite)", title_color=GOLD)

    bullet_box(s, [
        "cross_platform_analysis.csv — 10 songs",
        "Cross-platform viral data (TikTok + YouTube)",
        "Label = highest combined_viral_score section",
        "Ground truth anchors for training",
    ], 8.9, 1.3, 4.0, 2.2, title="Source C — Annotated Songs", title_color=CORAL)

    # Arrow → Combined set
    add_text(s, "→", 6.3, 3.8, 0.5, 0.5, font_size=24, color=GOLD, bold=True)

    bullet_box(s, [
        "979+  total sections",
        "180+  unique songs",
        "18.9% positive rate",
        "GroupKFold — leave-one-song-out CV",
        "class_weight='balanced' for imbalance",
    ], 4.0, 3.7, 5.0, 2.5, title="Combined Training Set", title_color=PURPLE)

    # ══ SLIDE 4 — 12 NLP Signals ══════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "12 NLP Signals per Section", 0.5, 0.3, 12.0, 0.65,
             font_size=30, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 8.0, 0.06, TEAL)

    signals = [
        ("Hook Keywords (15%)",       "love/yeah/baby/oh/wanna density",       CORAL),
        ("Cross-Song Repetition (13%)", "lines that repeat across the full song", TEAL),
        ("Internal Ngram Repeat (12%)", "oh oh oh / baby baby bigrams",          PURPLE),
        ("Emotional Extremity (10%)", "|VADER compound| — extreme = memorable", GOLD),
        ("End Rhyme Density (9%)",    "AABB/ABAB scheme detection",             GREEN),
        ("Line Conciseness (9%)",     "short punchy lines = clip-friendly",     CORAL),
        ("Syllable Rhythm (8%)",      "low syllable variance = singable",       TEAL),
        ("Phonetic Openness (7%)",    "vowel density — open sounds = beltable", PURPLE),
        ("Pronoun Relatability (7%)", "I/me/you/we — personal = shareable",    GOLD),
        ("Internal Rhyme (5%)",       "rhyming words within a single line",     GREEN),
        ("Energy Markers (3%)",       "!/? / ALL-CAPS word count",              CORAL),
        ("Song Position (2%)",        "bell curve at 40% position in song",     TEAL),
    ]

    cols = 3
    col_w = 4.2
    row_h = 0.82
    for i, (name, desc, col) in enumerate(signals):
        c = i % cols
        r = i // cols
        x = 0.4 + c * (col_w + 0.1)
        y = 1.3 + r * row_h
        add_rect(s, x, y, col_w, row_h - 0.06, BG_CARD)
        add_rect(s, x, y, 0.07, row_h - 0.06, col)
        add_text(s, name, x + 0.18, y + 0.05, col_w - 0.25, 0.32,
                 font_size=12, bold=True, color=col)
        add_text(s, desc, x + 0.18, y + 0.38, col_w - 0.25, 0.32,
                 font_size=11, color=LIGHT_GRAY)

    # ══ SLIDE 5 — Tatiana Audio Analysis ══════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "Tatiana DeMaria — Cyanite Audio Integration", 0.5, 0.3, 12.0, 0.65,
             font_size=28, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 5.0, 0.06, CORAL)

    bullet_box(s, [
        "16 songs analyzed with Cyanite.ai GraphQL API v6/v7",
        "Deep audio embeddings — valence, arousal, BPM, energy",
        "Mood tags: romantic, uplifting, sexy, chill, energetic",
        "Genre tags: folk/country, pop, R&B, rock",
        "Movement: flowing, bouncy, steady, driving, pulsing",
        "Recommended clip timestamps (15-sec TikTok clips)",
    ], 0.5, 1.3, 5.8, 3.2, title="Cyanite Features Used", title_color=TEAL)

    # Song table
    songs_data = [
        ("Beirut Fire",           "102 BPM", "romantic",           "1:30-1:45"),
        ("Anxiety",               "127 BPM", "romantic, chilled",  "0:45-1:00"),
        ("Guilt Free Heart",      "82 BPM",  "romantic, happy",    "2:30-2:45"),
        ("London Don't Lie",      "172 BPM", "uplifting, romantic","2:15-2:30"),
        ("Born Again Sinner",     "150 BPM", "sexy, happy",        "2:15-2:30"),
        ("What It Is About You",  "173 BPM", "uplifting, energetic","2:15-2:30"),
    ]

    add_text(s, "Sample Songs", 7.0, 1.2, 5.8, 0.4, font_size=14,
             bold=True, color=TEAL)
    add_rect(s, 7.0, 1.6, 6.0, 0.35, PURPLE)
    for i, (song, bpm, mood, clip) in enumerate(songs_data):
        bg = BG_CARD if i % 2 == 0 else RGBColor(0x25, 0x34, 0x4a)
        y = 1.95 + i * 0.52
        add_rect(s, 7.0, y, 6.0, 0.5, bg)
        add_text(s, song, 7.1, y + 0.08, 2.2, 0.32, font_size=11, bold=True, color=WHITE)
        add_text(s, bpm, 9.3, y + 0.08, 1.1, 0.32, font_size=11, color=GOLD)
        add_text(s, clip, 10.5, y + 0.08, 1.4, 0.32, font_size=10, color=TEAL)

    add_text(s, "NLP maps Cyanite clip timestamps → section labels for validation",
             0.5, 4.7, 12.0, 0.4, font_size=12, color=LIGHT_GRAY)

    # ══ SLIDE 6 — Models & Results ════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "Models & Validation — Leave-One-Song-Out CV", 0.5, 0.3, 12.0, 0.65,
             font_size=28, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 6.0, 0.06, GOLD)

    bullet_box(s, [
        "GroupKFold(n_splits=180) — one song held out each fold",
        "Remaining 179 songs used for training",
        "No leakage: song sections never split across folds",
        "Metric: can model predict a brand-new song's viral section?",
    ], 0.5, 1.3, 5.5, 2.0, title="Leave-One-Song-Out CV", title_color=PURPLE)

    # Results table
    results = [
        ("★ LogisticReg_L1", "71.7%", "0.837", "0.897", GOLD),
        ("LogisticReg (L2)", "71.1%", "0.833", "0.894", GREEN),
        ("Stacking (XGB+LGBM+ET)", "69.4%", "0.820", "0.882", TEAL),
        ("Voting (LR+L1+HGB)", "68.9%", "0.824", "0.886", TEAL),
        ("XGBoost", "65.6%", "0.803", "0.870", PURPLE),
        ("LightGBM", "65.6%", "0.799", "0.869", PURPLE),
        ("HGB (HistGradBoost)", "65.6%", "0.802", "0.864", PURPLE),
        ("ExtraTrees (500)", "63.9%", "0.788", "0.857", LIGHT_GRAY),
        ("Random Forest", "63.3%", "0.790", "0.867", LIGHT_GRAY),
        ("MLP (128→64→32)", "57.2%", "0.731", "0.791", CORAL),
    ]

    header_y = 1.3
    add_rect(s, 6.3, header_y, 6.6, 0.4, PURPLE)
    for hdr, xp in [("Model", 6.4), ("P@1", 9.6), ("MRR", 10.5), ("AUC", 11.4)]:
        add_text(s, hdr, xp, header_y + 0.05, 1.0, 0.28, font_size=11,
                 bold=True, color=WHITE)

    for i, (name, p1, mrr, auc, col) in enumerate(results):
        y = header_y + 0.42 + i * 0.52
        bg = RGBColor(0x2a, 0x3a, 0x52) if i % 2 == 0 else BG_CARD
        add_rect(s, 6.3, y, 6.6, 0.5, bg)
        if i == 0:
            add_rect(s, 6.3, y, 0.07, 0.5, GOLD)
        add_text(s, name, 6.4, y + 0.08, 3.1, 0.32, font_size=11,
                 bold=(i == 0), color=col if i == 0 else WHITE)
        add_text(s, p1,  9.6,  y + 0.08, 0.8, 0.32, font_size=11, bold=(i==0), color=col if i==0 else WHITE)
        add_text(s, mrr, 10.5, y + 0.08, 0.8, 0.32, font_size=11, color=WHITE)
        add_text(s, auc, 11.4, y + 0.08, 0.8, 0.32, font_size=11, color=WHITE)

    # ══ SLIDE 7 — Example: Espresso ═══════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, 'Example: "Espresso" by Sabrina Carpenter', 0.5, 0.3, 12.0, 0.65,
             font_size=28, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 5.5, 0.06, CORAL)

    sections_data = [
        ("[Intro]",       0.675, PURPLE),
        ("[Verse 1]",     0.395, TEAL),
        ("[Pre-Chorus]",  0.602, GREEN),
        ("[Chorus]",      0.716, CORAL),
        ("[Post-Chorus]", 0.201, LIGHT_GRAY),
        ("[Verse 2]",     0.376, TEAL),
        ("[Outro]",       0.435, PURPLE),
    ]

    add_text(s, "NLP Viral Score per Section:", 0.5, 1.3, 5.5, 0.4,
             font_size=14, bold=True, color=TEAL)
    bar_max_w = 4.0
    for i, (sec, score, col) in enumerate(sections_data):
        y = 1.75 + i * 0.62
        is_top = sec == "[Chorus]"
        add_text(s, sec, 0.5, y, 2.0, 0.35,
                 font_size=12, bold=is_top,
                 color=CORAL if is_top else WHITE)
        bar_w = score * bar_max_w
        add_rect(s, 2.6, y + 0.04, bar_w, 0.28, col)
        if is_top:
            add_text(s, "← #1 VIRAL", 2.6 + bar_w + 0.1, y, 1.8, 0.32,
                     font_size=11, bold=True, color=CORAL)
        add_text(s, f"{score:.0%}", 2.6 + bar_w + 0.08, y, 0.6, 0.32,
                 font_size=11, color=GOLD if is_top else LIGHT_GRAY)

    bullet_box(s, [
        "hook_vocabulary = 1.000   dense 'baby', 'know', 'yeah', 'wanna'",
        "cross_song_rep  = 0.833   chorus lines repeat 3× in raw lyrics",
        "internal_rep    = 1.000   'that's that me' / 'say you can't sleep'",
        "sentiment       = 0.979   strong upbeat positive emotion",
        "line_conciseness = 0.869  short 6-8 word lines — clip-friendly",
        "position_score  = 0.882   positioned at 43% — chorus zone",
        "",
        "\"That's that me espresso\" → exact TikTok viral line ✓",
    ], 6.6, 1.3, 6.3, 5.5, title="Why [Chorus] Scores Highest", title_color=GOLD)

    # ══ SLIDE 8 — Streamlit Dashboard ═════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "Streamlit Dashboard — 5 Tabs", 0.5, 0.3, 12.0, 0.65,
             font_size=30, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 4.0, 0.06, TEAL)

    tabs_info = [
        ("🔥 TikTok Trending",  "Live trending songs via RapidAPI. Bar chart of plays & likes.", CORAL),
        ("🎤 Lyric Sections",   "Genius annotation + text score breakdown per section.", PURPLE),
        ("🤖 NLP Detector",     "12-signal NLP scoring on any song. Radar chart + signal table.", GOLD),
        ("🎸 Tatiana's Songs",  "Cyanite valence/arousal scatter + per-song NLP predictions.", TEAL),
        ("📊 Model Prediction", "Paste lyrics → all model results + NLP signal breakdown.", GREEN),
    ]

    for i, (tab_name, desc, col) in enumerate(tabs_info):
        x = 0.4 + i * 2.58
        add_rect(s, x, 1.4, 2.3, 0.95, col)
        add_text(s, tab_name, x, 1.4, 2.3, 0.95,
                 font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    for i, (tab_name, desc, col) in enumerate(tabs_info):
        x = 0.4 + i * 2.58
        add_rect(s, x, 2.5, 2.3, 1.6, BG_CARD)
        add_rect(s, x, 2.5, 2.3, 0.04, col)
        add_text(s, desc, x + 0.1, 2.58, 2.1, 1.4, font_size=11, color=LIGHT_GRAY)

    bullet_box(s, [
        "Tab 5 — Paste any lyrics with [Chorus]/[Verse] headers → predict instantly",
        "12 NLP signals computed locally — no Genius API call at inference time",
        "Bar chart: viral probability per section + NLP signal breakdown",
        "Tatiana tab: Cyanite valence vs arousal scatter + clip recommendations",
        "Trained on TikTok songs + Tatiana/Cyanite + annotated cross-platform songs",
    ], 0.5, 4.3, 8.5, 2.5, title="Model Prediction Tab — No API Needed", title_color=GREEN)

    add_text(s, "streamlit run viral_lyrics_analyzer/dashboard_tatiana.py",
             0.5, 6.8, 12.5, 0.4, font_size=13, color=TEAL, bold=True)

    # ══ SLIDE 9 — Key Takeaways ════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    set_bg(s, BG_DARK)
    add_text(s, "Key Takeaways", 0.5, 0.3, 12.0, 0.65,
             font_size=32, bold=True, color=WHITE)
    add_rect(s, 0.5, 1.05, 3.0, 0.06, GOLD)

    takeaways = [
        ("71.7% P@1",         "LogisticReg_L1 picks the viral section 3.8× better than random (18.9% baseline)", TEAL),
        ("12 NLP Signals",    "hook vocab + cross-song repetition + emotional intensity drive predictions", GOLD),
        ("3 Data Sources",    "TikTok viral songs + Tatiana (Cyanite) + annotated cross-platform songs", CORAL),
        ("Linear > Complex",  "With limited data, L1 Logistic Regression beats XGBoost, LightGBM & Neural Nets", PURPLE),
        ("AUC 0.897",         "Excellent class separation — hook keywords, repetition & emotion capture structure", GREEN),
        ("No API Needed",     "Paste lyrics in Streamlit → ML predicts viral section instantly, no Genius API", TEAL),
    ]

    for i, (title_txt, desc, col) in enumerate(takeaways):
        c = i % 3
        r = i // 3
        x = 0.4 + c * 4.3
        y = 1.4 + r * 2.3
        add_rect(s, x, y, 4.1, 2.1, col)
        add_text(s, title_txt, x, y + 0.1, 4.1, 0.55,
                 font_size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_rect(s, x, y + 0.6, 4.1, 1.5, BG_CARD)
        add_text(s, desc, x + 0.12, y + 0.68, 3.85, 1.3,
                 font_size=11, color=LIGHT_GRAY)

    add_text(s, "SUPA Spring 2026  ·  NLP Viral Lyric Section Predictor",
             0.5, 7.0, 12.0, 0.35, font_size=11, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

    prs.save(str(out_path))
    print(f"  Saved PPTX → {out_path}")


def convert_to_pdf(pptx_path: Path) -> Path:
    """Convert PPTX to PDF using LibreOffice (if available)."""
    pdf_path = pptx_path.with_suffix(".pdf")
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(pptx_path.parent), str(pptx_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and pdf_path.exists():
            print(f"  Saved PDF  → {pdf_path}")
            return pdf_path
        else:
            print(f"  LibreOffice error: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  LibreOffice not found — trying pptx2pdf...")
    except Exception as e:
        print(f"  LibreOffice exception: {e}")

    # Fallback: try pptx2pdf
    try:
        from pptx2pdf import convert
        convert(str(pptx_path), str(pdf_path))
        if pdf_path.exists():
            print(f"  Saved PDF  → {pdf_path}")
            return pdf_path
    except ImportError:
        pass
    except Exception as e:
        print(f"  pptx2pdf error: {e}")

    # Fallback: try python-docx2pdf approach via subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"from pptx2pdf import convert; convert('{pptx_path}', '{pdf_path}')"],
            capture_output=True, text=True, timeout=60,
        )
        if pdf_path.exists():
            print(f"  Saved PDF  → {pdf_path}")
            return pdf_path
    except Exception:
        pass

    print("  ⚠ PDF conversion failed. PPTX saved — open in PowerPoint/Google Slides to export PDF.")
    print("  Or install LibreOffice: brew install libreoffice")
    return pptx_path


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    pptx_path = out_dir / "NLP_Viral_Lyric_Predictor.pptx"

    print("Building NLP Viral Lyric Predictor presentation...")
    build_pptx(pptx_path)

    print("Converting to PDF...")
    final = convert_to_pdf(pptx_path)

    print(f"\nDone!")
    print(f"  PPTX: {pptx_path}")
    if final.suffix == ".pdf":
        print(f"  PDF:  {final}")

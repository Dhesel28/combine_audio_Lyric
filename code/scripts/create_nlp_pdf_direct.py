"""
Direct PDF generator for NLP Viral Lyric Predictor presentation.
Uses reportlab — no LibreOffice needed.

Run: python create_nlp_pdf_direct.py
Output: NLP_Viral_Lyric_Predictor.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# ── Dimensions ─────────────────────────────────────────────────────────────────
PAGE = landscape(A4)   # 841.89 x 595.27 pt
W, H = PAGE

# ── Colours ────────────────────────────────────────────────────────────────────
BG_DARK    = HexColor("#0f1729")
BG_CARD    = HexColor("#1e293b")
BG_CARD2   = HexColor("#253449")
TEAL       = HexColor("#2dd4bf")
PURPLE     = HexColor("#818cf8")
CORAL      = HexColor("#f87171")
GOLD       = HexColor("#fbbf24")
GREEN      = HexColor("#34d399")
WHITE      = HexColor("#ffffff")
LGRAY      = HexColor("#94a3b8")
DGRAY      = HexColor("#334155")


# ── Drawing helpers ────────────────────────────────────────────────────────────

def bg(c: canvas.Canvas):
    c.setFillColor(BG_DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)


def rect(c, x, y, w, h, fill_color, radius=6):
    c.setFillColor(fill_color)
    c.setStrokeColor(fill_color)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=0)


def text(c, txt, x, y, size=14, color=WHITE, bold=False, align="left", max_width=None):
    c.setFillColor(color)
    if bold:
        c.setFont("Helvetica-Bold", size)
    else:
        c.setFont("Helvetica", size)

    if align == "center" and max_width:
        tw = c.stringWidth(txt, "Helvetica-Bold" if bold else "Helvetica", size)
        x = x + (max_width - tw) / 2
    elif align == "right" and max_width:
        tw = c.stringWidth(txt, "Helvetica-Bold" if bold else "Helvetica", size)
        x = x + max_width - tw

    c.drawString(x, y, txt)


def wrap_text(c, txt, x, y, max_width, size=12, color=WHITE, bold=False, line_height=16):
    """Draw text with word wrapping. Returns y after last line."""
    c.setFillColor(color)
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    words = txt.split()
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, font, size) <= max_width:
            line = test
        else:
            if line:
                c.drawString(x, y, line)
                y -= line_height
            line = word
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y


def card(c, x, y, w, h, fill=BG_CARD, stroke_color=None, stroke_w=2):
    rect(c, x, y, w, h, fill)
    if stroke_color:
        c.setStrokeColor(stroke_color)
        c.setLineWidth(stroke_w)
        c.roundRect(x, y, w, h, 6, fill=0, stroke=1)


def side_accent(c, x, y, h, color, width=5):
    rect(c, x, y, width, h, color)


def metric_box(c, x, y, w, h, value, label, color):
    rect(c, x, y, w, h, color, radius=8)
    vw = c.stringWidth(value, "Helvetica-Bold", 28)
    c.setFont("Helvetica-Bold", 28)
    c.setFillColor(WHITE)
    c.drawString(x + (w - vw) / 2, y + h - 42, value)
    lw = c.stringWidth(label, "Helvetica", 11)
    c.setFont("Helvetica", 11)
    c.drawString(x + (w - lw) / 2, y + 10, label)


def page_title(c, title, accent_color=TEAL, subtitle=None):
    text(c, title, 30, H - 60, size=28, bold=True)
    rect(c, 30, H - 75, 280, 4, accent_color)
    if subtitle:
        text(c, subtitle, 30, H - 92, size=13, color=LGRAY)


def footer(c, slide_num, total=9):
    text(c, f"SUPA Spring 2026  ·  NLP Viral Lyric Predictor  ·  {slide_num}/{total}",
         W/2 - 150, 18, size=10, color=LGRAY)


def next_slide(c):
    c.showPage()
    bg(c)


# ── SLIDE 1 — Title ────────────────────────────────────────────────────────────
def slide_title(c):
    bg(c)
    # Decorative circles
    c.setFillColor(PURPLE)
    c.circle(W - 80, H - 60, 80, fill=1, stroke=0)
    c.setFillColor(CORAL)
    c.circle(40, 60, 55, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.circle(W - 60, 60, 65, fill=1, stroke=0)

    # Title
    title_str = "NLP Viral Lyric Section Predictor"
    tw = c.stringWidth(title_str, "Helvetica-Bold", 38)
    c.setFont("Helvetica-Bold", 38)
    c.setFillColor(WHITE)
    c.drawString((W - tw) / 2, H - 160, title_str)

    # Underline
    rect(c, (W - 400) / 2, H - 178, 400, 4, TEAL)

    # Subtitle
    sub = "Which part of a song goes viral on TikTok?"
    sw = c.stringWidth(sub, "Helvetica", 18)
    c.setFont("Helvetica", 18)
    c.setFillColor(TEAL)
    c.drawString((W - sw) / 2, H - 210, sub)

    # Chip pills
    chips = [("NLP · 12 Signals", PURPLE), ("3 Data Sources", TEAL),
             ("71.7% P@1", CORAL), ("Streamlit App", GOLD)]
    total_w = len(chips) * 160 + (len(chips) - 1) * 15
    cx = (W - total_w) / 2
    for label, col in chips:
        rect(c, cx, H - 270, 160, 32, col, radius=16)
        lw = c.stringWidth(label, "Helvetica-Bold", 12)
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(WHITE)
        c.drawString(cx + (160 - lw) / 2, H - 252, label)
        cx += 175

    footer(c, 1)


# ── SLIDE 2 — Project Overview ─────────────────────────────────────────────────
def slide_overview(c):
    page_title(c, "Project Overview", TEAL)
    footer(c, 2)

    # Problem card
    card(c, 30, 80, 375, 380, BG_CARD)
    side_accent(c, 30, 80, 380, CORAL)
    text(c, "The Problem", 45, 440, size=14, color=CORAL, bold=True)
    problems = [
        "Songs go viral on TikTok via 10-30s lyric clips",
        "Creators manually hunt for the 'hook' by ear",
        "No data-driven tool predicts viral sections",
        "Hand-crafted formulas miss complex patterns",
        "Goal: ML to learn viral patterns from data",
    ]
    py = 415
    for p in problems:
        text(c, f"• {p}", 50, py, size=11, color=WHITE)
        py -= 22

    # Solution card
    card(c, 435, 80, 376, 380, BG_CARD)
    side_accent(c, 435, 80, 380, GREEN)
    text(c, "Our Solution", 450, 440, size=14, color=GREEN, bold=True)
    solutions = [
        "Parse any song into sections (with [Label] headers)",
        "12 NLP signals computed per section locally",
        "Train ML: TikTok songs + Tatiana/Cyanite + annotated",
        "Best: LogisticReg_L1 — no Genius API at inference",
        "AUC-ROC 0.897 — excellent class separation",
        "Paste lyrics in Streamlit → instant prediction",
    ]
    sy = 415
    for s in solutions:
        text(c, f"• {s}", 450, sy, size=11, color=WHITE)
        sy -= 22

    # Metric boxes
    metrics = [
        ("180+", "Songs", PURPLE, 30),
        ("979+", "Sections", TEAL, 230),
        ("12", "NLP Signals", GOLD, 430),
        ("71.7%", "P@1 Accuracy", CORAL, 630),
    ]
    for val, lbl, col, x in metrics:
        metric_box(c, x, 20, 175, 55, val, lbl, col)


# ── SLIDE 3 — Data Sources ─────────────────────────────────────────────────────
def slide_data(c):
    page_title(c, "Data Pipeline — 3 Training Sources", PURPLE)
    footer(c, 3)

    sources = [
        ("Source A — TikTok Viral Songs", TEAL, [
            "15 proven TikTok viral songs",
            "Espresso, APT., Cruel Summer...",
            "Blinding Lights, As It Was, Luther",
            "Lyrics fetched from Genius (training only)",
            "Label = highest NLP viral score section",
        ]),
        ("Source B — Tatiana (Cyanite)", GOLD, [
            "Tatiana_cyanite.csv — 16 songs",
            "Cyanite.ai audio features per song",
            "valence, arousal, BPM, mood tags",
            "Label = highest NLP viral score",
            "Cyanite clip timestamps for validation",
        ]),
        ("Source C — Annotated Songs", CORAL, [
            "cross_platform_analysis.csv",
            "10 songs with combined viral scores",
            "Cross-platform data (TikTok+YouTube)",
            "Label = highest combined_viral_score",
            "Ground truth anchors for training",
        ]),
    ]

    for i, (title, col, items) in enumerate(sources):
        x = 30 + i * 270
        card(c, x, 130, 255, 310, BG_CARD)
        side_accent(c, x, 130, 310, col)
        text(c, title, x + 15, 415, size=12, color=col, bold=True)
        iy = 390
        for item in items:
            wrap_text(c, f"• {item}", x + 15, iy, 230, size=10, color=WHITE, line_height=16)
            iy -= 22

    # Arrow
    text(c, "→", 840, 285, size=28, color=GOLD, bold=True)

    # Combined box
    card(c, 870, 150, 240, 280, BG_CARD)
    side_accent(c, 870, 150, 280, PURPLE)
    text(c, "Combined Set", 885, 400, size=13, color=PURPLE, bold=True)
    combined = [
        "979+ total sections",
        "180+ unique songs",
        "18.9% positive rate",
        "GroupKFold CV",
        "class_weight=balanced",
    ]
    cy = 375
    for item in combined:
        text(c, f"• {item}", 885, cy, size=11, color=WHITE)
        cy -= 25

    # Small metric row
    for val, lbl, col, x in [
        ("979+", "Sections", TEAL, 30),
        ("180+", "Songs", PURPLE, 230),
        ("18.9%", "Positive Rate", CORAL, 430),
        ("3", "Sources", GOLD, 630),
    ]:
        metric_box(c, x, 20, 175, 55, val, lbl, col)


# ── SLIDE 4 — 12 NLP Signals ──────────────────────────────────────────────────
def slide_signals(c):
    page_title(c, "12 NLP Signals per Section", TEAL,
               subtitle="Weights sum to 1.0  ·  derived from Random Forest feature importances")
    footer(c, 4)

    signals = [
        ("Hook Keywords (15%)",          "love/yeah/baby/oh density",          CORAL),
        ("Cross-Song Repetition (13%)",   "lines repeating across full song",   TEAL),
        ("Internal N-gram Repeat (12%)",  "oh oh oh / baby baby bigrams",       PURPLE),
        ("Emotional Extremity (10%)",     "|VADER compound| — extreme emotion", GOLD),
        ("End Rhyme Density (9%)",        "AABB / ABAB rhyme detection",        GREEN),
        ("Line Conciseness (9%)",         "short punchy lines = clip-friendly", CORAL),
        ("Syllable Rhythm (8%)",          "low syllable variance = singable",   TEAL),
        ("Phonetic Openness (7%)",        "vowel density — open = beltable",    PURPLE),
        ("Pronoun Relatability (7%)",     "I/me/you/we — personal = shareable", GOLD),
        ("Internal Rhyme (5%)",           "rhyming words within one line",      GREEN),
        ("Energy Markers (3%)",           "!/? / ALL-CAPS word count",          CORAL),
        ("Song Position (2%)",            "bell curve at 40% position",         TEAL),
    ]

    cols = 3
    cell_w = 260
    cell_h = 66
    x_start = 30
    y_start = H - 120

    for i, (name, desc, col) in enumerate(signals):
        col_i = i % cols
        row_i = i // cols
        x = x_start + col_i * (cell_w + 18)
        y = y_start - row_i * (cell_h + 8)

        card(c, x, y - cell_h + 10, cell_w, cell_h, BG_CARD)
        side_accent(c, x, y - cell_h + 10, cell_h, col, width=5)
        text(c, name, x + 14, y - 10, size=11, color=col, bold=True)
        text(c, desc, x + 14, y - 28, size=10, color=LGRAY)


# ── SLIDE 5 — Tatiana Analysis ─────────────────────────────────────────────────
def slide_tatiana(c):
    page_title(c, "Tatiana DeMaria — Cyanite Audio Integration", CORAL)
    footer(c, 5)

    # Left: features
    card(c, 30, 80, 370, 380, BG_CARD)
    side_accent(c, 30, 80, 380, TEAL)
    text(c, "Cyanite.ai Audio Features", 48, 435, size=13, color=TEAL, bold=True)
    feats = [
        "16 songs analyzed via Cyanite.ai GraphQL API",
        "Valence: emotional positivity (-1 to +1)",
        "Arousal: energy/excitement level",
        "Tempo BPM: rhythmic pace",
        "Energy level: low / medium / high / variable",
        "Mood tags: romantic, uplifting, sexy, chill",
        "Genre tags: folk, pop, R&B, rock",
        "Recommended clip timestamps (15-sec clips)",
        "NLP maps clip times → section labels",
    ]
    fy = 412
    for f in feats:
        text(c, f"• {f}", 48, fy, size=10, color=WHITE)
        fy -= 24

    # Right: song table
    text(c, "Tatiana's Songs — Audio Profile", 430, H - 100, size=13,
         color=GOLD, bold=True)

    songs = [
        ("Beirut Fire",          "102", "+0.33", "-0.09", "romantic"),
        ("Anxiety",              "127", "+0.40", "-0.11", "romantic, chilled"),
        ("Guilt Free Heart",     " 82", "+0.58", "-0.04", "romantic, happy"),
        ("London Don't Lie",     "172", "+0.44", "+0.02", "uplifting, romantic"),
        ("You Make Me",          "147", "+0.06", "-0.39", "calm, romantic"),
        ("Too Much",             "127", "+0.32", "+0.14", "uplifting, sexy"),
        ("Born Again Sinner",    "150", "+0.67", "+0.16", "sexy, happy"),
        ("What It Is About You", "173", "+0.54", "+0.38", "uplifting, energetic"),
    ]

    hx = 430
    # Header
    rect(c, hx, H - 130, 400, 24, PURPLE)
    for hdr, hxo in [("Song", 6), ("BPM", 192), ("Val", 232), ("Aro", 272), ("Mood", 312)]:
        text(c, hdr, hx + hxo, H - 120, size=10, bold=True)

    for i, (song, bpm, val, aro, mood) in enumerate(songs):
        y = H - 155 - i * 32
        bg_c = BG_CARD if i % 2 == 0 else BG_CARD2
        rect(c, hx, y, 400, 30, bg_c)
        text(c, song[:22], hx + 6, y + 9, size=10, color=WHITE)
        text(c, bpm,  hx + 192, y + 9, size=10, color=GOLD)
        text(c, val,  hx + 232, y + 9, size=10, color=TEAL if "+" in val else CORAL)
        text(c, aro,  hx + 272, y + 9, size=10, color=TEAL if "+" in aro else CORAL)
        text(c, mood[:18], hx + 312, y + 9, size=9, color=LGRAY)

    text(c, "NLP predicts the viral section for each song in the Streamlit dashboard",
         30, 55, size=11, color=LGRAY)


# ── SLIDE 6 — Model Results ────────────────────────────────────────────────────
def slide_results(c):
    page_title(c, "Model Results — Leave-One-Song-Out CV (180 songs)", GOLD)
    footer(c, 6)

    # Left: CV method
    card(c, 30, 80, 350, 370, BG_CARD)
    side_accent(c, 30, 80, 370, PURPLE)
    text(c, "Leave-One-Song-Out CV", 48, 425, size=13, color=PURPLE, bold=True)
    cv_pts = [
        "GroupKFold(n_splits=180)",
        "One full song held out as test set",
        "179 songs used for training",
        "No leakage — sections never split",
        "180 evaluation folds",
        "Metric: predict brand-new song's viral section",
    ]
    cvy = 400
    for p in cv_pts:
        text(c, f"• {p}", 48, cvy, size=11, color=WHITE)
        cvy -= 26

    # Right: results table
    text(c, "10 Models Evaluated", 410, H - 100, size=13, color=WHITE, bold=True)

    models = [
        ("★ LogisticReg_L1 (BEST)", "71.7%", "0.837", "0.897", GOLD, True),
        ("LogisticReg (L2)",         "71.1%", "0.833", "0.894", GREEN, False),
        ("Stacking (XGB+LGBM+ET)",  "69.4%", "0.820", "0.882", TEAL, False),
        ("Voting (LR+L1+HGB)",       "68.9%", "0.824", "0.886", TEAL, False),
        ("XGBoost (300 est.)",       "65.6%", "0.803", "0.870", PURPLE, False),
        ("LightGBM (300 est.)",      "65.6%", "0.799", "0.869", PURPLE, False),
        ("HGB (HistGradBoost)",      "65.6%", "0.802", "0.864", PURPLE, False),
        ("ExtraTrees (500)",         "63.9%", "0.788", "0.857", LGRAY, False),
        ("RandomForest (500)",       "63.3%", "0.790", "0.867", LGRAY, False),
        ("MLP (128→64→32)",          "57.2%", "0.731", "0.791", CORAL, False),
    ]

    hx = 400
    rect(c, hx, H - 128, 420, 26, PURPLE)
    for hdr, xo in [("Model", 8), ("P@1", 248), ("MRR", 298), ("AUC", 348)]:
        text(c, hdr, hx + xo, H - 117, size=10, bold=True)

    for i, (name, p1, mrr, auc, col, is_best) in enumerate(models):
        y = H - 155 - i * 36
        bg_c = BG_CARD if i % 2 == 0 else BG_CARD2
        rect(c, hx, y, 420, 34, bg_c)
        if is_best:
            side_accent(c, hx, y, 34, GOLD, width=5)
        text(c, name[:32], hx + 10, y + 10, size=10,
             color=col, bold=is_best)
        text(c, p1,  hx + 248, y + 10, size=10, color=col if is_best else WHITE, bold=is_best)
        text(c, mrr, hx + 298, y + 10, size=10, color=WHITE)
        text(c, auc, hx + 348, y + 10, size=10, color=WHITE)


# ── SLIDE 7 — Example: Espresso ───────────────────────────────────────────────
def slide_espresso(c):
    page_title(c, '"Espresso" by Sabrina Carpenter — Example', CORAL)
    footer(c, 7)

    # Section bars
    sections = [
        ("[Intro]",       0.675, PURPLE),
        ("[Verse 1]",     0.395, TEAL),
        ("[Pre-Chorus]",  0.602, GREEN),
        ("[Chorus]",      0.716, CORAL),
        ("[Post-Chorus]", 0.201, LGRAY),
        ("[Verse 2]",     0.376, TEAL),
        ("[Outro]",       0.435, PURPLE),
    ]

    text(c, "NLP Viral Score per Section:", 30, H - 100, size=13, color=TEAL, bold=True)
    bar_max = 260
    for i, (sec, score, col) in enumerate(sections):
        y = H - 140 - i * 50
        is_top = sec == "[Chorus]"
        text(c, sec, 30, y, size=11, color=CORAL if is_top else WHITE, bold=is_top)
        bar_w = int(score * bar_max)
        rect(c, 145, y - 4, bar_w, 22, col)
        text(c, f"{score:.0%}", 145 + bar_w + 8, y, size=10,
             color=GOLD if is_top else LGRAY, bold=is_top)
        if is_top:
            text(c, "← VIRAL", 145 + bar_w + 60, y, size=10, color=CORAL, bold=True)

    # Why card
    card(c, 440, 80, 400, 400, BG_CARD)
    side_accent(c, 440, 80, 400, GOLD)
    text(c, "Why [Chorus] Scores #1:", 455, 455, size=13, color=GOLD, bold=True)
    reasons = [
        ("hook_vocabulary",      "1.000", "'baby','know','yeah','wanna'"),
        ("internal_repetition",  "1.000", "'that's that me' repeats 4×"),
        ("cross_song_rep",       "0.833", "chorus lines appear 3× in raw"),
        ("sentiment_extremity",  "0.979", "strong upbeat positive emotion"),
        ("line_conciseness",     "0.869", "short 6-8 word lines"),
        ("position_score",       "0.882", "43% position — chorus zone"),
    ]
    ry = 430
    for feat, val, insight in reasons:
        rect(c, 455, ry - 18, 375, 22, BG_CARD2)
        text(c, feat, 460, ry - 8, size=10, color=TEAL)
        text(c, val,  630, ry - 8, size=10, color=GOLD, bold=True)
        text(c, insight, 685, ry - 8, size=9, color=LGRAY)
        ry -= 32

    text(c, '"That\'s that me espresso" → exact TikTok viral line ✓',
         455, 130, size=11, color=GREEN, bold=True)

    text(c, "NLP Score: 0.716  (vs random baseline 18.9% = 3.8× improvement)",
         30, 50, size=11, color=LGRAY)


# ── SLIDE 8 — Streamlit Dashboard ─────────────────────────────────────────────
def slide_streamlit(c):
    page_title(c, "Streamlit Dashboard — 5 Tabs", TEAL)
    footer(c, 8)

    tabs = [
        ("🔥 TikTok Trending",  "Live trending songs via RapidAPI\nBar chart of plays & likes", CORAL),
        ("🎤 Lyric Sections",   "Genius annotation score\nper section for saved songs", PURPLE),
        ("🤖 NLP Detector",     "12-signal NLP on any song\nRadar + signal breakdown table", GOLD),
        ("🎸 Tatiana's Songs",  "Cyanite valence/arousal scatter\n+ per-song NLP predictions", TEAL),
        ("📊 Model Prediction", "Paste lyrics → ML prediction\nno Genius API required", GREEN),
    ]

    tab_w = 150
    for i, (tab, desc, col) in enumerate(tabs):
        x = 30 + i * 162
        rect(c, x, H - 170, tab_w, 60, col, radius=8)
        lines = tab.split(" ", 1)
        text(c, lines[0] if len(lines) > 1 else tab, x + 6, H - 130, size=13, bold=True)
        if len(lines) > 1:
            text(c, lines[1][:15], x + 6, H - 148, size=10)

    for i, (tab, desc, col) in enumerate(tabs):
        x = 30 + i * 162
        card(c, x, H - 340, tab_w, 160, BG_CARD)
        side_accent(c, x, H - 340, 160, col, width=4)
        dy = H - 195
        for line in desc.split("\n"):
            wrap_text(c, line, x + 12, dy, tab_w - 20, size=10, color=LGRAY, line_height=15)
            dy -= 30

    # Detail box
    card(c, 30, 80, 780, 180, BG_CARD)
    side_accent(c, 30, 80, 180, GREEN)
    text(c, "Tab 5 — Model Prediction (no Genius API needed):", 48, 240, size=12, color=GREEN, bold=True)
    detail = [
        "Paste any song lyrics with [Chorus] / [Verse 1] headers → instant prediction",
        "12 NLP signals computed locally — zero API calls at inference time",
        "Bar chart: viral probability per section + NLP signal breakdown chart",
        "Trained on TikTok songs + Tatiana/Cyanite + annotated cross-platform songs",
    ]
    dy = 215
    for d in detail:
        text(c, f"• {d}", 48, dy, size=10, color=WHITE)
        dy -= 22

    code = "streamlit run viral_lyrics_analyzer/dashboard_tatiana.py"
    rect(c, 30, 48, 580, 28, DGRAY, radius=4)
    text(c, code, 42, 58, size=11, color=TEAL, bold=True)


# ── SLIDE 9 — Key Takeaways ────────────────────────────────────────────────────
def slide_takeaways(c):
    page_title(c, "Key Takeaways", GOLD)
    footer(c, 9)

    takeaways = [
        ("71.7% P@1",         "LogisticReg_L1 picks the viral section\n3.8× better than random (18.9% baseline)", TEAL),
        ("12 NLP Signals",    "hook vocab + cross-song repetition\n+ emotional intensity drive predictions", GOLD),
        ("3 Data Sources",    "TikTok songs + Tatiana (Cyanite)\n+ annotated cross-platform songs", CORAL),
        ("Linear > Complex",  "With limited data, L1 beats XGBoost,\nLightGBM & Neural Nets", PURPLE),
        ("AUC 0.897",         "Excellent class separation\n— hook + repetition + emotion work", GREEN),
        ("No API Needed",     "Paste lyrics in Streamlit → ML predicts\ninstantly without Genius API", TEAL),
    ]

    cols = 3
    bw = 265
    bh = 155
    for i, (title, desc, col) in enumerate(takeaways):
        ci = i % cols
        ri = i // cols
        x = 22 + ci * (bw + 18)
        y = H - 130 - ri * (bh + 14)

        rect(c, x, y, bw, 38, col, radius=8)
        tw = c.stringWidth(title, "Helvetica-Bold", 18)
        c.setFont("Helvetica-Bold", 18)
        c.setFillColor(WHITE)
        c.drawString(x + (bw - tw) / 2, y + 12, title)

        card(c, x, y - (bh - 38), bw, bh - 38, BG_CARD)
        dy = y - 25
        for line in desc.split("\n"):
            wrap_text(c, line, x + 12, dy, bw - 24, size=11, color=WHITE, line_height=18)
            dy -= 22

    text(c, "Next step: real TikTok view counts + Spotify audio features → push P@1 above 80%",
         30, 25, size=11, color=LGRAY)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    out = Path(__file__).parent / "NLP_Viral_Lyric_Predictor.pdf"
    c = canvas.Canvas(str(out), pagesize=PAGE)

    slides = [
        (slide_title,      "Title"),
        (slide_overview,   "Overview"),
        (slide_data,       "Data Sources"),
        (slide_signals,    "NLP Signals"),
        (slide_tatiana,    "Tatiana"),
        (slide_results,    "Results"),
        (slide_espresso,   "Espresso Example"),
        (slide_streamlit,  "Streamlit"),
        (slide_takeaways,  "Takeaways"),
    ]

    for i, (fn, name) in enumerate(slides, 1):
        fn(c)
        if i < len(slides):
            next_slide(c)
        print(f"  Slide {i}: {name}")

    c.save()
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    print("Generating NLP Viral Lyric Predictor PDF...")
    main()

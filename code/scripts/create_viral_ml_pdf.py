"""
Generate a professional PDF presentation for the ML Viral Lyric Section Predictor.
Uses ReportLab for direct PDF generation.
Updated: v2 — 38 features, 10 models, LogisticReg_L1 best (71.7% P@1)
"""

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas

W, H = landscape(A4)  # 841.9 x 595.3 pts

# ── Palette ────────────────────────────────────────────────────────────────────
PURPLE  = HexColor("#6C5CE7")
VIOLET  = HexColor("#A29BFA")
TEAL    = HexColor("#00B894")
CORAL   = HexColor("#FF6B6B")
GOLD    = HexColor("#FDCB6E")
DARK    = HexColor("#1E1E2E")
DARK2   = HexColor("#2D2D44")
DARK3   = HexColor("#252538")
MID     = HexColor("#AAA8C8")
GREEN   = HexColor("#55EFC4")
WHITE_C = HexColor("#FFFFFF")
DARK_GREEN = HexColor("#1A2A1E")


def new_slide(c, title_text=None, title_color=TEAL):
    c.showPage()
    c.setPageSize(landscape(A4))
    c.setFillColor(DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(DARK2)
    c.rect(0, H - 72, W, 72, fill=1, stroke=0)
    if title_text:
        c.setFillColor(WHITE_C)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(36, H - 48, title_text)
        c.setFillColor(title_color)
        c.rect(0, H - 73, W, 3, fill=1, stroke=0)


def card(c, x, y, w, h, color=DARK2, radius=6):
    c.setFillColor(color)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=0)


def label(c, text, x, y, size=13, color=WHITE_C, bold=False, align="left"):
    c.setFillColor(color)
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def multiline(c, text, x, y, max_width, size=12, color=WHITE_C, bold=False, line_h=None):
    line_h = line_h or size * 1.5
    c.setFillColor(color)
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    words = text.split()
    line = ""
    current_y = y
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, font, size) <= max_width:
            line = test
        else:
            if line:
                c.drawString(x, current_y, line)
                current_y -= line_h
            line = word
    if line:
        c.drawString(x, current_y, line)
    return current_y


def bullet_list(c, items, x, y, max_width, size=12, color=WHITE_C,
                dot_color=TEAL, line_h=20, indent=16):
    cy = y
    for item in items:
        c.setFillColor(dot_color)
        c.circle(x + 4, cy + size * 0.3, 3, fill=1, stroke=0)
        multiline(c, item, x + indent, cy, max_width - indent,
                  size=size, color=color, line_h=line_h)
        cy -= line_h
    return cy


def hbar(c, x, y, full_w, h, value, max_val, fill_color, bg_color=DARK3):
    c.setFillColor(bg_color)
    c.roundRect(x, y, full_w, h, 3, fill=1, stroke=0)
    filled = full_w * value / max_val
    if filled > 0:
        c.setFillColor(fill_color)
        c.roundRect(x, y, filled, h, 3, fill=1, stroke=0)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title
# ─────────────────────────────────────────────────────────────────────────────
c = canvas.Canvas(
    "/Users/dhekha/Desktop/SUPA_Spring/ML_Viral_Lyric_Predictor.pdf",
    pagesize=landscape(A4)
)
c.setPageSize(landscape(A4))

c.setFillColor(DARK)
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(DARK2)
c.rect(0, 0, W * 0.52, H, fill=1, stroke=0)

for cx2, cy2, r2, col in [
    (W - 70, H - 40, 80, PURPLE),
    (W - 110, 50, 55, TEAL),
    (30, 55, 35, CORAL),
]:
    c.setFillColor(col)
    c.circle(cx2, cy2, r2, fill=1, stroke=0)

c.setFillColor(WHITE_C)
c.setFont("Helvetica-Bold", 34)
c.drawString(45, H - 120, "ML Viral Lyric Section Predictor")

c.setFillColor(TEAL)
c.rect(45, H - 145, 360, 3, fill=1, stroke=0)

c.setFillColor(VIOLET)
c.setFont("Helvetica", 18)
c.drawString(45, H - 175, "Which part of a song goes viral on TikTok?")

c.setFillColor(MID)
c.setFont("Helvetica-Oblique", 11)
c.drawString(45, H - 205, "Machine Learning  ·  180 Songs  ·  38 Features  ·  10 Models  ·  Leave-One-Song-Out CV")

pills = [("180", "Songs", PURPLE), ("979", "Sections", TEAL),
         ("38",  "Features", CORAL), ("71.7%", "P@1 Accuracy", GOLD)]
for i, (val, lbl, col) in enumerate(pills):
    px = 45 + i * 150
    py = H - 320
    c.setFillColor(col)
    c.roundRect(px, py, 135, 70, 8, fill=1, stroke=0)
    c.setFillColor(WHITE_C)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(px + 67, py + 40, val)
    c.setFont("Helvetica", 10)
    c.drawCentredString(px + 67, py + 18, lbl)

c.setFillColor(MID)
c.setFont("Helvetica-Oblique", 10)
c.drawRightString(W - 36, 25, "SUPA Spring 2026")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Project Overview
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Project Overview", TEAL)

card(c, 28, 60, W/2 - 42, H - 130)
label(c, "The Problem", 44, H - 115, size=15, color=TEAL, bold=True)
bullet_list(c, [
    "Songs go viral on TikTok via specific 10-30s lyric clips",
    "Creators manually hunt for the 'hook' by ear",
    "No data-driven tool predicts which section will trend",
    "Hand-crafted scoring formulas miss complex patterns",
    "Goal: train ML to learn viral patterns from song data",
], 36, H - 145, W/2 - 60, size=12, dot_color=TEAL, line_h=22)

card(c, W/2 + 14, 60, W/2 - 42, H - 130)
label(c, "Our Solution", W/2 + 30, H - 115, size=15, color=CORAL, bold=True)
bullet_list(c, [
    "Parse any song's lyrics into sections via Genius API",
    "Extract 38 engineered features per section (+6 new)",
    "Train 10 ML models with leave-one-song-out validation",
    "Best: LogisticReg_L1 — 71.7% P@1 (L1 regularisation)",
    "Live Streamlit dashboard for predictions on any song",
    "AUC-ROC of 0.897 — excellent class separation",
], W/2 + 22, H - 145, W/2 - 60, size=12, dot_color=CORAL, line_h=22)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Data Pipeline — 2 Sources → 979 Labeled Sections", PURPLE)

card(c, 28, H - 260, W * 0.44, 175)
label(c, "Source A — Valentine Songs Dataset (173 songs)", 44, H - 110, size=13, color=PURPLE, bold=True)
bullet_list(c, [
    "valentine_songs_full_dataset.csv",
    "Full lyrics column (scraped from Genius)",
    "Label = section with highest text_virality_score",
    "Yields 906 labeled sections",
], 36, H - 138, W * 0.44 - 20, size=11, dot_color=PURPLE, line_h=19)

card(c, 28, H - 455, W * 0.44, 175)
label(c, "Source B — Annotated Songs (10 songs)", 44, H - 305, size=13, color=TEAL, bold=True)
bullet_list(c, [
    "cross_platform_analysis.csv",
    "Genius annotation scores + YouTube Shorts data",
    "Label = section with highest combined_viral_score",
    "Yields 73 labeled sections",
], 36, H - 333, W * 0.44 - 20, size=11, dot_color=TEAL, line_h=19)

c.setFillColor(GOLD)
c.setFont("Helvetica-Bold", 28)
c.drawCentredString(W * 0.52, H / 2, "→")

card(c, W * 0.55, H - 480, W * 0.42, 395)
label(c, "Combined Training Set", W * 0.55 + 16, H - 115, size=15, color=GOLD, bold=True)

stats = [("979", "total sections", WHITE_C),
         ("180", "unique songs",   WHITE_C),
         ("185", "positive labels", CORAL),
         ("18.9%", "positive rate", MID)]
sy = H - 150
for val, lbl, col in stats:
    c.setFillColor(col)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(W * 0.55 + 16, sy, val)
    c.setFillColor(MID)
    c.setFont("Helvetica", 11)
    c.drawString(W * 0.55 + 16 + c.stringWidth(val, "Helvetica-Bold", 22) + 8, sy + 2, lbl)
    sy -= 32

c.setFillColor(TEAL)
c.rect(W * 0.55 + 16, sy - 5, W * 0.38, 2, fill=1, stroke=0)
sy -= 20

bullet_list(c, [
    "GroupKFold(n_splits=180) — each fold holds out one full song",
    "Prevents leakage: song sections never split across folds",
    "Imbalance handled: class_weight='balanced' or scale_pos_weight",
], W * 0.55 + 8, sy, W * 0.42 - 20, size=11, dot_color=GOLD, line_h=19)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Feature Engineering (38 features)
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Feature Engineering — 38 Features per Section  (+6 New)", CORAL)

groups = [
    ("Text Scores (4)",         CORAL,   "repeat_score · short_line_score · rhyme_score · hook_keyword_score"),
    ("Position (4)",            PURPLE,  "section_index · normalized_position · is_first · is_last"),
    ("Section Type (7)",        TEAL,    "chorus/hook · verse · bridge · pre-chorus · intro · outro · other"),
    ("VADER Sentiment (4)",     GOLD,    "compound · positive · negative · neutral"),
    ("TextBlob (2)",            VIOLET,  "polarity · subjectivity"),
    ("Linguistic (4)",          GREEN,   "pronoun_density · question_count · exclamation_count · cap_word_ratio"),
    ("Singability (2)",         CORAL,   "vowel_density · avg_syllables_per_word"),
    ("Complexity (2)",          MID,     "unique_word_ratio · avg_word_length"),
    ("Structural (2)",          PURPLE,  "has_repetition_pattern · call_response_pattern"),
    ("Readability (1)",         TEAL,    "flesch_reading_ease  (pyphen syllable counts)"),
    ("★ NEW: Structure (2)",    GREEN,   "word_count · line_count"),
    ("★ NEW: Hook & Emotion (4)", GOLD,  "emotional_intensity · ends_strong · starts_strong · hook_word_variety"),
]

cols = 2
per_col = 6
box_w = (W - 60) / 2 - 8
box_h = (H - 100) / per_col - 5

for idx, (gname, col, features) in enumerate(groups):
    ci = idx // per_col
    ri = idx % per_col
    bx = 28 + ci * (box_w + 16)
    by = H - 100 - (ri + 1) * (box_h + 5)

    bg_col = DARK_GREEN if "NEW" in gname else DARK2
    card(c, bx, by, box_w, box_h, bg_col)
    c.setFillColor(col)
    c.roundRect(bx, by, 8, box_h, 3, fill=1, stroke=0)
    label(c, gname, bx + 18, by + box_h - 19, size=11, color=col, bold=True)
    label(c, features, bx + 18, by + 8, size=8.5, color=MID)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Feature Importances (updated)
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Top Feature Importances  (Random Forest, 38 Features)", GOLD)

top_features = [
    ("hook_keyword_score",     0.0879, GOLD,   False),
    ("unique_word_ratio",      0.0843, GOLD,   False),
    ("repeat_score",           0.0788, CORAL,  False),
    ("has_repetition_pattern", 0.0531, PURPLE, False),
    ("line_count",             0.0489, GREEN,  True),
    ("emotional_intensity",    0.0472, GREEN,  True),
    ("rhyme_score",            0.0441, TEAL,   False),
    ("short_line_score",       0.0424, CORAL,  False),
    ("vowel_density",          0.0383, TEAL,   False),
    ("word_count",             0.0365, GREEN,  True),
    ("flesch_reading_ease",    0.0365, TEAL,   False),
    ("avg_word_length",        0.0317, MID,    False),
]

bar_l = 220
bar_full = W - bar_l - 130
bar_h2 = 22
row_h = 37
start_y = H - 100

for i, (feat, val, col, is_new) in enumerate(top_features):
    by = start_y - i * row_h
    label(c, feat, 28, by + 5, size=11, color=WHITE_C)
    hbar(c, bar_l, by, bar_full, bar_h2, val, 0.095, col)
    label(c, f"{val*100:.1f}%", bar_l + bar_full + 8, by + 5,
          size=11, color=col, bold=True)
    if is_new:
        label(c, "NEW", bar_l + bar_full + 65, by + 5, size=9, color=GREEN, bold=True)

c.setFillColor(MID)
c.setFont("Helvetica-Oblique", 10)
c.drawString(28, 35,
    "3 new features enter the top-10: line_count, emotional_intensity, word_count "
    "— section length & absolute emotion are key viral signals.")


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Models & Validation (10 models)
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Models & Validation Strategy  —  10 Models", VIOLET)

card(c, 28, 55, W * 0.40, H - 125)
label(c, "Leave-One-Song-Out CV", 44, H - 110, size=14, color=VIOLET, bold=True)
bullet_list(c, [
    "GroupKFold(n_splits=180)",
    "All sections from one song held out as test set",
    "Remaining 179 songs used for training",
    "No leakage: song sections never split across folds",
    "180 evaluation folds — rigorous real-world test",
    "Metric: can model predict on a brand-new song?",
], 36, H - 145, W * 0.40 - 20, size=12, dot_color=VIOLET, line_h=22)

rx = W * 0.43 + 14
card(c, rx, 55, W * 0.55, H - 125)
label(c, "10 Models Evaluated", rx + 16, H - 110, size=14, color=TEAL, bold=True)

model_defs = [
    ("LogisticReg_L1  ★ BEST",        "L1 penalty — auto feature selection, C=0.5",      TEAL),
    ("Logistic Regression",            "L2 baseline · class_weight='balanced'",            VIOLET),
    ("Stacking (XGB+LGBM+ET → LR)",   "ensemble meta-learner · 5-fold inner CV",          GOLD),
    ("Voting (LR + L1 + HGB)",        "soft-vote of top 3 linear/gradient models",        GREEN),
    ("XGBoost (300 est.)",             "scale_pos_weight · min_child_weight=3 · gamma=0.1", GOLD),
    ("LightGBM (300 est.)",            "num_leaves=20 · reg_alpha/lambda=0.1",             GREEN),
    ("HGB (HistGradientBoosting)",     "sklearn native GBDT · small-data friendly",        TEAL),
    ("Random Forest (500 trees)",      "max_features='sqrt' · balanced class_weight",      PURPLE),
    ("Extra Trees (500 trees)",        "more randomisation than RF",                       VIOLET),
    ("MLP 128→64→32",                  "early stopping · StandardScaler · alpha=0.001",    CORAL),
]
my = H - 148
row_gap = (H - 185) / len(model_defs)
for name, note, col in model_defs:
    rh = row_gap - 2
    card(c, rx + 10, my - rh + 4, W * 0.55 - 20, rh, DARK3)
    c.setFillColor(col)
    c.roundRect(rx + 10, my - rh + 4, 6, rh, 2, fill=1, stroke=0)
    is_best = "BEST" in name
    label(c, name, rx + 26, my - rh + rh * 0.6, size=10,
          color=TEAL if is_best else WHITE_C, bold=is_best)
    label(c, note, rx + 26, my - rh + rh * 0.2, size=8.5, color=MID)
    my -= row_gap


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Results Table (10 models, updated numbers)
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Model Results — Leave-One-Song-Out CV  (180 songs)", CORAL)

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

for hname, hx in [("Model", 28), ("P@1  (↑ best)", 290),
                   ("MRR  (↑ best)", 510), ("AUC-ROC  (↑ best)", 700)]:
    label(c, hname, hx, H - 95, size=12, color=MID, bold=True)

c.setFillColor(MID)
c.rect(28, H - 102, W - 56, 1, fill=1, stroke=0)

row_h2 = (H - 148) / len(results)
for i, (name, p1, mrr, auc, col, best) in enumerate(results):
    ry = H - 110 - (i + 1) * row_h2
    bg_col = DARK_GREEN if best else DARK2
    card(c, 28, ry, W - 56, row_h2 - 3, bg_col)

    if best:
        c.setFillColor(TEAL)
        c.roundRect(28, ry, 6, row_h2 - 3, 2, fill=1, stroke=0)
        label(c, "★ BEST", 38, ry + row_h2 * 0.55, size=8, color=TEAL, bold=True)

    label(c, name, 80, ry + row_h2 * 0.35, size=11,
          color=TEAL if best else WHITE_C, bold=best)

    hbar(c, 290, ry + 5, 190, row_h2 - 14, p1, 1.0, col)
    label(c, f"{p1:.1%}", 488, ry + row_h2 * 0.35, size=10, color=col, bold=best)

    hbar(c, 520, ry + 5, 155, row_h2 - 14, mrr, 1.0, col)
    label(c, f"{mrr:.3f}", 683, ry + row_h2 * 0.35, size=10, color=col, bold=best)

    hbar(c, 720, ry + 5, 155, row_h2 - 14, auc, 1.0, col)
    label(c, f"{auc:.3f}", 883, ry + row_h2 * 0.35, size=10, color=col, bold=best)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Metrics Explained (updated numbers)
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Understanding the Metrics", GOLD)

metrics = [
    ("P@1  —  Precision at Rank 1", TEAL,
     "Did the model pick the correct\nviral section as #1?",
     ["Score = 1 if top prediction = true viral section",
      "Score = 0 otherwise",
      "Our best: 71.7%  →  right ~5 out of 7 songs",
      "Random baseline ≈ 18.9%  (positive rate)",
      "3.8× improvement over random guessing"]),
    ("MRR  —  Mean Reciprocal Rank", PURPLE,
     "How high up is the correct\nsection in the ranked list?",
     ["MRR = 1/rank of first correct prediction",
      "MRR=1.0 means always ranked #1",
      "Our best: 0.837  →  usually #1 or #2",
      "Confirms model ranks viral sections near top",
      "not just accidentally correct"]),
    ("AUC-ROC  —  Area Under Curve", CORAL,
     "Can the model separate viral\nfrom non-viral sections?",
     ["AUC=1.0 perfect  ·  AUC=0.5 random",
      "Our best: 0.897  →  excellent separation",
      "Robust to class imbalance (18.9% positive rate)",
      "Meaningful across all 180 songs",
      "Near 0.90 is strong for real-world NLP"]),
]

card_w = (W - 56) / 3 - 8
for i, (title, col, question, bullets) in enumerate(metrics):
    cx2 = 28 + i * (card_w + 12)
    card(c, cx2, 55, card_w, H - 125, DARK2)
    c.setFillColor(col)
    c.roundRect(cx2, H - 130, card_w, 35, 4, fill=1, stroke=0)
    label(c, title, cx2 + 10, H - 118, size=12, color=WHITE_C, bold=True)

    label(c, question.split("\n")[0], cx2 + 10, H - 160, size=11, color=GOLD, bold=True)
    if "\n" in question:
        label(c, question.split("\n")[1], cx2 + 10, H - 177, size=11, color=GOLD, bold=True)

    bullet_list(c, bullets, cx2 + 6, H - 210, card_w - 12,
                size=11, dot_color=col, line_h=20)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — Why LogisticReg_L1 Wins
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Why Does LogisticReg_L1 Win?", TEAL)

card(c, 28, 55, W * 0.44, H - 125)
label(c, "The Bias-Variance Tradeoff", 44, H - 110, size=14, color=TEAL, bold=True)
bullet_list(c, [
    "180 songs is a small dataset for complex models",
    "Tree ensembles overfit to training song patterns",
    "Neural nets (MLP) need far more data to generalise",
    "LR + StandardScaler finds a robust linear boundary",
    "L1 (Lasso) penalty zeroes noisy features — built-in",
    "  feature selection across 38 dimensions",
    "class_weight='balanced' handles 18.9% imbalance cleanly",
    "C=0.5 regularisation prevents overfitting effectively",
], 36, H - 145, W * 0.44 - 20, size=12, dot_color=TEAL, line_h=21)

rx = W * 0.47
card(c, rx, H - 290, W * 0.51, 210)
label(c, "L2 LogisticReg close behind at 71.1%", rx + 16, H - 110, size=13, color=GOLD, bold=True)
bullet_list(c, [
    "Only 0.6% gap: L1 vs L2 is a minor difference here",
    "Both beat all tree/ensemble models by wide margin",
    "Stacking (69.4%) and Voting (68.9%) trail by ~2.5%",
    "Linear signal dominates — features are well-engineered",
], rx + 8, H - 140, W * 0.51 - 20, size=12, dot_color=GOLD, line_h=22)

card(c, rx, 55, W * 0.51, H - 310)
label(c, "Next steps to improve further", rx + 16, H - 310 + 75 + 55 - 28,
      size=14, color=CORAL, bold=True)
bullet_list(c, [
    "Real TikTok view counts as ground-truth labels",
    "Audio features: BPM, energy, danceability (Spotify API)",
    "BERT/sentence embeddings for semantic meaning",
    "Fine-tune on platform-specific virality signals",
], rx + 8, H - 310 + 75 + 55 - 58, W * 0.51 - 20, size=12, dot_color=CORAL, line_h=22)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "The Streamlit Dashboard — 5 Tabs", GREEN)

tabs = [
    ("TikTok Trending",   CORAL,  "Live TikTok trending\nsongs via RapidAPI.\nBar chart of plays & likes."),
    ("Lyric Sections",    PURPLE, "Viral score breakdown\nby section for saved songs.\nProgress bars per section."),
    ("Cross-Platform",    TEAL,   "TikTok vs YouTube\nShorts scatter.\nGenius annotations vs YT matches."),
    ("Analyze Any Song",  GOLD,   "Live Genius fetch.\nRadar chart + feature\nbreakdown + lyrics preview."),
    ("Model Prediction",  GREEN,  "ML viral probability\nbars + feature importances\n+ predicted lyrics."),
]
tw = (W - 56) / 5 - 6
for i, (name, col, desc) in enumerate(tabs):
    tx = 28 + i * (tw + 6)
    card(c, tx, H - 190, tw, 110, col)
    label(c, name, tx + tw/2, H - 122, size=10, color=WHITE_C, bold=True, align="center")

    card(c, tx, H - 375, tw, 172, DARK2)
    dy = H - 220
    for line in desc.split("\n"):
        label(c, line, tx + 8, dy, size=10, color=WHITE_C)
        dy -= 16

card(c, 28, 30, W - 56, 115, DARK2)
label(c, "Tab 5 — Model Prediction Detail:", 44, 120, size=13, color=GREEN, bold=True)
detail_items = [
    "Enter any song → Genius API fetches & parses sections automatically",
    "38 features extracted per section → LogisticReg_L1 model scores each section",
    "Bar chart: viral probability per section  |  Top-15 feature importances (RF)",
    "Scatter: ML probability vs Genius annotation score  |  Predicted lyrics shown",
]
dy2 = 103
for item in detail_items:
    c.setFillColor(GREEN)
    c.circle(44, dy2 + 4, 3, fill=1, stroke=0)
    label(c, item, 58, dy2, size=10, color=WHITE_C)
    dy2 -= 18


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Example: Espresso
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, 'Example: "Espresso" by Sabrina Carpenter', CORAL)

card(c, 28, 55, W * 0.42, H - 125)
label(c, "Model scores each section:", 44, H - 110, size=13, color=CORAL, bold=True)

sections_ex = [
    ("[Intro]",           0.12, MID,    False),
    ("[Verse 1]",         0.28, TEAL,   False),
    ("[Pre-Chorus]",      0.51, VIOLET, False),
    ("[Chorus]  <- #1 *", 0.88, CORAL,  True),
    ("[Post-Chorus]",     0.62, GOLD,   False),
    ("[Verse 2]",         0.30, TEAL,   False),
    ("[Bridge]",          0.43, PURPLE, False),
    ("[Outro]",           0.19, MID,    False),
]
bar_l3 = 165; bar_fw3 = W * 0.42 - 185
sy2 = H - 138
for sec, prob, col, best in sections_ex:
    label(c, sec, 36, sy2 + 4, size=11, color=CORAL if best else WHITE_C, bold=best)
    hbar(c, bar_l3, sy2, bar_fw3, 16, prob, 1.0, col)
    label(c, f"{prob:.0%}", bar_l3 + bar_fw3 + 6, sy2 + 4, size=10, color=col, bold=best)
    sy2 -= 27

rx = W * 0.45 + 14
card(c, rx, 55, W * 0.53, H - 125)
label(c, "Why [Chorus] scores highest:", rx + 16, H - 110, size=13, color=TEAL, bold=True)
bullet_list(c, [
    "hook_keyword_score HIGH: 'sweet', 'baby', 'feel', 'yeah'",
    "hook_word_variety HIGH: 5 unique hook words (NEW feature)",
    "repeat_score HIGH: chorus lines repeat 4x across the song",
    "stype_chorus_hook = 1  (strongest categorical signal)",
    "emotional_intensity HIGH: strong absolute sentiment (NEW)",
    "vader_compound: positive upbeat emotional sentiment",
    "short_line_score: punchy 6-8 word lines — easily singable",
    "has_repetition_pattern = 1 — memorable structure detected",
    "flesch_reading_ease: very easy — sing-along friendly",
], rx + 8, H - 145, W * 0.53 - 20, size=11, dot_color=TEAL, line_h=20)

c.setFillColor(MID)
c.setFont("Helvetica-Oblique", 10)
c.drawString(rx + 16, 65,
    '"And now he\'s thinkin\' \'bout me every night, oh / Is it that sweet? I guess so..."')


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12 — Key Takeaways (updated)
# ─────────────────────────────────────────────────────────────────────────────
new_slide(c, "Key Takeaways", PURPLE)

takeaways = [
    (TEAL,   "71.7% P@1",
     "LogisticReg_L1 correctly picks the viral section 71.7% of the time — 3.8x better than random (18.9%)"),
    (CORAL,  "38 Features Work",
     "6 new features added: line_count, emotional_intensity, word_count, ends_strong, starts_strong, hook_word_variety"),
    (PURPLE, "Linear > Complex",
     "With 180 songs, L1 Logistic Regression beats XGBoost, LightGBM & Neural Nets — features matter more than model complexity"),
    (GOLD,   "AUC 0.897",
     "Excellent discrimination — hook keywords, repetition & emotional intensity capture real viral lyric structure"),
    (GREEN,  "Live & Deployable",
     "Streamlit dashboard runs real-time inference on any song via Genius API — usable as songwriter / A&R tool"),
    (VIOLET, "Next: Better Labels",
     "Real TikTok view counts + Spotify audio features could push P@1 above 80% with the same architecture"),
]

card_w2 = (W - 56) / 3 - 8
card_h2 = (H - 125) / 2 - 8
for i, (col, title, body) in enumerate(takeaways):
    ci = i % 3; ri = i // 3
    bx = 28 + ci * (card_w2 + 12)
    by = H - 120 - (ri + 1) * (card_h2 + 8)
    card(c, bx, by, card_w2, card_h2, DARK2)
    c.setFillColor(col)
    c.roundRect(bx, by + card_h2 - 32, card_w2, 32, 4, fill=1, stroke=0)
    label(c, title, bx + 10, by + card_h2 - 20, size=14, color=WHITE_C, bold=True)
    multiline(c, body, bx + 10, by + card_h2 - 55, card_w2 - 20,
              size=11, color=WHITE_C, line_h=17)


# ── Save ───────────────────────────────────────────────────────────────────────
c.save()
print("PDF saved -> /Users/dhekha/Desktop/SUPA_Spring/ML_Viral_Lyric_Predictor.pdf")

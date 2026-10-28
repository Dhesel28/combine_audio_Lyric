"""
Segmentation Model — Minimalistic PowerPoint
Covers: what was built, how it works, results, and what's next.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path
from config import DATA_DIR

OUT = DATA_DIR / "Segmentation_Model.pptx"

WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BLACK  = RGBColor(0x11, 0x11, 0x11)
GRAY   = RGBColor(0x88, 0x88, 0x88)
LGRAY  = RGBColor(0xF2, 0xF2, 0xF2)
DGRAY  = RGBColor(0x33, 0x33, 0x33)

def _bg(slide):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = WHITE

def _box(slide, l, t, w, h, color=LGRAY):
    s = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    return s

def _t(slide, text, l, t, w, h, size=12, bold=False,
       color=BLACK, align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    r  = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.color.rgb = color; r.font.italic = italic
    return tb

def _rule(slide, l, t, w, color=LGRAY, h=0.018):
    _box(slide, l, t, w, h, color)

def _footer(slide, note=""):
    _rule(slide, 0.5, 7.05, 9.0)
    _t(slide, note or "Lyric Segmentation  ·  TikTok Viral Detector",
       0.5, 7.15, 9.0, 0.3, size=8, color=GRAY)


# ── Slide 1: Title ─────────────────────────────────────────────────────────────
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _box(s, 0.5, 2.4, 0.06, 1.6, BLACK)
    _t(s, "Lyric Segmentation\nModel", 0.75, 2.3, 8.5, 1.4, size=38, bold=True)
    _t(s, "Detecting Chorus · Verse · Bridge from lyrics alone",
       0.75, 3.85, 8.5, 0.45, size=14, color=GRAY)
    _rule(s, 0.75, 4.5, 8.3)
    _t(s, "Three models built  ·  4,088 training blocks  ·  80.1% accuracy",
       0.75, 4.65, 8.3, 0.35, size=10, color=GRAY)
    _footer(s)


# ── Slide 2: The Problem ───────────────────────────────────────────────────────
def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "The Problem", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    _t(s, "TikTok clips are 15–60 seconds. Most go viral from one specific section.",
       0.5, 1.25, 9.0, 0.5, size=13, color=DGRAY)
    _t(s, "But raw lyrics have no timestamps — just text.",
       0.5, 1.75, 9.0, 0.4, size=13, color=GRAY, italic=True)

    _rule(s, 0.5, 2.35, 9.0)

    # Three challenges
    challenges = [
        ("01", "Not all lyrics are labeled",
         "Only 67% of songs have [Chorus] / [Verse] markers from Genius. "
         "The rest is raw text with no section boundaries."),
        ("02", "Labels alone are not enough",
         "Of 464 songs, 0 had BOTH [Chorus] AND [Verse] in the same song. "
         "Hip-hop/rap songs use different labeling conventions."),
        ("03", "Audio tells the full story",
         "Section boundaries are defined by melody and beat changes — "
         "not just lyrics. Lyrics alone are an approximation."),
    ]
    for i, (num, head, body) in enumerate(challenges):
        y = 2.55 + i * 1.35
        _box(s, 0.5, y, 9.0, 1.2, LGRAY)
        _t(s, num,  0.72, y + 0.3, 0.6, 0.5,  size=18, bold=True, color=GRAY)
        _t(s, head, 1.4,  y + 0.18, 7.5, 0.4, size=13, bold=True)
        _t(s, body, 1.4,  y + 0.6,  7.5, 0.5, size=10, color=GRAY)

    _footer(s)


# ── Slide 3: What Was Built ────────────────────────────────────────────────────
def slide_what_was_built(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "What Was Built", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    models = [
        ("Lyric Segmentor",
         "Rule-based + Heuristic",
         "Parses Genius section markers. For songs without markers, "
         "finds the most repeated block of lines and calls it the chorus. "
         "Scores each section for TikTok viral potential.",
         "lyric_segmentor.py"),
        ("Song Segmentation Model",
         "K-Means Clustering  (k=2)",
         "Clusters full songs by 18 lyrical features — energy, sentiment, "
         "repetition rate, hook ratio, vocab richness, 7 emotional themes. "
         "Found two clusters: High-Energy/Party and Love & Romance.",
         "segmentation_model.py"),
        ("Section Detector",
         "Gradient Boosting  (ML)",
         "Trains on 4,088 labeled blocks from 500+ Genius songs. "
         "Learns what each section TYPE looks like from 12 features. "
         "Predicts Chorus / Verse / Bridge / Intro / Outro for any lyrics.",
         "section_detector.py"),
    ]
    for i, (name, method, desc, file) in enumerate(models):
        y = 1.25 + i * 1.8
        _box(s, 0.5, y, 9.0, 1.65, LGRAY)
        _t(s, name,   0.75, y + 0.15, 5.0, 0.4,  size=14, bold=True)
        _t(s, method, 6.0,  y + 0.18, 3.2, 0.35, size=10, color=GRAY,
           align=PP_ALIGN.RIGHT, italic=True)
        _t(s, desc,   0.75, y + 0.6,  7.5, 0.7,  size=10, color=DGRAY)
        _t(s, file,   0.75, y + 1.3,  4.0, 0.25, size=8,  color=GRAY, italic=True)

    _footer(s)


# ── Slide 4: How the Section Detector Works ───────────────────────────────────
def slide_how_it_works(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Section Detector  —  How It Works", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    # Pipeline steps
    steps = [
        ("Split into blocks",
         "Lyrics are divided into blocks — either by [Section] markers or by blank lines"),
        ("Extract 12 features",
         "Per block: repetition count, similarity to other blocks, position in song, "
         "length, line length, rhyme density, energy, sentiment, brevity"),
        ("Classify",
         "Gradient Boosting predicts: Chorus · Verse · Bridge · Intro · Outro"),
    ]
    for i, (step, detail) in enumerate(steps):
        y = 1.2 + i * 0.88
        num_x = 0.5
        _t(s, f"0{i+1}", num_x, y + 0.08, 0.5, 0.45, size=20, bold=True, color=GRAY)
        _t(s, step,   1.1, y,        7.5, 0.38, size=13, bold=True)
        _t(s, detail, 1.1, y + 0.4,  7.5, 0.38, size=10, color=GRAY)
        if i < 2:
            _rule(s, 0.5, y + 0.83, 9.0)

    _rule(s, 0.5, 3.9, 9.0)

    # Feature importances
    _t(s, "WHAT THE MODEL RELIES ON MOST", 0.5, 4.05, 9.0, 0.3, size=8, bold=True, color=GRAY)

    features = [
        ("Similarity to other blocks",  0.372, "Chorus blocks are similar to themselves when repeated"),
        ("Position in song",            0.336, "Intro = early  ·  Outro = late  ·  Chorus = middle"),
        ("Block length",                0.085, "Chorus blocks tend to be shorter and punchier"),
        ("Avg line length",             0.059, "Shorter lines signal chorus or hook"),
        ("Unique line ratio",           0.043, "Low uniqueness = high repetition = chorus"),
    ]
    col_x = [0.5, 3.6, 5.5]
    for i, (name, imp, desc) in enumerate(features):
        y = 4.4 + i * 0.48
        if i % 2 == 0:
            _box(s, 0.48, y - 0.03, 9.1, 0.46, LGRAY)
        bar_w  = imp * 1.5
        bar_bg = _box(s, col_x[1], y + 0.07, 1.8, 0.25, RGBColor(0xE0,0xE0,0xE0))
        _box(s, col_x[1], y + 0.07, bar_w, 0.25, BLACK)
        _t(s, name,          col_x[0], y + 0.07, 2.9, 0.32, size=10)
        _t(s, f"{imp:.0%}",  col_x[1] + 1.85, y + 0.07, 0.6, 0.32, size=10, bold=True,
           align=PP_ALIGN.RIGHT)
        _t(s, desc,          col_x[2], y + 0.07, 3.9, 0.32, size=9, color=GRAY)

    _footer(s)


# ── Slide 5: Results ───────────────────────────────────────────────────────────
def slide_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _t(s, "Results", 0.5, 0.4, 9.0, 0.6, size=26, bold=True)
    _rule(s, 0.5, 1.05, 9.0, BLACK, 0.015)

    # Big accuracy stat
    _box(s, 0.5, 1.2, 4.2, 1.6, LGRAY)
    _t(s, "80.1%", 0.75, 1.28, 3.5, 0.9, size=46, bold=True)
    _t(s, "5-fold cross-validation accuracy", 0.75, 2.1, 3.5, 0.4, size=10, color=GRAY)
    _t(s, "on 4,088 blocks from 500+ songs", 0.75, 2.45, 3.5, 0.3, size=9, color=GRAY)

    # Per-class accuracy
    _box(s, 4.9, 1.2, 4.6, 1.6, LGRAY)
    _t(s, "PER CLASS", 5.1, 1.28, 4.0, 0.28, size=8, bold=True, color=GRAY)
    classes = [
        ("Chorus",  "98%"),
        ("Verse",   "97%"),
        ("Bridge",  "96%"),
        ("Intro",   "95%"),
        ("Outro",   "100%"),
    ]
    for i, (lbl, acc) in enumerate(classes):
        x = 5.1 + (i % 3) * 1.5
        y = 1.62 + (i // 3) * 0.52
        _t(s, lbl, x,       y,        0.9, 0.3, size=9,  color=GRAY)
        _t(s, acc, x + 0.9, y,        0.5, 0.3, size=10, bold=True, align=PP_ALIGN.RIGHT)

    _rule(s, 0.5, 2.98, 9.0)

    # Example output
    _t(s, "EXAMPLE  —  Say My Name  (Destiny's Child)",
       0.5, 3.12, 9.0, 0.3, size=8, bold=True, color=GRAY)

    blocks = [
        ("Block 01", "INTRO",   "100%", "(Darkchild '99)"),
        ("Block 02", "CHORUS",  " 64%", "Say my name, say my name…"),
        ("Block 03", "VERSE",   " 95%", "Any other day I would call, you would say…"),
        ("Block 04", "INTRO",   " 74%", "I know you say that I am assuming things…"),
        ("Block 05", "CHORUS",  " 98%", "Say my name, say my name (Say my name)…"),
        ("Block 06", "VERSE",   " 99%", "What is up with this? Tell the truth…"),
        ("Block 08", "CHORUS",  " 96%", "Say my name, say my name (Say my name)…"),
        ("Block 09", "BRIDGE",  " 85%", "Oh, yeah (Where my ladies at?)…"),
    ]
    col_x = [0.5, 1.55, 3.0, 4.15]
    col_w = [0.9, 1.3,  0.9, 5.1]
    hdrs  = ["Block", "Label", "Conf.", "Top line"]
    for j, (h, x, w) in enumerate(zip(hdrs, col_x, col_w)):
        al = PP_ALIGN.RIGHT if j == 2 else PP_ALIGN.LEFT
        _t(s, h, x, 3.44, w, 0.28, size=8, bold=True, color=GRAY, align=al)
    _rule(s, 0.5, 3.72, 9.2, BLACK, 0.01)

    for i, (blk, lbl, conf, line) in enumerate(blocks):
        y = 3.78 + i * 0.42
        if i % 2 == 0:
            _box(s, 0.48, y - 0.02, 9.12, 0.40, LGRAY)
        _t(s, blk,  col_x[0], y + 0.04, col_w[0], 0.32, size=9, color=GRAY)
        _t(s, lbl,  col_x[1], y + 0.04, col_w[1], 0.32, size=9, bold=True)
        _t(s, conf, col_x[2], y + 0.04, col_w[2], 0.32, size=9,
           align=PP_ALIGN.RIGHT, color=GRAY)
        _t(s, line, col_x[3], y + 0.04, col_w[3], 0.32, size=9, color=DGRAY)

    _footer(s, "Section Detector  ·  section_detector.py  ·  80.1% cross-val accuracy")


# ── Slide 6: What's Next ───────────────────────────────────────────────────────
def slide_next(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _box(s, 0.5, 1.8, 0.06, 2.0, BLACK)
    _t(s, "What's Next", 0.75, 1.75, 8.5, 0.7, size=36, bold=True)
    _rule(s, 0.75, 2.55, 8.3)

    steps = [
        ("Add audio features",
         "Pair lyrics with Spotify audio signals — danceability, energy, valence, tempo. "
         "Beat drops and energy changes define section boundaries better than text alone. "
         "This alone could push accuracy past 90%."),
        ("Improve bridge & pre-chorus detection",
         "The model struggles to separate bridge from pre-chorus because both "
         "appear once and sit between verse and chorus. A sequence model (CRF or LSTM) "
         "that considers surrounding blocks would fix this."),
        ("Integrate into the Streamlit app",
         "Any song pasted into the app gets a full section breakdown: "
         "which part is the chorus, which part to clip for TikTok, "
         "and the predicted virality score per section."),
    ]
    for i, (head, body) in enumerate(steps):
        y = 2.75 + i * 1.5
        _t(s, f"{i+1:02d}", 0.75, y,        0.55, 0.5,  size=20, bold=True, color=GRAY)
        _t(s, head,         1.4,  y,         8.0,  0.42, size=14, bold=True)
        _t(s, body,         1.4,  y + 0.48,  8.0,  0.75, size=11, color=GRAY)

    _footer(s)


# ── Build ───────────────────────────────────────────────────────────────────────
def build():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_problem(prs)
    slide_what_was_built(prs)
    slide_how_it_works(prs)
    slide_results(prs)
    slide_next(prs)

    prs.save(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    build()

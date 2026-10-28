"""
Export Weekly_Update_Oct7_2026 as PDF using reportlab.
Mirrors the 12-slide deck content.
"""
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from pathlib import Path

PDF_OUT = Path("/Volumes/Dhesel Driv /Work") / "Weekly_Update_Oct7_2026.pdf"

W, H = landscape((10 * inch, 7.5 * inch))

RED   = HexColor("#E5203A")
BLACK = HexColor("#111111")
GRAY  = HexColor("#888888")
LGRAY = HexColor("#E8E8E8")
FLGRAY= HexColor("#F7F7F7")
WHITE = colors.white
GREEN = HexColor("#1AA34A")
PINK  = HexColor("#FFEDED")
LPINK = HexColor("#FFF0F0")


def px(inches): return inches * inch
def rule(c, x, y, w, color=LGRAY, h=0.02):
    c.setFillColor(color); c.rect(px(x), H - px(y) - px(h), px(w), px(h), fill=1, stroke=0)

def box(c, x, y, w, h, color=FLGRAY):
    c.setFillColor(color); c.rect(px(x), H - px(y) - px(h), px(w), px(h), fill=1, stroke=0)

def text(c, t, x, y, size=11, bold=False, color=BLACK, align="left", italic=False):
    c.setFillColor(color)
    if bold and italic: c.setFont("Helvetica-BoldOblique", size)
    elif bold:          c.setFont("Helvetica-Bold", size)
    elif italic:        c.setFont("Helvetica-Oblique", size)
    else:               c.setFont("Helvetica", size)
    ry = H - px(y)
    if align == "center":
        c.drawCentredString(px(x), ry, t)
    elif align == "right":
        c.drawRightString(px(x), ry, t)
    else:
        c.drawString(px(x), ry, t)

def footer(c, note=""):
    rule(c, 0.6, 6.8, 8.8)
    if note:
        text(c, note, 0.6, 6.95, size=8, color=GRAY, italic=True)

def header(c, title, subtitle=None):
    text(c, title, 0.6, 0.62, size=22, bold=True)
    if subtitle:
        text(c, subtitle, 0.6, 1.02, size=9, color=GRAY, italic=True)
        rule(c, 0.6, 1.18, 8.8)
    else:
        rule(c, 0.6, 0.88, 8.8)

def bg(c):
    c.setFillColor(WHITE); c.rect(0, 0, W, H, fill=1, stroke=0)


# ── Slide 1: Title ────────────────────────────────────────────────────────────
def s1(c):
    bg(c)
    box(c, 0.6, 1.5, 0.06, 2.8, RED)
    text(c, "Weekly Update",   0.9, 2.2,  size=40, bold=True)
    text(c, "October 7, 2026", 0.9, 3.1,  size=22, bold=True, color=RED)
    text(c, "TikTok Viral Detector  ·  Chartmetric Data  ·  Billboard Comparison  ·  Model Retraining",
         0.9, 3.72, size=11, color=GRAY)
    rule(c, 0.9, 3.98, 8.1)
    stats = [("2,737","songs scraped"), ("995","with lyrics"), ("590","Billboard songs"), ("0.956","new model F1")]
    for i, (val, lbl) in enumerate(stats):
        x = 0.9 + i * 2.3
        text(c, val, x, 4.45, size=20, bold=True, color=RED)
        text(c, lbl, x, 4.78, size=9, color=GRAY)
    footer(c, "Dhesel  ·  SUPA Spring 2026")


# ── Slide 2: Agenda ───────────────────────────────────────────────────────────
def s2(c):
    bg(c); header(c, "This Week at a Glance")
    items = [
        ("01","Chartmetric Data Scrape",     "40 weeks of TikTok charts → 2,737 songs"),
        ("02","Genius Lyrics Fetch",          "995 songs now have full lyrics"),
        ("03","Billboard vs TikTok Analysis", "590 Billboard vs 995 TikTok songs compared"),
        ("04","Model Retraining",             "Training set grew from 231 → 1,218 songs"),
        ("05","Problem Found",               "Class imbalance: 92% viral → scores too high"),
        ("06","Next Week",                   "Add non-hits to fix model and re-score Tatiana"),
    ]
    for i, (num, head, sub) in enumerate(items):
        y = 1.32 + i * 0.87
        box(c, 0.55, y, 8.95, 0.8, FLGRAY if i % 2 == 0 else WHITE)
        box(c, 0.55, y, 0.04, 0.8, RED)
        text(c, num,  0.72, y + 0.52, size=10, bold=True, color=RED)
        text(c, head, 1.32, y + 0.3,  size=12, bold=True)
        text(c, sub,  1.32, y + 0.6,  size=9,  color=GRAY)
    footer(c, "6 topics covered  ·  October 7, 2026")


# ── Slide 3: Chartmetric Scrape ───────────────────────────────────────────────
def s3(c):
    bg(c)
    header(c, "Chartmetric — 40 Weeks of TikTok Data",
           "Fetched every weekly chart from Jan 5 → Oct 5, 2026")
    stats = [
        ("40",    "weeks scraped",   "Jan 5 → Oct 5, 2026"),
        ("7,000", "raw rows",        "100 sounds + 100 tracks per week"),
        ("2,737", "unique songs",    "After dedup by song + artist"),
        ("2×",    "chart types",     "TikTok Sounds + TikTok Tracks"),
    ]
    for i, (val, lbl, note) in enumerate(stats):
        x = 0.55 + i * 2.35
        box(c, x, 1.35, 2.22, 1.9, FLGRAY)
        box(c, x, 1.35, 2.22, 0.04, RED)
        text(c, val,  x + 0.15, 1.95, size=28, bold=True, color=RED)
        text(c, lbl,  x + 0.15, 2.55, size=10, bold=True)
        text(c, note, x + 0.15, 2.88, size=8,  color=GRAY)
    rule(c, 0.6, 3.45, 8.8)
    text(c, "HOW IT WORKS", 0.6, 3.72, size=9, bold=True, color=GRAY)
    steps = [
        ("Authenticate",  "POST /api/token → 1-hour Bearer token cached to disk"),
        ("Fetch Sounds",  "GET /api/charts/tiktok/sounds?date=YYYY-MM-DD&limit=100"),
        ("Fetch Tracks",  "GET /api/charts/tiktok/tracks?date=YYYY-MM-DD&limit=100"),
        ("Deduplicate",   "Keep highest video_count per (song, artist) across all weeks"),
        ("Save",          "Saved → /Volumes/Dhesel Driv /Work/cm_training_songs.csv"),
    ]
    for i, (step, desc) in enumerate(steps):
        x = 0.55 + (i % 3) * 3.08
        y = 3.92 + (i // 3) * 1.1
        box(c, x, y, 2.95, 1.0, FLGRAY)
        text(c, step, x + 0.12, y + 0.35, size=10, bold=True, color=RED)
        text(c, desc, x + 0.12, y + 0.72, size=7.5, color=GRAY)
    footer(c, "Script: fetch_cm_2026.py  ·  2s delay between API calls  ·  Token auto-refreshes")


# ── Slide 4: Lyrics Fetch ─────────────────────────────────────────────────────
def s4(c):
    bg(c)
    header(c, "Genius Lyrics Fetch — 2,737 Songs",
           "Fetched lyrics for all Chartmetric songs — saved to external drive")
    text(c, "RESULTS", 0.6, 1.45, size=9, bold=True, color=GRAY)
    BW = 7.5
    text(c, "Songs with lyrics",  0.6, 1.75, size=11, bold=True)
    box(c, 0.6, 1.88, BW * (995/2737),  0.24, RED)
    box(c, 0.6, 1.88, BW,               0.24, LGRAY)
    box(c, 0.6, 1.88, BW * (995/2737),  0.24, RED)
    text(c, "995 / 2,737  (36%)", 8.22, 1.95, size=9, bold=True, color=RED)

    text(c, "Without lyrics",  0.6, 2.38, size=11, color=GRAY)
    box(c, 0.6, 2.52, BW * (1742/2737), 0.24, LGRAY)
    text(c, "1,742 / 2,737  (64%)", 8.22, 2.59, size=9, color=GRAY)

    rule(c, 0.6, 3.0, 8.8)
    text(c, "WHY 1,742 SONGS HAVE NO LYRICS", 0.6, 3.22, size=9, bold=True, color=GRAY)
    reasons = [
        ("Instrumentals & Beats",    "Background music, lo-fi, binaural beats, piano — no lyrics exist."),
        ('"Original Sound" clips',   "TikTok user-created audio — Genius has nothing to search."),
        ("Non-English Tracks",       "Thai, Indonesian, Vietnamese, Khmer — limited Genius coverage."),
        ("Obscure / Viral Sounds",   "Niche regional music, meme sounds, no official lyrics published."),
    ]
    for i, (head, body) in enumerate(reasons):
        x = 0.55 + (i % 2) * 4.6
        y = 3.5 + (i // 2) * 1.4
        box(c, x, y, 4.45, 1.28, FLGRAY)
        text(c, head, x + 0.15, y + 0.38, size=10, bold=True)
        text(c, body, x + 0.15, y + 0.78, size=8.5, color=GRAY)
    footer(c, "Script: fetch_cm_lyrics.py  ·  0.5s delay  ·  Progress saved every 10 songs")


# ── Slide 5: Billboard Overview ───────────────────────────────────────────────
def s5(c):
    bg(c)
    header(c, "Billboard Hot 100 vs TikTok — Overview",
           "Full year 2026  ·  Both datasets scraped, lyrics fetched, features computed")
    text(c, "THE TWO DATASETS", 0.6, 1.38, size=9, bold=True, color=GRAY)
    box(c, 0.55, 1.58, 4.35, 2.2, FLGRAY)
    box(c, 0.55, 1.58, 4.35, 0.04, LGRAY)
    text(c, "Billboard Hot 100", 0.75, 1.92, size=13, bold=True)
    for i, (v, l) in enumerate([("590","songs with lyrics"),("40","weekly charts"),("All genres","pop, hip-hop, R&B"),("Source","billboard.com scraper")]):
        text(c, f"{v}  —  {l}", 0.75, 2.32 + i*0.36, size=9)
    box(c, 5.1, 1.58, 4.45, 2.2, FLGRAY)
    box(c, 5.1, 1.58, 4.45, 0.04, RED)
    text(c, "Chartmetric TikTok", 5.3, 1.92, size=13, bold=True, color=RED)
    for i, (v, l) in enumerate([("995","songs with lyrics"),("2,737","total unique songs"),("Global","sounds + tracks charts"),("Source","Chartmetric API")]):
        text(c, f"{v}  —  {l}", 5.3, 2.32 + i*0.36, size=9)
    rule(c, 0.6, 3.92, 8.8)
    text(c, "10 LYRIC FEATURES COMPARED", 0.6, 4.12, size=9, bold=True, color=GRAY)
    features = ["Hook Strength","Energy Score","Rhyme Density","Repetition Rate","Chorus/Hook Density",
                "Brevity","Emotional Punch","Vocab Richness","Avg Line Length","Song Length"]
    for i, feat in enumerate(features):
        x = 0.55 + (i % 5) * 1.88
        y = 4.35 + (i // 5) * 0.58
        box(c, x, y, 1.8, 0.45, FLGRAY)
        text(c, feat, x + 0.1, y + 0.28, size=7.5)
    footer(c, "Report: /Volumes/Dhesel Driv /Work/lyric_comparison_report.txt")


# ── Slide 6: Feature Table ────────────────────────────────────────────────────
def s6(c):
    bg(c)
    header(c, "Billboard vs TikTok — Feature by Feature",
           "590 Billboard songs  ·  995 Chartmetric TikTok songs")
    box(c, 0.55, 1.28, 9.1, 0.36, BLACK)
    text(c, "Feature",    0.7,  1.52, size=9, bold=True, color=WHITE)
    text(c, "Billboard",  5.3,  1.52, size=9, bold=True, color=WHITE, align="center")
    text(c, "TikTok",     6.8,  1.52, size=9, bold=True, color=WHITE, align="center")
    text(c, "Difference", 8.5,  1.52, size=9, bold=True, color=WHITE, align="center")
    rows = [
        ("Hook Strength  (most-repeated line freq.)",        0.0817, 0.1044, +27.8),
        ("Rhyme Density  (consecutive end-rhymes)",          0.1297, 0.1535, +18.4),
        ("Repetition Rate  (fraction of lines that repeat)", 0.2415, 0.2834, +17.4),
        ("Song Length  (total lyric lines)",                 47.56,  50.65,  +6.5),
        ("Brevity  (optimal 4–9 words = 1.0)",               0.9093, 0.9172, +0.9),
        ("Avg Line Length  (words per line)",                7.857,  7.062,  -10.1),
        ("Vocabulary Richness  (unique word ratio)",         0.4031, 0.3857, -4.3),
        ("Chorus/Hook Density  (hook section %)",            0.4887, 0.3897, -20.3),
        ("Emotional Punch  (sentiment intensity)",           0.0367, 0.0278, -24.4),
        ("Energy Score  (energy word density)",              0.0223, 0.0157, -29.6),
    ]
    for i, (feat, bb, tt, pct) in enumerate(rows):
        y = 1.66 + i * 0.5
        box(c, 0.55, y, 9.1, 0.48, FLGRAY if i % 2 == 0 else WHITE)
        text(c, feat,          0.68, y + 0.28, size=8.5)
        text(c, f"{bb:.4f}",   5.3,  y + 0.28, size=9, align="center")
        col = RED if pct > 0 else GRAY
        text(c, f"{tt:.4f}",   6.8,  y + 0.28, size=9, color=col, bold=abs(pct)>=10, align="center")
        arrow = "↑" if pct > 0 else "↓"
        text(c, f"{arrow} {abs(pct):.1f}%", 8.5, y + 0.28, size=9, color=col, bold=abs(pct)>=10, align="center")
    footer(c, "↑ = TikTok higher  ·  ↓ = TikTok lower  ·  Bold = ≥10% gap")


# ── Slide 7: TikTok Formula ───────────────────────────────────────────────────
def s7(c):
    bg(c)
    header(c, "What Makes a TikTok Song Different?",
           "Key takeaways from the Billboard vs TikTok comparison")
    higher = [("+28%","Hook Strength","TikTok songs repeat their best line far more — brain locks onto one phrase, perfect for lip-syncs."),
              ("+18%","Rhyme Density","Almost every couplet rhymes. The satisfying click makes viewers replay the 15-second clip."),
              ("+17%","Repetition Rate","More lines repeat. Repeated lines become earworms after just one short listen.")]
    lower  = [("−30%","Energy Score","Lower energy-word density. TikTok doesn't need hype words — one sticky hook does the work."),
              ("−24%","Emotional Punch","Less overt sentiment. TikTok is minimal and repetitive rather than emotionally complex."),
              ("−20%","Chorus Density","Less of the song is chorus. The hook line carries the song, not a big traditional chorus.")]
    text(c, "TIKTOK SCORES HIGHER ↑", 0.6, 1.38, size=9, bold=True, color=RED)
    for i, (pct, feat, note) in enumerate(higher):
        y = 1.58 + i * 1.55
        box(c, 0.55, y, 4.35, 1.44, FLGRAY)
        box(c, 0.55, y, 0.04, 1.44, RED)
        text(c, pct,  0.72, y + 0.52, size=20, bold=True, color=RED)
        text(c, feat, 1.9,  y + 0.32, size=11, bold=True)
        text(c, note, 0.72, y + 0.92, size=8.5, color=GRAY)
    text(c, "TIKTOK SCORES LOWER ↓", 5.2, 1.38, size=9, bold=True, color=GRAY)
    for i, (pct, feat, note) in enumerate(lower):
        y = 1.58 + i * 1.55
        box(c, 5.2, y, 4.45, 1.44, FLGRAY)
        box(c, 5.2, y, 0.04, 1.44, LGRAY)
        text(c, pct,  5.37, y + 0.52, size=20, bold=True, color=GRAY)
        text(c, feat, 6.55, y + 0.32, size=11, bold=True)
        text(c, note, 5.37, y + 0.92, size=8.5, color=GRAY)
    box(c, 0.55, 6.26, 9.1, 0.42, LPINK)
    text(c, "The TikTok Formula:", 0.75, 6.53, size=10, bold=True, color=RED)
    text(c, "Short lines  ·  One hook repeated  ·  Dense rhymes  ·  Minimal structure  ·  15 seconds is all it needs",
         3.1, 6.53, size=9.5)
    footer(c, "Based on 590 Billboard + 995 TikTok songs  ·  Full year 2026")


# ── Slide 8: Model Retraining ─────────────────────────────────────────────────
def s8(c):
    bg(c)
    header(c, "Model Retraining — Before vs After",
           "All 3 models retrained with Chartmetric data integrated")
    cols = [
        ("BEFORE", LGRAY, BLACK, [("Training songs","231"),("Training lines","12,282"),
          ("Line model F1","0.845"),("Song model F1","0.690"),
          ("Weight range","0.80 – 2.00"),("TikTok signal","None (chart tiers only)")]),
        ("AFTER",  RED,   WHITE, [("Training songs","1,218"),("Training lines","61,991"),
          ("Line model F1","0.843  (stable)"),("Song model F1","0.956  ↑"),
          ("Weight range","0.80 – 8.42"),("TikTok signal","Real video counts (Chartmetric)")]),
    ]
    for ci, (label, hc, htxt, items) in enumerate(cols):
        x = 0.55 + ci * 4.7
        box(c, x, 1.28, 4.5, 0.42, hc)
        text(c, label, x + 0.2, 1.56, size=12, bold=True, color=htxt)
        for i, (key, val) in enumerate(items):
            y = 1.72 + i * 0.72
            box(c, x, y, 4.5, 0.66, FLGRAY if i%2==0 else WHITE)
            text(c, key, x + 0.15, y + 0.25, size=8.5, color=GRAY)
            vc = RED if ci == 1 and "↑" in val else BLACK
            text(c, val, x + 0.15, y + 0.52, size=11, bold=True, color=vc)
    rule(c, 0.6, 6.22, 8.8)
    text(c, "Why song model F1 jumped 0.690 → 0.956:  Real TikTok video counts replace chart-tier guesses. "
         "Songs with 98M+ videos get weight 8.42×.", 0.6, 6.52, size=8.5, color=GRAY, italic=True)
    footer(c, "GradientBoostingClassifier  ·  300 estimators  ·  max_depth=4  ·  StandardScaler")


# ── Slide 9: Problem ──────────────────────────────────────────────────────────
def s9(c):
    bg(c)
    header(c, "Problem — Model Too Optimistic",
           "Why all of Tatiana's songs are scoring 93–100%")
    box(c, 0.55, 1.28, 8.95, 1.08, PINK)
    box(c, 0.55, 1.28, 0.06, 1.08, RED)
    text(c, "Class Imbalance: 92% Viral vs 8% Non-Viral", 0.75, 1.62, size=14, bold=True, color=RED)
    text(c, "The model barely knows what a non-viral song looks like — predicts high for almost everything.",
         0.75, 1.98, size=9.5, color=GRAY)
    text(c, "TRAINING DATA COMPOSITION", 0.6, 2.58, size=9, bold=True, color=GRAY)
    box(c, 0.6, 2.82, 8.55, 0.5, LGRAY)
    box(c, 0.6, 2.82, 8.55 * 0.92, 0.5, RED)
    text(c, "Viral  1,077 songs  (92%)", 0.75, 3.12, size=9.5, bold=True, color=WHITE)
    text(c, "Non-viral 92 (8%)", 8.35, 3.12, size=8.5, color=BLACK)
    rule(c, 0.6, 3.52, 8.8)
    text(c, "HOW WE GOT HERE", 0.6, 3.72, size=9, bold=True, color=GRAY)
    phases = [
        ("Old model  (balanced)",
         "95 viral  vs  92 non-viral  →  51% / 49%  ✓  Healthy split", GREEN, "0.690 F1"),
        ("After adding 987 Chartmetric songs  (all viral)",
         "1,077 viral  vs  92 non-viral  →  92% / 8%  ✗  Severely skewed", RED, "0.956 (misleading)"),
    ]
    for i, (head, body, col, f1) in enumerate(phases):
        y = 3.95 + i * 1.12
        box(c, 0.55, y, 8.95, 1.0, FLGRAY)
        box(c, 0.55, y, 0.04, 1.0, col)
        text(c, head, 0.72, y + 0.32, size=11, bold=True)
        text(c, body, 0.72, y + 0.68, size=9, color=GRAY)
        text(c, f1,   9.25, y + 0.48, size=10, bold=True, color=col, align="right")
    rule(c, 0.6, 6.18, 8.8)
    text(c, "Result: Tatiana's songs all score 93–100% — model can't distinguish stronger from weaker tracks.",
         0.6, 6.45, size=9, color=BLACK, italic=True)
    footer(c, "F1 = 0.956 is inflated — model predicts viral for almost everything")


# ── Slide 10: Tatiana Rankings ────────────────────────────────────────────────
def s10(c):
    bg(c)
    header(c, "Tatiana DeMaria — Current Rankings",
           "Retrained model scores  ·  ⚠ All high due to class imbalance — will be fixed next week")
    box(c, 0.55, 1.28, 8.95, 0.36, LPINK)
    text(c, "⚠  Scores are currently inflated — next week's fix will give a meaningful spread.",
         0.72, 1.52, size=9, bold=True, color=RED)
    box(c, 0.55, 1.66, 9.1, 0.36, BLACK)
    text(c, "Song",       0.7,  1.9,  size=9, bold=True, color=WHITE)
    text(c, "Score",      4.8,  1.9,  size=9, bold=True, color=WHITE, align="center")
    text(c, "Viral Line", 5.55, 1.9,  size=9, bold=True, color=WHITE)
    songs = [
        ("Make Me Feel",         "100%","Caught here"),
        ("You Make Me",          "100%","You, You Make Me"),
        ("Save Me",              "100%","God only knows"),
        ("Born Again Sinner",    "100%","I'm a born again sinner"),
        ("Fresh Meat",           "100%","Rotting here down in the belly of the beast"),
        ("What It Is About You", "99%", "With your guilt free heart"),
        ("Without You",          "99%", "Chasing the good life"),
        ("AWS",                  "98%", "tired of running"),
        ("Gallery",              "96%", "when you're afraid and all alone?"),
        ("London Don't Lie",     "94%", "London don't lie"),
        ("Anxiety - Acoustic",   "94%", "I hear the voices saying"),
        ("Too Much",             "93%", "Too much it ties me up"),
    ]
    for i, (song, score, line) in enumerate(songs):
        y = 2.04 + i * 0.37
        box(c, 0.55, y, 9.1, 0.35, FLGRAY if i%2==0 else WHITE)
        text(c, f"{i+1}. {song}", 0.68, y + 0.23, size=9)
        text(c, score, 4.8, y + 0.23, size=9, bold=True, color=RED, align="center")
        text(c, f'"{line}"', 5.55, y + 0.23, size=8, color=GRAY, italic=True)
    footer(c, "13 songs analysed  ·  Scores will spread meaningfully after next week's rebalance")


# ── Slide 11: Next Week ───────────────────────────────────────────────────────
def s11(c):
    bg(c)
    header(c, "Next Week — Fix the Model",
           "Add non-viral songs → rebalance → re-score Tatiana")
    steps = [
        ("01","Source Non-Hit Songs",
         "Identify songs with low TikTok traction — Billboard songs that never went viral, or Chartmetric songs with < 1,000 videos.",
         "Target: ~800–1,000 non-viral songs"),
        ("02","Fetch Their Lyrics",
         "Run Genius lyrics fetch on the non-viral dataset. English-language chart songs should have high Genius coverage.",
         "Expected: ~70–80% hit rate"),
        ("03","Retrain With Balanced Data",
         "Merge non-viral songs. Target 50/50 split: ~1,000 viral vs ~1,000 non-viral. Model should produce scores from 10–90%.",
         "Expected F1: ~0.80–0.85 (honest)"),
        ("04","Re-Score Tatiana's 13 Songs",
         "Run retrained model on Tatiana's catalog. Scores will now meaningfully separate her stronger and weaker tracks.",
         "Deliverable: ranked report + slide"),
    ]
    for i, (num, head, body, target) in enumerate(steps):
        y = 1.28 + i * 1.3
        box(c, 0.55, y, 8.95, 1.2, FLGRAY)
        box(c, 0.55, y, 0.04, 1.2, GREEN)
        text(c, num,    0.72, y + 0.58, size=12, bold=True, color=GREEN)
        text(c, head,   1.35, y + 0.32, size=12, bold=True)
        text(c, body,   1.35, y + 0.65, size=8.5, color=GRAY)
        text(c, target, 7.5,  y + 0.58, size=8, color=GREEN, italic=True, align="right")
    footer(c, "Goal: balanced 50/50 training set  ·  Meaningful Tatiana spread  ·  Honest model performance")


# ── Slide 12: Close ───────────────────────────────────────────────────────────
def s12(c):
    bg(c)
    box(c, 0.6, 1.6, 0.06, 2.6, RED)
    text(c, "See you next week.", 0.9, 2.5,  size=34, bold=True)
    text(c, "October 14, 2026",  0.9, 3.32, size=18, bold=True, color=RED)
    text(c, "What's coming:", 0.9, 4.08, size=10, bold=True, color=GRAY)
    for i, item in enumerate([
        "Non-viral song dataset sourced and lyrics fetched",
        "Models retrained on balanced 50/50 split",
        "Tatiana's 13 songs re-scored with meaningful spread",
    ]):
        text(c, f"→  {item}", 0.9, 4.48 + i * 0.38, size=10)
    footer(c, "2,737 songs  ·  995 lyrics  ·  590 Billboard  ·  40 weeks of 2026  ·  F1: 0.956 (needs rebalance)")


# ── Build ─────────────────────────────────────────────────────────────────────
def build():
    c = canvas.Canvas(str(PDF_OUT), pagesize=(W, H))
    slides = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
    for i, fn in enumerate(slides, 1):
        fn(c)
        c.showPage()
        print(f"  Slide {i}/12 done")
    c.save()
    print(f"\nSaved → {PDF_OUT}")


if __name__ == "__main__":
    build()

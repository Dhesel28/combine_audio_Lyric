"""Create PDF presentation for TikTok Virality Data Collection"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_styles():
    """Create custom paragraph styles"""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='SlideTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#003366'),
        spaceAfter=30,
        alignment=TA_CENTER
    ))

    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=36,
        textColor=colors.HexColor('#000000'),
        spaceAfter=20,
        alignment=TA_CENTER,
        spaceBefore=150
    ))

    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=18,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceBefore=20
    ))

    styles.add(ParagraphStyle(
        name='BulletPoint',
        parent=styles['Normal'],
        fontSize=16,
        leftIndent=40,
        spaceAfter=12,
        bulletIndent=20
    ))

    styles.add(ParagraphStyle(
        name='SubBullet',
        parent=styles['Normal'],
        fontSize=14,
        leftIndent=70,
        spaceAfter=8,
        bulletIndent=50,
        textColor=colors.HexColor('#444444')
    ))

    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=colors.HexColor('#006699'),
        spaceAfter=15,
        spaceBefore=20
    ))

    return styles

def add_title_slide(story, styles, title, subtitle):
    """Add a title slide"""
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(title, styles['MainTitle']))
    story.append(Paragraph(subtitle, styles['Subtitle']))
    story.append(PageBreak())

def add_content_slide(story, styles, title, bullets):
    """Add a content slide"""
    story.append(Paragraph(title, styles['SlideTitle']))
    story.append(Spacer(1, 0.3*inch))

    for bullet in bullets:
        if bullet.startswith("   "):
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{bullet.strip()}", styles['SubBullet']))
        else:
            story.append(Paragraph(f"<bullet>&bull;</bullet> {bullet}", styles['BulletPoint']))

    story.append(PageBreak())

def add_table_slide(story, styles, title, headers, rows):
    """Add a slide with a table"""
    story.append(Paragraph(title, styles['SlideTitle']))
    story.append(Spacer(1, 0.3*inch))

    # Create table data
    table_data = [headers] + rows

    # Create table
    col_widths = [2.5*inch, 1.5*inch, 3.5*inch]
    table = Table(table_data, colWidths=col_widths)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))

    story.append(table)
    story.append(PageBreak())

def add_two_column_slide(story, styles, title, left_title, left_points, right_title, right_points):
    """Add a two-column slide"""
    story.append(Paragraph(title, styles['SlideTitle']))
    story.append(Spacer(1, 0.2*inch))

    # Create two-column table
    left_content = [Paragraph(f"<b>{left_title}</b>", styles['SectionTitle'])]
    for point in left_points:
        left_content.append(Paragraph(f"<bullet>&bull;</bullet> {point}", styles['BulletPoint']))

    right_content = [Paragraph(f"<b>{right_title}</b>", styles['SectionTitle'])]
    for point in right_points:
        right_content.append(Paragraph(f"<bullet>&bull;</bullet> {point}", styles['BulletPoint']))

    # Combine into paragraphs
    left_text = "<br/>".join([f"<b>{left_title}</b><br/><br/>"] + [f"&bull; {p}<br/>" for p in left_points])
    right_text = "<br/>".join([f"<b>{right_title}</b><br/><br/>"] + [f"&bull; {p}<br/>" for p in right_points])

    col_style = ParagraphStyle(
        'ColStyle',
        fontSize=14,
        leading=20,
        leftIndent=10
    )

    table_data = [[
        Paragraph(left_text, col_style),
        Paragraph(right_text, col_style)
    ]]

    table = Table(table_data, colWidths=[3.8*inch, 3.8*inch])
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
    ]))

    story.append(table)
    story.append(PageBreak())

def main():
    print("Creating PDF presentation...")

    # Setup document
    pdf_path = OUTPUT_DIR / "TikTok_Virality_Data_Collection.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(letter),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    styles = create_styles()
    story = []

    # Slide 1: Title
    add_title_slide(
        story, styles,
        "TikTok Music Virality Analysis",
        f"Data Collection Progress | {datetime.now().strftime('%B %Y')}"
    )

    # Slide 2: Project Overview
    add_content_slide(story, styles, "Project Overview", [
        "Objective: Analyze what makes songs go viral on TikTok",
        "Approach: Collect multi-source music data for analysis",
        "Goal: Build predictive models for song virality",
        "Focus: Audio features, lyrics, and platform-specific metrics",
        "Application: Help artists and labels optimize for TikTok success"
    ])

    # Slide 3: Data Sources Overview
    add_content_slide(story, styles, "Data Collection Strategy", [
        "Multi-platform approach for comprehensive coverage",
        "TikTok trending tracks with Spotify audio features",
        "Spotify editorial playlists (TikTok Viral collections)",
        "Genius lyrics database for text analysis",
        "Video and creator metadata from TikTok",
        "Total: 490,000+ records across all sources"
    ])

    # Slide 4: Dataset Summary Table
    headers = ["Dataset", "Records", "Key Features"]
    rows = [
        ["TikTok Trending Tracks", "6,747", "Audio features, popularity, tempo"],
        ["Spotify TikTok Viral", "509", "Track metadata, playlist source"],
        ["Genius Lyrics", "480,000+", "Full lyrics, artist metadata"],
        ["TikTok Videos", "1,400+", "Views, likes, shares, hashtags"],
        ["Trending Authors", "200+", "Creator statistics"],
    ]
    add_table_slide(story, styles, "Datasets Collected", headers, rows)

    # Slide 5: TikTok Trending Tracks Details
    add_content_slide(story, styles, "TikTok Trending Tracks Dataset", [
        "6,747 songs that trended on TikTok",
        "Linked with Spotify audio features:",
        "   - Danceability, Energy, Valence (mood)",
        "   - Tempo, Key, Loudness",
        "   - Speechiness, Acousticness, Instrumentalness",
        "Includes popularity scores and release dates",
        "Perfect for ML feature engineering"
    ])

    # Slide 6: Spotify TikTok Playlists
    add_content_slide(story, styles, "Spotify TikTok Viral Playlists", [
        "Scraped 3 major TikTok editorial playlists:",
        "   - TikTok Music 2026 Viral Trending (201 tracks)",
        "   - Viral TikTok Songs 2025 (131 tracks)",
        "   - TikTok Hits 2024 (229 tracks)",
        "509 unique tracks after deduplication",
        "Current/recent viral songs for validation",
        "Includes track popularity and release dates"
    ])

    # Slide 7: Genius Lyrics Database
    add_content_slide(story, styles, "Genius Lyrics Database", [
        "480,000+ English songs with full lyrics",
        "Source: HuggingFace (brunokreiner/genius-lyrics)",
        "Enables NLP and sentiment analysis",
        "Key fields: lyrics, artist, song title, metadata",
        "302 MB dataset in Parquet format",
        "Can be matched with TikTok/Spotify data"
    ])

    # Slide 8: Audio Features Explained
    add_two_column_slide(
        story, styles,
        "Audio Features for Analysis",
        "Rhythm & Energy",
        [
            "Danceability: Suitable for dancing (0-1)",
            "Energy: Intensity and activity (0-1)",
            "Tempo: Beats per minute (BPM)",
            "Loudness: Overall volume (dB)"
        ],
        "Mood & Character",
        [
            "Valence: Musical positivity (0-1)",
            "Speechiness: Presence of words (0-1)",
            "Acousticness: Acoustic vs electronic",
            "Instrumentalness: Vocal presence"
        ]
    )

    # Slide 9: Challenges & Limitations
    add_content_slide(story, styles, "Challenges & Limitations", [
        "TikTok API (Unofficial) - Bot detection blocked all requests",
        "   - Tried TikTokApi library with multiple browser configurations",
        "   - TikTok actively detects and blocks automated scraping",
        "Tokchart.com - JavaScript-rendered, requires headless browser",
        "TikTok Research API (Official) - Restricted access",
        "   - Only available to academic researchers (PhD students)",
        "   - Approval process takes approximately 4 weeks",
        "Workaround: Used Kaggle datasets + Spotify API successfully"
    ])

    # Slide 10: Data Pipeline
    add_content_slide(story, styles, "Data Collection Pipeline", [
        "Automated Python scripts for reproducibility",
        "Kaggle API for TikTok datasets",
        "Spotify API (spotipy) for playlist extraction",
        "HuggingFace datasets library for lyrics",
        "Data stored in CSV and Parquet formats",
        "Modular design for easy updates"
    ])

    # Slide 11: Next Steps
    add_two_column_slide(
        story, styles,
        "Next Steps",
        "Data Processing",
        [
            "Clean and preprocess all datasets",
            "Match songs across platforms",
            "Engineer virality features",
            "Create train/test splits",
            "Handle missing values"
        ],
        "Analysis & Modeling",
        [
            "Exploratory data analysis (EDA)",
            "Correlation analysis",
            "NLP on lyrics (sentiment, topics)",
            "Build classification models",
            "Feature importance analysis"
        ]
    )

    # Slide 12: Summary
    add_content_slide(story, styles, "Summary", [
        "Successfully collected 490,000+ music records",
        "Multi-source approach: TikTok, Spotify, Genius",
        "Rich audio features for ML modeling",
        "Lyrics data enables NLP analysis",
        "Pipeline ready for data processing phase",
        "Foundation set for virality prediction models"
    ])

    # Slide 13: Thank You
    add_title_slide(
        story, styles,
        "Questions?",
        "TikTok Music Virality Analysis"
    )

    # Build PDF
    doc.build(story)
    print(f"PDF created: {pdf_path}")

    return pdf_path

if __name__ == "__main__":
    pdf_path = main()
    print(f"\nPresentation created successfully!")
    print(f"Location: {pdf_path}")

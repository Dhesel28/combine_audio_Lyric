"""
Generate PDF Project Plan for TikTok Song Virality Prediction using ReportLab
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime


def create_project_pdf():
    output_path = '/Users/dhekha/Desktop/SUPA_Spring/TikTok_Virality_Project_Plan.pdf'
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

    # Styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1E3C72'),
        alignment=TA_CENTER,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=30
    )

    chapter_style = ParagraphStyle(
        'Chapter',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1E3C72'),
        spaceBefore=20,
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#333333'),
        spaceBefore=15,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#444444'),
        spaceAfter=8,
        leading=14
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#444444'),
        leftIndent=20,
        spaceAfter=4,
        bulletIndent=10,
        leading=14
    )

    # Build content
    story = []

    # ============ TITLE PAGE ============
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("TikTok Song Virality", title_style))
    story.append(Paragraph("Prediction via Lyrics Analysis", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("NLP-Based Approach to Understanding Viral Music", subtitle_style))
    story.append(Spacer(1, inch))
    story.append(Paragraph("Project Plan Document", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(PageBreak())

    # ============ EXECUTIVE SUMMARY ============
    story.append(Paragraph("1. Executive Summary", chapter_style))
    story.append(Paragraph(
        "This project aims to build a machine learning system that analyzes song lyrics to predict "
        "TikTok virality. By examining linguistic patterns, emotional content, and structural elements "
        "of lyrics from viral TikTok songs (2020-2025), we will develop an algorithm capable of "
        "identifying characteristics that contribute to a song's viral potential on the platform.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Project Goals", section_style))
    goals = [
        "Collect and curate dataset of TikTok viral songs with lyrics (2020-2025)",
        "Identify linguistic features that correlate with virality",
        "Build predictive model for song virality based on lyrics",
        'Detect the specific "snippet" most likely to go viral',
        "Create interpretable insights for music industry applications"
    ]
    for goal in goals:
        story.append(Paragraph(f"• {goal}", bullet_style))

    # ============ PROBLEM STATEMENT ============
    story.append(Paragraph("2. Problem Statement", chapter_style))
    story.append(Paragraph(
        "TikTok has become a dominant force in music discovery, with songs going viral on the platform "
        'often translating to streaming success. However, what makes a song\'s lyrics "TikTok-able" '
        "remains poorly understood. This project focuses specifically on the lyrical component, "
        "asking: Can we predict virality from lyrics alone?",
        body_style
    ))

    story.append(Paragraph("Key Questions", section_style))
    questions = [
        "What lexical patterns appear in viral TikTok songs?",
        "Do certain emotional tones or themes predict virality?",
        "How does repetition and structure affect viral potential?",
        'Can we identify the 15-30 second "viral snippet" from lyrics?'
    ]
    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"{i}. {q}", bullet_style))

    story.append(Paragraph("Scope & Limitations", section_style))
    story.append(Paragraph(
        "This project focuses exclusively on lyrics. We acknowledge that virality depends on many "
        "factors beyond lyrics including: beat/production quality, artist popularity, timing, "
        "danceability, and TikTok algorithm dynamics. The lyrics-only approach provides a focused "
        "analysis but will have an accuracy ceiling due to these confounding variables.",
        body_style
    ))

    # ============ DATA SOURCES ============
    story.append(Paragraph("3. Data Sources", chapter_style))
    story.append(Paragraph("We will combine multiple datasets to create a comprehensive training corpus:", body_style))

    story.append(Paragraph("TikTok Song Datasets", section_style))
    tiktok_data = [
        ['Dataset', 'Description'],
        ['TikTok Trending Tracks', '7,000 tracks with technical info (Kaggle)'],
        ['TikTok Popular Songs 2020', 'Viral songs from 2020 (Kaggle)'],
        ['TikTok 2025 Dataset', 'Recent TikTok post data (Kaggle)'],
        ['Viral TikTok Songs 2022', '263 songs with audio features (Opendatabay)'],
        ['Popular TikTok Videos', 'Videos, authors, and music metadata (Kaggle)']
    ]
    t1 = Table(tiktok_data, colWidths=[2.5*inch, 4*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3C72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Lyrics Datasets", section_style))
    lyrics_data = [
        ['Dataset', 'Description'],
        ['Genius Lyrics (HuggingFace)', '480k English songs with metadata'],
        ['5 Million Song Lyrics', 'Large-scale lyrics corpus (Kaggle)'],
        ['Music Dataset 1950-2019', 'Lyrics + audio features (Mendeley)'],
        ['Hit Song Prediction Dataset', 'MSD + Genius IDs + BERT embeddings']
    ]
    t2 = Table(lyrics_data, colWidths=[2.5*inch, 4*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3C72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Data Collection Tools", section_style))
    tools = [
        "LyricsGenius Python API - for fetching lyrics from Genius.com",
        "Spotify API - for audio features and popularity metrics",
        "Custom scrapers - for supplementary data collection"
    ]
    for tool in tools:
        story.append(Paragraph(f"• {tool}", bullet_style))

    story.append(PageBreak())

    # ============ METHODOLOGY ============
    story.append(Paragraph("4. Methodology", chapter_style))

    story.append(Paragraph("Phase 1: Data Collection & Preparation", section_style))
    phase1 = [
        "Download TikTok viral song datasets from Kaggle",
        "Compile master list of viral songs (2020-2025) with virality scores",
        "Collect control group of non-viral songs from same period",
        "Fetch lyrics using LyricsGenius API",
        "Clean and standardize lyrics text",
        "Create train/validation/test splits (time-based to avoid leakage)"
    ]
    for i, item in enumerate(phase1, 1):
        story.append(Paragraph(f"{i}. {item}", bullet_style))

    story.append(Paragraph("Phase 2: Lyrics Preprocessing", section_style))
    phase2 = [
        "Remove structural markers ([Verse], [Chorus], etc.)",
        "Handle special characters and formatting",
        "Detect and handle multilingual content",
        "Segment lyrics into sections (verse, chorus, bridge, hook)",
        'Identify potential "viral snippets" based on repetition patterns'
    ]
    for i, item in enumerate(phase2, 1):
        story.append(Paragraph(f"{i}. {item}", bullet_style))

    story.append(Paragraph("Phase 3: Feature Engineering", section_style))
    story.append(Paragraph("We will extract multiple categories of features from lyrics:", body_style))

    feature_categories = {
        "Lexical Features": [
            "Word count, unique word count, vocabulary richness (TTR)",
            "Average word length, syllable count",
            "Repetition rate (word-level, phrase-level, line-level)",
            "Profanity/explicit content indicators"
        ],
        "Semantic Features": [
            "Sentiment polarity and subjectivity (VADER, TextBlob)",
            "Emotion detection (joy, sadness, anger, fear, surprise)",
            "Topic/theme clustering (LDA, BERTopic)",
            "Semantic coherence scores"
        ],
        "Phonetic Features": [
            "Rhyme density and patterns",
            "Alliteration and assonance frequency",
            "Syllable stress patterns"
        ],
        "Structural Features": [
            "Chorus-to-verse ratio",
            "Hook placement and frequency",
            "Line length distribution",
            "Section repetition patterns"
        ],
        "Embedding Features": [
            "Sentence embeddings (SBERT, all-MiniLM-L6-v2)",
            "Fine-tuned BERT/RoBERTa representations",
            "TF-IDF weighted word vectors"
        ]
    }

    for category, features in feature_categories.items():
        story.append(Paragraph(f"<b>{category}:</b>", body_style))
        for f in features:
            story.append(Paragraph(f"• {f}", bullet_style))

    story.append(Paragraph("Phase 4: Model Development", section_style))
    story.append(Paragraph("We will implement a progression of models from simple to complex:", body_style))

    models = {
        "Baseline Models": [
            "Logistic Regression with TF-IDF features",
            "Random Forest with engineered features",
            "XGBoost/LightGBM gradient boosting"
        ],
        "Deep Learning Models": [
            "Fine-tuned DistilBERT for classification",
            "RoBERTa with custom classification head",
            "Ensemble of transformer + traditional features"
        ],
        "Snippet Detection": [
            "Sliding window approach over lyrics sections",
            "Attention-based identification of key phrases",
            "Repetition-weighted scoring for hook detection"
        ]
    }

    for model_type, model_list in models.items():
        story.append(Paragraph(f"<b>{model_type}:</b>", body_style))
        for m in model_list:
            story.append(Paragraph(f"• {m}", bullet_style))

    story.append(Paragraph("Phase 5: Evaluation & Analysis", section_style))
    eval_steps = [
        "Classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC",
        "Feature importance analysis using SHAP values",
        "Error analysis on misclassified examples",
        "Ablation studies on feature groups",
        "Cross-temporal validation (train on 2020-2023, test on 2024-2025)"
    ]
    for i, step in enumerate(eval_steps, 1):
        story.append(Paragraph(f"{i}. {step}", bullet_style))

    story.append(PageBreak())

    # ============ TECHNICAL STACK ============
    story.append(Paragraph("5. Technical Stack", chapter_style))

    tech_data = [
        ['Category', 'Tools/Libraries'],
        ['Data Processing', 'pandas, numpy, datasets (HuggingFace)'],
        ['Data Collection', 'lyricsgenius, spotipy, requests, BeautifulSoup'],
        ['NLP Preprocessing', 'spaCy, NLTK, regex, langdetect'],
        ['Feature Engineering', 'scikit-learn, textstat, vaderSentiment'],
        ['Deep Learning', 'PyTorch, transformers (HuggingFace)'],
        ['ML Models', 'scikit-learn, XGBoost, LightGBM'],
        ['Visualization', 'matplotlib, seaborn, plotly'],
        ['Interpretability', 'SHAP, LIME, BertViz'],
        ['Experiment Tracking', 'Weights & Biases, MLflow']
    ]
    t3 = Table(tech_data, colWidths=[2*inch, 4.5*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3C72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.3*inch))

    # ============ PROJECT PHASES & DELIVERABLES ============
    story.append(Paragraph("6. Project Phases & Deliverables", chapter_style))

    phases = {
        "Phase 1: Data Collection": [
            "Deliverable: Cleaned dataset with ~10,000+ songs (viral + non-viral)",
            "Deliverable: Lyrics corpus matched to song metadata",
            "Milestone: Data exploration report with initial statistics"
        ],
        "Phase 2: Feature Engineering": [
            "Deliverable: Feature extraction pipeline",
            "Deliverable: Feature matrix with 50+ engineered features",
            "Milestone: Feature correlation analysis"
        ],
        "Phase 3: Model Development": [
            "Deliverable: Trained baseline models with benchmarks",
            "Deliverable: Fine-tuned transformer model",
            "Milestone: Model comparison report"
        ],
        "Phase 4: Snippet Detection": [
            "Deliverable: Viral snippet identification algorithm",
            "Deliverable: Ranked snippets for test songs",
            "Milestone: Snippet accuracy evaluation"
        ],
        "Phase 5: Final Analysis": [
            "Deliverable: Final model with documentation",
            "Deliverable: Interpretability report (what makes songs viral)",
            "Deliverable: API/demo for virality prediction"
        ]
    }

    for phase, deliverables in phases.items():
        story.append(Paragraph(phase, section_style))
        for d in deliverables:
            story.append(Paragraph(f"• {d}", bullet_style))

    story.append(PageBreak())

    # ============ EXPECTED OUTCOMES ============
    story.append(Paragraph("7. Expected Outcomes", chapter_style))

    story.append(Paragraph("Quantitative Goals", section_style))
    quant_goals = [
        "Binary classification accuracy: 65-75% (lyrics-only ceiling)",
        "AUC-ROC: 0.70-0.80",
        "Snippet identification accuracy: 60%+ (when ground truth available)"
    ]
    for g in quant_goals:
        story.append(Paragraph(f"• {g}", bullet_style))

    story.append(Paragraph("Qualitative Insights", section_style))
    qual_insights = [
        "Identified patterns: repetition, themes, emotional arcs",
        "Genre-specific virality indicators",
        "Temporal trends in viral lyrics (2020 vs 2025)"
    ]
    for i in qual_insights:
        story.append(Paragraph(f"• {i}", bullet_style))

    story.append(Paragraph("Limitations & Honest Assessment", section_style))
    story.append(Paragraph(
        "Lyrics alone cannot fully predict TikTok virality. Our model will capture linguistic "
        "patterns but miss crucial factors like production quality, artist fame, timing, and "
        "platform dynamics. We expect accuracy to plateau around 70% due to these inherent "
        "limitations. The primary value is in understanding lyrical patterns, not perfect prediction.",
        body_style
    ))

    # ============ NEXT STEPS ============
    story.append(Paragraph("8. Immediate Next Steps", chapter_style))
    next_steps = [
        "Download TikTok Trending Tracks dataset from Kaggle",
        "Download TikTok Popular Songs 2020 dataset",
        "Download Genius Lyrics dataset from HuggingFace",
        "Set up Genius API access for supplementary lyrics fetching",
        "Create data loading and merging pipeline",
        "Perform initial exploratory data analysis",
        "Define virality labeling criteria",
        "Build lyrics preprocessing pipeline"
    ]
    for i, step in enumerate(next_steps, 1):
        story.append(Paragraph(f"{i}. {step}", bullet_style))

    # ============ APPENDIX ============
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Appendix: Key Resources & Links", chapter_style))

    story.append(Paragraph("Datasets", section_style))
    datasets = [
        "Kaggle - TikTok Trending Tracks",
        "Kaggle - TikTok Popular Songs 2020",
        "Kaggle - 5 Million Song Lyrics",
        "HuggingFace - brunokreiner/genius-lyrics",
        "Mendeley - Music Dataset 1950-2019",
        "GitHub - Orfium/hsp-lyrics-dataset"
    ]
    for d in datasets:
        story.append(Paragraph(f"• {d}", bullet_style))

    story.append(Paragraph("Tools & Libraries", section_style))
    libs = [
        "LyricsGenius: github.com/johnwmillr/LyricsGenius",
        "HuggingFace Transformers: huggingface.co/transformers",
        "SHAP: github.com/slundberg/shap"
    ]
    for l in libs:
        story.append(Paragraph(f"• {l}", bullet_style))

    story.append(Paragraph("Related Research", section_style))
    research = [
        "Hit Song Science (HSS) literature",
        "Music Information Retrieval (MIR) approaches",
        "Sentiment analysis in lyrics"
    ]
    for r in research:
        story.append(Paragraph(f"• {r}", bullet_style))

    # Build PDF
    doc.build(story)
    print(f"PDF generated successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    create_project_pdf()

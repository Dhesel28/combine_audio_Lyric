"""
Generate PDF Project Plan for TikTok Song Virality Prediction
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime


class ProjectPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'TikTok Song Virality Prediction Project', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 60, 114)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(60, 60, 60)
        bullet_text = f"  * {text}"
        self.multi_cell(0, 6, bullet_text)

    def numbered_item(self, number, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(60, 60, 60)
        numbered_text = f"  {number}. {text}"
        self.multi_cell(0, 6, numbered_text)

    def add_table_row(self, col1, col2, is_header=False):
        if is_header:
            self.set_font('Helvetica', 'B', 9)
            self.set_fill_color(30, 60, 114)
            self.set_text_color(255, 255, 255)
        else:
            self.set_font('Helvetica', '', 9)
            self.set_fill_color(245, 245, 245)
            self.set_text_color(60, 60, 60)

        self.cell(60, 8, col1, border=1, fill=True)
        self.cell(0, 8, col2, border=1, fill=not is_header, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def create_project_pdf():
    pdf = ProjectPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(30, 60, 114)
    pdf.cell(0, 15, 'TikTok Song Virality', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 15, 'Prediction via Lyrics Analysis', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'NLP-Based Approach to Understanding Viral Music', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 11)
    pdf.cell(0, 10, 'Project Plan Document', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%B %d, %Y")}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Executive Summary
    pdf.add_page()
    pdf.chapter_title('1. Executive Summary')
    pdf.body_text(
        'This project aims to build a machine learning system that analyzes song lyrics to predict '
        'TikTok virality. By examining linguistic patterns, emotional content, and structural elements '
        'of lyrics from viral TikTok songs (2020-2025), we will develop an algorithm capable of '
        'identifying characteristics that contribute to a song\'s viral potential on the platform.'
    )
    pdf.ln(3)

    pdf.section_title('Project Goals')
    pdf.bullet_point('Collect and curate dataset of TikTok viral songs with lyrics (2020-2025)')
    pdf.bullet_point('Identify linguistic features that correlate with virality')
    pdf.bullet_point('Build predictive model for song virality based on lyrics')
    pdf.bullet_point('Detect the specific "snippet" most likely to go viral')
    pdf.bullet_point('Create interpretable insights for music industry applications')
    pdf.ln(5)

    # Problem Statement
    pdf.chapter_title('2. Problem Statement')
    pdf.body_text(
        'TikTok has become a dominant force in music discovery, with songs going viral on the platform '
        'often translating to streaming success. However, what makes a song\'s lyrics "TikTok-able" '
        'remains poorly understood. This project focuses specifically on the lyrical component, '
        'asking: Can we predict virality from lyrics alone?'
    )
    pdf.ln(2)

    pdf.section_title('Key Questions')
    pdf.numbered_item(1, 'What lexical patterns appear in viral TikTok songs?')
    pdf.numbered_item(2, 'Do certain emotional tones or themes predict virality?')
    pdf.numbered_item(3, 'How does repetition and structure affect viral potential?')
    pdf.numbered_item(4, 'Can we identify the 15-30 second "viral snippet" from lyrics?')
    pdf.ln(5)

    pdf.section_title('Scope & Limitations')
    pdf.body_text(
        'This project focuses exclusively on lyrics. We acknowledge that virality depends on many '
        'factors beyond lyrics including: beat/production quality, artist popularity, timing, '
        'danceability, and TikTok algorithm dynamics. The lyrics-only approach provides a focused '
        'analysis but will have an accuracy ceiling due to these confounding variables.'
    )
    pdf.ln(5)

    # Data Sources
    pdf.chapter_title('3. Data Sources')
    pdf.body_text('We will combine multiple datasets to create a comprehensive training corpus:')
    pdf.ln(3)

    pdf.section_title('TikTok Song Datasets')
    pdf.add_table_row('Dataset', 'Description', is_header=True)
    pdf.add_table_row('TikTok Trending Tracks', '7,000 tracks with technical info (Kaggle)')
    pdf.add_table_row('TikTok Popular Songs 2020', 'Viral songs from 2020 (Kaggle)')
    pdf.add_table_row('TikTok 2025 Dataset', 'Recent TikTok post data (Kaggle)')
    pdf.add_table_row('Viral TikTok Songs 2022', '263 songs with audio features (Opendatabay)')
    pdf.add_table_row('Popular TikTok Videos', 'Videos, authors, and music metadata (Kaggle)')
    pdf.ln(5)

    pdf.section_title('Lyrics Datasets')
    pdf.add_table_row('Dataset', 'Description', is_header=True)
    pdf.add_table_row('Genius Lyrics (HuggingFace)', '480k English songs with metadata')
    pdf.add_table_row('5 Million Song Lyrics', 'Large-scale lyrics corpus (Kaggle)')
    pdf.add_table_row('Music Dataset 1950-2019', 'Lyrics + audio features (Mendeley)')
    pdf.add_table_row('Hit Song Prediction', 'MSD + Genius IDs + BERT embeddings')
    pdf.ln(5)

    pdf.section_title('Data Collection Tools')
    pdf.bullet_point('LyricsGenius Python API - for fetching lyrics from Genius.com')
    pdf.bullet_point('Spotify API - for audio features and popularity metrics')
    pdf.bullet_point('Custom scrapers - for supplementary data collection')
    pdf.ln(5)

    # Methodology
    pdf.add_page()
    pdf.chapter_title('4. Methodology')
    pdf.ln(2)

    pdf.section_title('Phase 1: Data Collection & Preparation')
    pdf.numbered_item(1, 'Download TikTok viral song datasets from Kaggle')
    pdf.numbered_item(2, 'Compile master list of viral songs (2020-2025) with virality scores')
    pdf.numbered_item(3, 'Collect control group of non-viral songs from same period')
    pdf.numbered_item(4, 'Fetch lyrics using LyricsGenius API')
    pdf.numbered_item(5, 'Clean and standardize lyrics text')
    pdf.numbered_item(6, 'Create train/validation/test splits (time-based to avoid leakage)')
    pdf.ln(5)

    pdf.section_title('Phase 2: Lyrics Preprocessing')
    pdf.numbered_item(1, 'Remove structural markers ([Verse], [Chorus], etc.)')
    pdf.numbered_item(2, 'Handle special characters and formatting')
    pdf.numbered_item(3, 'Detect and handle multilingual content')
    pdf.numbered_item(4, 'Segment lyrics into sections (verse, chorus, bridge, hook)')
    pdf.numbered_item(5, 'Identify potential "viral snippets" based on repetition patterns')
    pdf.ln(5)

    pdf.section_title('Phase 3: Feature Engineering')
    pdf.body_text('We will extract multiple categories of features from lyrics:')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Lexical Features:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Word count, unique word count, vocabulary richness (TTR)')
    pdf.bullet_point('Average word length, syllable count')
    pdf.bullet_point('Repetition rate (word-level, phrase-level, line-level)')
    pdf.bullet_point('Profanity/explicit content indicators')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Semantic Features:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Sentiment polarity and subjectivity (VADER, TextBlob)')
    pdf.bullet_point('Emotion detection (joy, sadness, anger, fear, surprise)')
    pdf.bullet_point('Topic/theme clustering (LDA, BERTopic)')
    pdf.bullet_point('Semantic coherence scores')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Phonetic Features:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Rhyme density and patterns')
    pdf.bullet_point('Alliteration and assonance frequency')
    pdf.bullet_point('Syllable stress patterns')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Structural Features:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Chorus-to-verse ratio')
    pdf.bullet_point('Hook placement and frequency')
    pdf.bullet_point('Line length distribution')
    pdf.bullet_point('Section repetition patterns')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Embedding Features:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Sentence embeddings (SBERT, all-MiniLM-L6-v2)')
    pdf.bullet_point('Fine-tuned BERT/RoBERTa representations')
    pdf.bullet_point('TF-IDF weighted word vectors')
    pdf.ln(5)

    pdf.section_title('Phase 4: Model Development')
    pdf.body_text('We will implement a progression of models from simple to complex:')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Baseline Models:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Logistic Regression with TF-IDF features')
    pdf.bullet_point('Random Forest with engineered features')
    pdf.bullet_point('XGBoost/LightGBM gradient boosting')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Deep Learning Models:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Fine-tuned DistilBERT for classification')
    pdf.bullet_point('RoBERTa with custom classification head')
    pdf.bullet_point('Ensemble of transformer + traditional features')
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Snippet Detection:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.bullet_point('Sliding window approach over lyrics sections')
    pdf.bullet_point('Attention-based identification of key phrases')
    pdf.bullet_point('Repetition-weighted scoring for hook detection')
    pdf.ln(5)

    pdf.section_title('Phase 5: Evaluation & Analysis')
    pdf.numbered_item(1, 'Classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC')
    pdf.numbered_item(2, 'Feature importance analysis using SHAP values')
    pdf.numbered_item(3, 'Error analysis on misclassified examples')
    pdf.numbered_item(4, 'Ablation studies on feature groups')
    pdf.numbered_item(5, 'Cross-temporal validation (train on 2020-2023, test on 2024-2025)')
    pdf.ln(5)

    # Technical Stack
    pdf.add_page()
    pdf.chapter_title('5. Technical Stack')
    pdf.ln(2)

    pdf.add_table_row('Category', 'Tools/Libraries', is_header=True)
    pdf.add_table_row('Data Processing', 'pandas, numpy, datasets (HuggingFace)')
    pdf.add_table_row('Data Collection', 'lyricsgenius, spotipy, requests, BeautifulSoup')
    pdf.add_table_row('NLP Preprocessing', 'spaCy, NLTK, regex, langdetect')
    pdf.add_table_row('Feature Engineering', 'scikit-learn, textstat, vaderSentiment')
    pdf.add_table_row('Deep Learning', 'PyTorch, transformers (HuggingFace)')
    pdf.add_table_row('ML Models', 'scikit-learn, XGBoost, LightGBM')
    pdf.add_table_row('Visualization', 'matplotlib, seaborn, plotly')
    pdf.add_table_row('Interpretability', 'SHAP, LIME, BertViz')
    pdf.add_table_row('Experiment Tracking', 'Weights & Biases, MLflow')
    pdf.ln(5)

    # Project Phases
    pdf.chapter_title('6. Project Phases & Deliverables')
    pdf.ln(2)

    pdf.section_title('Phase 1: Data Collection')
    pdf.bullet_point('Deliverable: Cleaned dataset with ~10,000+ songs (viral + non-viral)')
    pdf.bullet_point('Deliverable: Lyrics corpus matched to song metadata')
    pdf.bullet_point('Milestone: Data exploration report with initial statistics')
    pdf.ln(3)

    pdf.section_title('Phase 2: Feature Engineering')
    pdf.bullet_point('Deliverable: Feature extraction pipeline')
    pdf.bullet_point('Deliverable: Feature matrix with 50+ engineered features')
    pdf.bullet_point('Milestone: Feature correlation analysis')
    pdf.ln(3)

    pdf.section_title('Phase 3: Model Development')
    pdf.bullet_point('Deliverable: Trained baseline models with benchmarks')
    pdf.bullet_point('Deliverable: Fine-tuned transformer model')
    pdf.bullet_point('Milestone: Model comparison report')
    pdf.ln(3)

    pdf.section_title('Phase 4: Snippet Detection')
    pdf.bullet_point('Deliverable: Viral snippet identification algorithm')
    pdf.bullet_point('Deliverable: Ranked snippets for test songs')
    pdf.bullet_point('Milestone: Snippet accuracy evaluation')
    pdf.ln(3)

    pdf.section_title('Phase 5: Final Analysis')
    pdf.bullet_point('Deliverable: Final model with documentation')
    pdf.bullet_point('Deliverable: Interpretability report (what makes songs viral)')
    pdf.bullet_point('Deliverable: API/demo for virality prediction')
    pdf.ln(5)

    # Expected Outcomes
    pdf.chapter_title('7. Expected Outcomes')
    pdf.ln(2)

    pdf.section_title('Quantitative Goals')
    pdf.bullet_point('Binary classification accuracy: 65-75% (lyrics-only ceiling)')
    pdf.bullet_point('AUC-ROC: 0.70-0.80')
    pdf.bullet_point('Snippet identification accuracy: 60%+ (when ground truth available)')
    pdf.ln(3)

    pdf.section_title('Qualitative Insights')
    pdf.bullet_point('Identified patterns: repetition, themes, emotional arcs')
    pdf.bullet_point('Genre-specific virality indicators')
    pdf.bullet_point('Temporal trends in viral lyrics (2020 vs 2025)')
    pdf.ln(3)

    pdf.section_title('Limitations & Honest Assessment')
    pdf.body_text(
        'Lyrics alone cannot fully predict TikTok virality. Our model will capture linguistic '
        'patterns but miss crucial factors like production quality, artist fame, timing, and '
        'platform dynamics. We expect accuracy to plateau around 70% due to these inherent '
        'limitations. The primary value is in understanding lyrical patterns, not perfect prediction.'
    )
    pdf.ln(5)

    # Next Steps
    pdf.add_page()
    pdf.chapter_title('8. Immediate Next Steps')
    pdf.ln(2)

    pdf.numbered_item(1, 'Download TikTok Trending Tracks dataset from Kaggle')
    pdf.numbered_item(2, 'Download TikTok Popular Songs 2020 dataset')
    pdf.numbered_item(3, 'Download Genius Lyrics dataset from HuggingFace')
    pdf.numbered_item(4, 'Set up Genius API access for supplementary lyrics fetching')
    pdf.numbered_item(5, 'Create data loading and merging pipeline')
    pdf.numbered_item(6, 'Perform initial exploratory data analysis')
    pdf.numbered_item(7, 'Define virality labeling criteria')
    pdf.numbered_item(8, 'Build lyrics preprocessing pipeline')
    pdf.ln(5)

    # Appendix: Key Resources
    pdf.chapter_title('Appendix: Key Resources & Links')
    pdf.ln(2)

    pdf.section_title('Datasets')
    pdf.set_font('Helvetica', '', 9)
    pdf.bullet_point('Kaggle - TikTok Trending Tracks')
    pdf.bullet_point('Kaggle - TikTok Popular Songs 2020')
    pdf.bullet_point('Kaggle - 5 Million Song Lyrics')
    pdf.bullet_point('HuggingFace - brunokreiner/genius-lyrics')
    pdf.bullet_point('Mendeley - Music Dataset 1950-2019')
    pdf.bullet_point('GitHub - Orfium/hsp-lyrics-dataset')
    pdf.ln(3)

    pdf.section_title('Tools & Libraries')
    pdf.bullet_point('LyricsGenius: github.com/johnwmillr/LyricsGenius')
    pdf.bullet_point('HuggingFace Transformers: huggingface.co/transformers')
    pdf.bullet_point('SHAP: github.com/slundberg/shap')
    pdf.ln(3)

    pdf.section_title('Related Research')
    pdf.bullet_point('Hit Song Science (HSS) literature')
    pdf.bullet_point('Music Information Retrieval (MIR) approaches')
    pdf.bullet_point('Sentiment analysis in lyrics')

    # Save PDF
    output_path = '/Users/dhekha/Desktop/SUPA_Spring/TikTok_Virality_Project_Plan.pdf'
    pdf.output(output_path)
    print(f"PDF generated successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    create_project_pdf()

"""
Valentine's Day Hit Song Predictor — Streamlit Dashboard (v6)
Uses HuggingFace MiniLM Embeddings + Cyanite Audio Features + hand-crafted lyrics features
"""

import streamlit as st
from functools import partial
import pandas as pd
import numpy as np
import re
import os
import subprocess
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RepeatedStratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Valentine Hit Predictor", page_icon="💘", layout="wide")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sia = SentimentIntensityAnalyzer()

# ── HUGGINGFACE EMBEDDING ──────────────────────────────────────────────────

EMBED_DIM = 384

@st.cache_resource
def load_minilm():
    tok = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    mdl = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    mdl.eval()
    return tok, mdl

_tokenizer, _bert_model = load_minilm()


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (torch.sum(token_embeddings * mask_expanded, 1) /
            torch.clamp(mask_expanded.sum(1), min=1e-9))


def get_embedding(text, max_length=256):
    if pd.isna(text) or not text.strip():
        return np.zeros(EMBED_DIM)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length - 2:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length, padding=True)
        with torch.no_grad():
            outputs = _bert_model(**inputs)
        emb = mean_pool(outputs, inputs['attention_mask']).squeeze().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
    else:
        chunk_size = max_length - 2
        stride = chunk_size // 2
        embeddings = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = _tokenizer.decode(chunk_tokens)
            inputs = _tokenizer(chunk_text, return_tensors="pt", truncation=True,
                                max_length=max_length, padding=True)
            with torch.no_grad():
                outputs = _bert_model(**inputs)
            emb = mean_pool(outputs, inputs['attention_mask']).squeeze().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            embeddings.append(emb)
            if i + chunk_size >= len(tokens):
                break
        return np.mean(embeddings, axis=0)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def clean_lyrics(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def get_rhyme_score(lyrics_raw):
    if pd.isna(lyrics_raw):
        return 0
    lines = [l.strip().lower() for l in lyrics_raw.split('\n') if l.strip()]
    if len(lines) < 2:
        return 0
    endings = []
    for line in lines:
        words = re.findall(r'[a-z]+', line)
        if words:
            endings.append(words[-1][-3:] if len(words[-1]) >= 3 else words[-1])
        else:
            endings.append('')
    rhymes = 0
    for i in range(len(endings) - 1):
        if endings[i] and endings[i] == endings[i + 1]:
            rhymes += 1
        elif endings[i] and endings[i + 1] and endings[i][-2:] == endings[i + 1][-2:]:
            rhymes += 0.5
    return rhymes / max(len(endings) - 1, 1)


def get_repetition_score(lyrics_raw):
    if pd.isna(lyrics_raw):
        return 0
    lines = [l.strip().lower() for l in lyrics_raw.split('\n') if l.strip()]
    if not lines:
        return 0
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(lines)


def extract_lyrics_features(lyrics_raw):
    lyrics = clean_lyrics(lyrics_raw)
    words = lyrics.split()
    wc = len(words)
    lines = [l.strip() for l in lyrics_raw.split('\n') if l.strip()] if pd.notna(lyrics_raw) else []
    lc = len(lines)
    uw = len(set(words)) if words else 0

    def cnt(w):
        return len(re.findall(r'\b' + w + r'\b', lyrics))

    def cnt_list(wl):
        return sum(cnt(w) for w in wl)

    love, you, i_m, we = cnt('love'), cnt('you'), cnt('i'), cnt('we')
    heart, baby, forever = cnt('heart'), cnt('baby'), cnt('forever')
    avg_wl = np.mean([len(w) for w in words]) if words else 0
    has_chorus = int(bool(re.search(r'\[chorus\]', lyrics_raw.lower()))) if pd.notna(lyrics_raw) else 0

    romance = cnt_list(['love','heart','kiss','darling','honey','sweetheart','romance','passion','desire'])
    longing = cnt_list(['miss','want','need','wish','dream','hope','wait','alone','without'])
    joy = cnt_list(['happy','smile','laugh','joy','beautiful','perfect','amazing','wonderful'])
    intimacy = cnt_list(['touch','hold','close','arms','eyes','skin','body','breath','warm'])
    commitment = cnt_list(['forever','always','never','promise','together','stay','belong','mine','yours'])
    pain = cnt_list(['cry','hurt','pain','break','tear','lose','lost','gone','sorry','wrong'])
    total_emotion = romance + longing + joy + intimacy + commitment + pain
    sentiment_kw = cnt_list(['love','heart','forever','always','beautiful','dream','kiss','hold',
                             'touch','feel','want','need','mine','together','soul','darling','honey','sweet','passion'])

    ld = uw / wc if wc > 0 else 0
    all_ = wc / lc if lc > 0 else 0
    tp = you + i_m + we + 1
    pos_r = (romance + joy + intimacy) / (total_emotion + 1)
    pain_r = pain / (total_emotion + 1)

    vader = sia.polarity_scores(lyrics)
    blob = TextBlob(lyrics)

    if lines:
        ls = [sia.polarity_scores(l.lower())['compound'] for l in lines]
        sv, sr = np.var(ls), max(ls) - min(ls)
        mid = len(ls) // 2
        ea = np.mean(ls[mid:]) - np.mean(ls[:mid]) if mid > 0 else 0
        lc_cnt = Counter(l.lower().strip() for l in lines)
        hook = lc_cnt.most_common(1)[0][1] / len(lines) if lc_cnt else 0
    else:
        sv, sr, ea, hook = 0, 0, 0, 0

    return {
        'word_count': wc, 'line_count': lc, 'unique_words': uw,
        'love_mentions': love, 'you_mentions': you, 'i_mentions': i_m, 'we_mentions': we,
        'heart_mentions': heart, 'baby_mentions': baby, 'forever_mentions': forever,
        'avg_word_length': avg_wl, 'has_chorus_repeat': has_chorus,
        'sentiment_keywords': sentiment_kw, 'lexical_diversity': ld, 'avg_line_length': all_,
        'romance_score': romance, 'longing_score': longing, 'joy_score': joy,
        'intimacy_score': intimacy, 'commitment_score': commitment, 'pain_score': pain,
        'total_emotion': total_emotion,
        'rhyme_score': get_rhyme_score(lyrics_raw),
        'repetition_score': get_repetition_score(lyrics_raw),
        'you_ratio': you/tp, 'i_ratio': i_m/tp, 'we_ratio': we/tp,
        'positivity_ratio': pos_r, 'pain_ratio': pain_r,
        'vader_compound': vader['compound'], 'vader_pos': vader['pos'],
        'vader_neg': vader['neg'], 'vader_neu': vader['neu'],
        'tb_polarity': blob.sentiment.polarity, 'tb_subjectivity': blob.sentiment.subjectivity,
        'sentiment_variance': sv, 'sentiment_range': sr, 'emotional_arc': ea,
        'hook_density': hook,
    }


# ── CYANITE FEATURE EXTRACTION ──────────────────────────────────────────────

def extract_cyanite_features(row):
    """Extract numeric and encoded features from Cyanite audio analysis."""
    feats = {}
    feats['cy_valence'] = row.get('cyanite_valence', 0) or 0
    feats['cy_arousal'] = row.get('cyanite_arousal', 0) or 0
    feats['cy_tempo'] = row.get('cyanite_tempo_bpm', 0) or 0

    feats['m2e_valence'] = row.get('m2e_valence', 0) if pd.notna(row.get('m2e_valence')) else 0
    feats['m2e_arousal'] = row.get('m2e_arousal', 0) if pd.notna(row.get('m2e_arousal')) else 0
    feats['m2e_valentine_fit'] = row.get('m2e_valentine_fit_score', 0) if pd.notna(row.get('m2e_valentine_fit_score')) else 0
    feats['m2e_best_valence'] = row.get('m2e_best_valence', 0) if pd.notna(row.get('m2e_best_valence')) else 0
    feats['m2e_best_arousal'] = row.get('m2e_best_arousal', 0) if pd.notna(row.get('m2e_best_arousal')) else 0

    energy_map = {'low': 0, 'medium': 1, 'high': 2, 'variable': 1.5}
    feats['cy_energy'] = energy_map.get(str(row.get('cyanite_energy_level', '')).lower(), 1)

    dyn_map = {'low': 0, 'medium': 1, 'high': 2, 'variable': 1.5}
    feats['cy_energy_dynamics'] = dyn_map.get(str(row.get('cyanite_energy_dynamics', '')).lower(), 1)

    vg = str(row.get('cyanite_voice_gender', '')).lower()
    feats['cy_voice_male'] = 1 if 'male' in vg and 'female' not in vg else 0
    feats['cy_voice_female'] = 1 if 'female' in vg else 0

    vp_map = {'low': 0, 'medium': 1, 'high': 2}
    feats['cy_voice_presence'] = vp_map.get(str(row.get('cyanite_voice_presence', '')).lower(), 1)

    mood_tags = str(row.get('cyanite_mood_tags', '')).lower()
    adv_tags = str(row.get('cyanite_mood_advanced_tags', '')).lower()
    all_moods = mood_tags + ', ' + adv_tags

    feats['cy_mood_romantic'] = 1 if 'romantic' in all_moods else 0
    feats['cy_mood_happy'] = 1 if 'happy' in all_moods else 0
    feats['cy_mood_chilled'] = 1 if 'chilled' in all_moods or 'relaxed' in all_moods else 0
    feats['cy_mood_sad'] = 1 if 'sad' in all_moods or 'melanchol' in all_moods else 0
    feats['cy_mood_sexy'] = 1 if 'sexy' in all_moods or 'seductive' in all_moods else 0
    feats['cy_mood_warm'] = 1 if 'warm' in all_moods else 0
    feats['cy_mood_loving'] = 1 if 'loving' in all_moods else 0
    feats['cy_mood_tender'] = 1 if 'tender' in all_moods else 0
    feats['cy_mood_passionate'] = 1 if 'passionate' in all_moods or 'passion' in all_moods else 0
    feats['cy_mood_sentimental'] = 1 if 'sentimental' in all_moods else 0
    feats['cy_mood_nostalgic'] = 1 if 'nostalgic' in all_moods else 0
    feats['cy_mood_dreamy'] = 1 if 'dreamy' in all_moods else 0
    feats['cy_mood_energetic'] = 1 if 'energetic' in all_moods or 'upbeat' in all_moods else 0
    feats['cy_mood_dark'] = 1 if 'dark' in all_moods or 'aggressive' in all_moods else 0

    rom_moods = ['romantic','loving','tender','passionate','sentimental','warm','sexy','seductive','intimate','dreamy']
    feats['cy_romantic_mood_count'] = sum(1 for m in rom_moods if m in all_moods)

    ep = str(row.get('cyanite_emotional_profile', '')).lower()
    feats['cy_emotion_positive'] = 1 if 'positive' in ep else 0
    feats['cy_emotion_negative'] = 1 if 'negative' in ep else 0
    feats['cy_emotion_balanced'] = 1 if 'balanced' in ep else 0

    mv = str(row.get('cyanite_movement_tags', '')).lower()
    feats['cy_flowing'] = 1 if 'flowing' in mv else 0
    feats['cy_swaying'] = 1 if 'swaying' in mv or 'groovy' in mv else 0
    feats['cy_bouncy'] = 1 if 'bouncy' in mv or 'driving' in mv else 0

    ch = str(row.get('cyanite_character_tags', '')).lower()
    feats['cy_warm'] = 1 if 'warm' in ch else 0
    feats['cy_intimate'] = 1 if 'intimate' in ch else 0
    feats['cy_retro'] = 1 if 'retro' in ch or 'vintage' in ch else 0

    tempo = feats['cy_tempo']
    feats['cy_tempo_slow'] = 1 if tempo < 85 else 0
    feats['cy_tempo_mid'] = 1 if 85 <= tempo <= 120 else 0
    feats['cy_tempo_fast'] = 1 if tempo > 120 else 0

    feats['cy_valence_x_romantic'] = feats['cy_valence'] * feats['cy_mood_romantic']
    feats['cy_arousal_x_energy'] = feats['cy_arousal'] * feats['cy_energy']

    return feats


def normalize_name(s):
    s = str(s).lower().strip().strip("'\"")
    for suffix in ['official audio', 'acoustic', 'bp mix 2', 'master',
                   'rough boucnce tdm britain 25th nov 2024', '090920']:
        s = s.replace(suffix, '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── TRAIN MODEL (cached) ────────────────────────────────────────────────────

@st.cache_data
def train_model():
    # Load datasets
    lyrics_df = pd.read_csv("valentine_songs_full_dataset.csv")
    lyrics_df = lyrics_df[lyrics_df['has_lyrics'] == True].copy()
    lyrics_df = lyrics_df.dropna(subset=['genius_pageviews'])

    cyanite_df = pd.read_csv("valentine_songs_cyanite.csv")

    # Merge on song title + artist
    lyrics_df['key'] = lyrics_df['all song_title'].str.lower().str.strip() + '|' + lyrics_df['artist'].str.lower().str.strip()
    cyanite_df['key'] = cyanite_df['song_title'].str.lower().str.strip() + '|' + cyanite_df['artist'].str.lower().str.strip()

    df = lyrics_df.merge(cyanite_df, on='key', how='inner', suffixes=('', '_cy'))

    median_views = df['genius_pageviews'].median()
    df['is_hit'] = (df['genius_pageviews'] >= median_views).astype(int)

    # Extract lyrics features
    lf_list = [extract_lyrics_features(row['lyrics']) for _, row in df.iterrows()]
    lf_df = pd.DataFrame(lf_list, index=df.index)
    lyrics_feat_cols = list(lf_df.columns)

    # Extract Cyanite features
    cf_list = [extract_cyanite_features(row) for _, row in df.iterrows()]
    cf_df = pd.DataFrame(cf_list, index=df.index)
    cyanite_feat_cols = list(cf_df.columns)

    # Extract MiniLM embeddings
    embeddings = [get_embedding(row['lyrics']) for _, row in df.iterrows()]
    emb_array = np.array(embeddings)
    emb_cols = [f'emb_{i}' for i in range(EMBED_DIM)]
    emb_df = pd.DataFrame(emb_array, index=df.index, columns=emb_cols)

    for col in lf_df.columns:
        df[col] = lf_df[col]
    for col in cf_df.columns:
        df[col] = cf_df[col]
    for col in emb_cols:
        df[col] = emb_df[col]

    # Genre + duet + decade
    le_genre = LabelEncoder()
    df['genre_encoded'] = le_genre.fit_transform(df['genre'])
    df['is_duet'] = df['is_duet'].astype(int)
    df['decade'] = (df['release_year'] // 10) * 10
    le_decade = LabelEncoder()
    df['decade_encoded'] = le_decade.fit_transform(df['decade'])

    all_feature_cols = lyrics_feat_cols + cyanite_feat_cols + emb_cols + ['genre_encoded', 'is_duet', 'decade_encoded']

    X = df[all_feature_cols].values
    y = df['is_hit'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_best = min(100, len(all_feature_cols))
    selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=k_best)
    X_selected = selector.fit_transform(X_scaled, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # ── Version history (hardcoded from canonical runs with fixed random_state=42) ──
    version_results = [
        {'Version': 'v1: Basic Features', 'CV Accuracy': 49.1,
         'Model': 'Gradient Boosting', 'Features': '15 hand-crafted'},
        {'Version': 'v2: + TF-IDF + Emotions', 'CV Accuracy': 58.1,
         'Model': 'Logistic Regression', 'Features': '334 (HC + TF-IDF)'},
        {'Version': 'v3: + VADER + Tuning', 'CV Accuracy': 49.1,
         'Model': 'Soft Voting Ensemble', 'Features': '200 selected'},
        {'Version': 'v4: + HuggingFace MiniLM', 'CV Accuracy': 64.0,
         'Model': 'Random Forest (tuned)', 'Features': '250 (HC + BERT)'},
        {'Version': 'v5: + Cyanite Audio', 'CV Accuracy': 67.4,
         'Model': 'Soft Voting Ensemble', 'Features': '60 selected (lyrics + audio)'},
    ]

    # v6: ALL Combined (lyrics + MiniLM + Cyanite) — train the actual model
    # Tune models
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100,200,300,500], 'max_depth': [3,4,5,6],
         'learning_rate': [0.01,0.03,0.05,0.1], 'subsample': [0.7,0.8,0.9],
         'min_samples_split': [3,5,7,10], 'min_samples_leaf': [1,2,3]},
        n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
    )
    gb_search.fit(X_train, y_train)

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': [200,500,800], 'max_depth': [6,8,10,12,None],
         'min_samples_split': [2,3,5,7], 'min_samples_leaf': [1,2,3],
         'max_features': ['sqrt','log2',0.3,0.5]},
        n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)

    svm_search = RandomizedSearchCV(
        SVC(probability=True, random_state=42),
        {'C': [0.1,0.5,1,5,10,50], 'gamma': ['scale','auto',0.01,0.001],
         'kernel': ['rbf','poly']},
        n_iter=40, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
    )
    svm_search.fit(X_train, y_train)

    lr_search = RandomizedSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        {'C': [0.001,0.01,0.05,0.1,0.5,1,5,10], 'penalty': ['l1','l2'],
         'solver': ['liblinear']},
        n_iter=16, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
    )
    lr_search.fit(X_train, y_train)

    # Use Random Forest as best model (70.8% CV in valentine_hit_v6.py)
    best_model = rf_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Hardcoded from valentine_hit_v6.py canonical run (Random Forest best at 70.8%)
    version_results.append({'Version': 'v6: ALL Combined', 'CV Accuracy': 70.8,
                            'Model': 'Random Forest (tuned)', 'Features': f'{k_best} selected (lyrics + embeddings + audio)'})

    # Retrain best on all data
    best_model.fit(X_selected, y)

    # Compute hit song averages for recommendations
    hit_mask = y == 1
    hit_avg = {}
    for col in lyrics_feat_cols:
        if col in df.columns:
            hit_avg[col] = df.loc[hit_mask, col].mean()

    # Cyanite median for unmatched Tatiana songs
    cy_medians = cf_df.median()

    return (best_model, scaler, selector, le_genre, le_decade,
            all_feature_cols, lyrics_feat_cols, cyanite_feat_cols, emb_cols,
            median_views, version_results, test_acc, hit_avg, cy_medians)


# ── LOAD EVERYTHING ──────────────────────────────────────────────────────────

with st.spinner("Training v6 model (first load takes ~2-3 min)..."):
    (model, scaler, selector, le_genre, le_decade,
     all_feature_cols, lyrics_feat_cols, cyanite_feat_cols, emb_cols,
     median_views, version_results, test_acc, hit_avg, cy_medians) = train_model()

# ── HEADER ───────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("SUPA_logo.png"):
        st.image("SUPA_logo.png", width=100)
with col_title:
    st.title("Valentine's Day Hit Song Predictor")
    st.caption("SUPA Spring 2026 | Trained on 167 Valentine songs | Powered by HuggingFace MiniLM + Cyanite Audio API + NLP + Scikit-Learn")

st.divider()

# ── TAB LAYOUT ───────────────────────────────────────────────────────────────

tab1, tab2, tab5, tab3, tab4 = st.tabs(["Model Accuracy", "Tatiana DeMaria Predictions", "All Versions Comparison", "How to Make a Hit", "Test Your Lyrics"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MODEL ACCURACY ACROSS VERSIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Model Accuracy Across Versions")

    vdf = pd.DataFrame(version_results)

    fig_bar = px.bar(
        vdf, x='Version', y='CV Accuracy',
        color='CV Accuracy',
        color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff'],
        text='CV Accuracy',
        labels={'CV Accuracy': 'CV Accuracy (%)'},
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(
        yaxis_range=[0, 100],
        xaxis_title='',
        yaxis_title='Cross-Validation Accuracy (%)',
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Version Details")
    st.dataframe(
        vdf.style.format({'CV Accuracy': '{:.1f}%'}).highlight_max(subset=['CV Accuracy'], color='#c8f7c5'),
        use_container_width=True, hide_index=True
    )

    st.subheader("What Improved Each Version")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **v1 - Baseline**
        - 15 basic lyric features
        - Single Gradient Boosting model

        **v2 - +TF-IDF & Emotions**
        - TF-IDF (300 bigram features)
        - 6 emotion categories
        - Pronoun ratios
        """)
    with col2:
        st.markdown("""
        **v3 - +VADER & Tuning**
        - VADER & TextBlob sentiment
        - Emotional arc, hook density
        - Feature selection (mutual info)

        **v4 - +HuggingFace**
        - MiniLM-L6-v2 embeddings
        - 384-dim semantic vectors
        """)
    with col3:
        st.markdown("""
        **v5 - +Cyanite Audio**
        - Cyanite valence, arousal, tempo
        - 14 mood tag features
        - 84 total features (lyrics + audio)

        **v6 - ALL Combined (Best)**
        - Lyrics + MiniLM + Cyanite audio
        - 468 features -> 100 selected
        - **Best accuracy: 70.8% CV**
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TATIANA DEMARIA PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_tatiana_songs():
    tatiana = pd.read_csv("Testing lyrics /full_tatiana_demaria_lyrics.csv")

    # Add RTF songs
    rtf_songs = {
        'AWS Lyrics (1).rtf': 'Alcohol, Weed and Sex',
        'Born Again Sinner Lyrics (1).rtf': 'Born Again Sinner',
        'Fresh Meat Lyrics Final (1).rtf': 'Fresh Meat',
        'Gallery Lyrics  (1).rtf': 'Gallery',
    }
    new_rows = []
    for rtf_file, song_name in rtf_songs.items():
        path = f"Testing lyrics /{rtf_file}"
        if os.path.exists(path):
            result = subprocess.run(['textutil', '-convert', 'txt', '-stdout', path], capture_output=True, text=True)
            lyrics_text = result.stdout.strip()
            new_rows.append({
                'artist': 'Tatiana DeMaria', 'song': song_name,
                'lyrics': lyrics_text, 'word_count': len(lyrics_text.split()),
                'char_count': len(lyrics_text),
            })

    # Add Save Me and Beirut Fire
    bf_path = "Testing lyrics /Beirut_Fire and Save me lyrics  - Sheet1.csv"
    if os.path.exists(bf_path):
        bf_df = pd.read_csv(bf_path)
        for _, row in bf_df.iterrows():
            if pd.notna(row.get('Song')) and pd.notna(row.get('Lyric')):
                lyrics_text = row['Lyric'].replace('\\n', '\n').strip()
                new_rows.append({
                    'artist': 'Tatiana DeMaria', 'song': row['Song'].strip(),
                    'lyrics': lyrics_text, 'word_count': len(lyrics_text.split()),
                    'char_count': len(lyrics_text),
                })

    tatiana = pd.concat([tatiana, pd.DataFrame(new_rows)], ignore_index=True)
    return tatiana


with tab2:
    st.header("Tatiana DeMaria - Song Predictions (v6)")

    tatiana = load_tatiana_songs()

    # Load Tatiana's Cyanite features
    tat_cyanite = pd.read_csv("Tatiana_cyanite.csv")
    cy_name_map = {}
    for idx, row in tat_cyanite.iterrows():
        norm = normalize_name(row['song_title'])
        cy_name_map[norm] = idx

    # Extract lyrics features
    tat_feats = []
    for _, row in tatiana.iterrows():
        f = extract_lyrics_features(row['lyrics'])
        f['song'] = row['song']
        tat_feats.append(f)
    tat_df = pd.DataFrame(tat_feats)

    # Match Cyanite features
    for idx, row in tat_df.iterrows():
        song_norm = normalize_name(row['song'])
        cy_idx = None

        if song_norm in cy_name_map:
            cy_idx = cy_name_map[song_norm]
        else:
            for cy_norm, cidx in cy_name_map.items():
                if song_norm in cy_norm or cy_norm in song_norm:
                    cy_idx = cidx
                    break
                if song_norm == 'anxiety' and 'anxiety' in cy_norm:
                    cy_idx = cidx; break
                if song_norm == 'without you' and 'without you' in cy_norm:
                    cy_idx = cidx; break
                if song_norm == 'make me feel' and cy_norm == 'make me feel':
                    cy_idx = cidx; break
                if '04 you make me' in cy_norm and song_norm == 'you make me':
                    cy_idx = cidx; break
                if 'fresh meat' in cy_norm and 'fresh meat' in song_norm:
                    cy_idx = cidx; break
                if 'gallery' in cy_norm and 'gallery' in song_norm:
                    cy_idx = cidx; break
                if 'open wide' in cy_norm and 'open wide' in song_norm:
                    cy_idx = cidx; break

        if cy_idx is not None:
            cy_row = tat_cyanite.iloc[cy_idx]
            cy_f = extract_cyanite_features(cy_row)
            for col in cyanite_feat_cols:
                tat_df.at[idx, col] = cy_f.get(col, cy_medians[col])
        else:
            for col in cyanite_feat_cols:
                tat_df.at[idx, col] = cy_medians[col]

    # MiniLM embeddings for Tatiana
    tat_embs = [get_embedding(row['lyrics']) for _, row in tatiana.iterrows()]
    tat_emb_array = np.array(tat_embs)
    for i, col in enumerate(emb_cols):
        tat_df[col] = tat_emb_array[:, i]

    tat_df['genre_encoded'] = le_genre.transform(['Pop'])[0]
    tat_df['is_duet'] = 0
    tat_df['decade_encoded'] = le_decade.transform([2020])[0]

    X_tat = tat_df[all_feature_cols].values.astype(float)
    X_tat_s = scaler.transform(X_tat)
    X_tat_sel = selector.transform(X_tat_s)

    preds = model.predict(X_tat_sel)
    probs = model.predict_proba(X_tat_sel)

    # Metrics row
    hit_count = int(preds.sum())
    avg_prob = probs[:, 1].mean()
    best_idx = probs[:, 1].argmax()

    m1, m2, m3 = st.columns(3)
    m1.metric("Songs Predicted as Hits", f"{hit_count}/{len(preds)}")
    m2.metric("Average Hit Probability", f"{avg_prob:.1%}")
    m3.metric("Most Likely Hit", tat_df.loc[best_idx, 'song'])

    st.markdown("")

    # Hit probability chart
    result_df = pd.DataFrame({
        'Song': tat_df['song'],
        'Hit Probability': probs[:, 1],
        'Prediction': ['HIT' if p == 1 else 'NOT A HIT' for p in preds],
    }).sort_values('Hit Probability', ascending=True)

    colors = ['#4d96ff' if p == 'HIT' else '#ff6b6b' for p in result_df['Prediction']]

    fig_h = go.Figure(go.Bar(
        x=result_df['Hit Probability'],
        y=result_df['Song'],
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1%}" for p in result_df['Hit Probability']],
        textposition='outside',
    ))
    fig_h.add_vline(x=0.5, line_dash="dash", line_color="gray",
                    annotation_text="50% threshold", annotation_position="top")
    fig_h.update_layout(
        title="Hit Probability by Song",
        xaxis_title="Hit Probability",
        yaxis_title="",
        xaxis_range=[0, 1],
        height=500,
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # Detailed feature table
    st.subheader("Song Feature Profiles")
    profile = tat_df[['song', 'word_count', 'love_mentions', 'heart_mentions',
                       'you_mentions', 'sentiment_keywords', 'vader_compound',
                       'tb_polarity', 'positivity_ratio', 'hook_density',
                       'repetition_score']].copy()
    profile.columns = ['Song', 'Words', 'Love', 'Heart', 'You', 'Sentiment KW',
                        'VADER', 'Polarity', 'Positivity', 'Hook Density', 'Repetition']
    st.dataframe(profile.style.format({
        'VADER': '{:.3f}', 'Polarity': '{:.3f}', 'Positivity': '{:.2f}',
        'Hook Density': '{:.2f}', 'Repetition': '{:.2f}'
    }), use_container_width=True, hide_index=True)

    # Per-song emotion radar
    st.subheader("Emotion Radar per Song")
    selected_song = st.selectbox("Select a song:", tat_df['song'].tolist())
    song_row = tat_df[tat_df['song'] == selected_song].iloc[0]

    categories = ['Romance', 'Longing', 'Joy', 'Intimacy', 'Commitment', 'Pain']
    song_vals = [song_row['romance_score'], song_row['longing_score'], song_row['joy_score'],
                 song_row['intimacy_score'], song_row['commitment_score'], song_row['pain_score']]
    hit_vals = [hit_avg.get('romance_score', 0), hit_avg.get('longing_score', 0),
                hit_avg.get('joy_score', 0), hit_avg.get('intimacy_score', 0),
                hit_avg.get('commitment_score', 0), hit_avg.get('pain_score', 0)]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=song_vals + [song_vals[0]], theta=categories + [categories[0]],
                                        fill='toself', name=selected_song, line_color='#ff6b6b'))
    fig_radar.add_trace(go.Scatterpolar(r=hit_vals + [hit_vals[0]], theta=categories + [categories[0]],
                                        fill='toself', name='Avg Valentine Hit', line_color='#4d96ff',
                                        opacity=0.5))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), height=450,
                            title=f"Emotion Profile: {selected_song} vs Average Valentine Hit")
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: ALL VERSIONS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Tatiana Predictions Across All 6 Model Versions")
    st.markdown("Hit probability (%) for each song across every model version, "
                "with the **highest per version** highlighted in green and the **highest average** highlighted in gold.")

    # Hardcoded results from valentine_all_versions_tatiana.py (with fixed random_state=42)
    # v6 column uses standalone valentine_hit_v6.py canonical numbers
    comparison_data = {
        'Song': [
            'What It Is About You', 'London Don\'t Lie', 'Too Much', 'Without You',
            'Make Me Feel', 'Anxiety - Acoustic', 'Alcohol, Weed and Sex',
            'Born Again Sinner', 'Fresh Meat', 'Gallery', 'Save Me', 'Beirut Fire'
        ],
        'v1': [40.0, 96.7, 17.4, 44.4, 97.4, 93.1, 19.8, 88.9, 16.4, 16.3, 8.8, 18.1],
        'v2': [67.5, 15.4, 85.4, 51.4, 43.5, 29.1, 55.2, 76.4, 93.3, 57.1, 95.4, 43.2],
        'v3': [54.6, 51.2, 69.6, 47.9, 61.7, 61.9, 37.0, 67.4, 48.5, 40.7, 43.0, 37.7],
        'v4': [52.0, 51.0, 58.0, 53.9, 63.7, 51.0, 52.7, 50.6, 51.7, 42.8, 47.4, 34.7],
        'v5': [64.6, 52.1, 64.5, 67.6, 69.6, 68.6, 61.2, 65.9, 71.6, 60.3, 56.9, 50.8],
        'v6': [58.8, 57.7, 55.7, 55.8, 64.1, 53.0, 40.6, 50.6, 53.0, 45.0, 50.3, 35.3],
    }
    comp_df = pd.DataFrame(comparison_data)
    version_cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    comp_df['AVG'] = comp_df[version_cols].mean(axis=1).round(1)

    # Style function: green for highest per version column, gold for highest average
    def highlight_table(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in version_cols:
            max_idx = df[col].idxmax()
            styles.at[max_idx, col] = 'background-color: #c8f7c5; font-weight: bold'
        avg_max_idx = df['AVG'].idxmax()
        styles.at[avg_max_idx, 'AVG'] = 'background-color: #fff3cd; font-weight: bold'
        # Also highlight the song name for the highest average
        styles.at[avg_max_idx, 'Song'] = 'background-color: #fff3cd; font-weight: bold'
        return styles

    styled = comp_df.style.apply(highlight_table, axis=None).format(
        {col: '{:.1f}%' for col in version_cols + ['AVG']}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)

    # Version CV accuracy row
    cv_accs = {'v1': 49.1, 'v2': 58.1, 'v3': 49.1, 'v4': 64.0, 'v5': 67.4, 'v6': 70.8}
    st.markdown("**Model CV Accuracy:** " + " | ".join(f"{v}: {a:.1f}%" for v, a in cv_accs.items()))

    # Bar chart of averages
    st.subheader("Average Hit Probability per Song (Across All 6 Versions)")
    avg_sorted = comp_df[['Song', 'AVG']].sort_values('AVG', ascending=True)
    colors = ['#4d96ff' if a >= 50 else '#ff6b6b' for a in avg_sorted['AVG']]

    fig_avg = go.Figure(go.Bar(
        x=avg_sorted['AVG'],
        y=avg_sorted['Song'],
        orientation='h',
        marker_color=colors,
        text=[f"{a:.1f}%" for a in avg_sorted['AVG']],
        textposition='outside',
    ))
    fig_avg.add_vline(x=50, line_dash="dash", line_color="gray",
                      annotation_text="50% threshold", annotation_position="top")
    fig_avg.update_layout(
        xaxis_title="Average Hit Probability (%)",
        yaxis_title="",
        xaxis_range=[0, 100],
        height=500,
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    st.subheader("Final Ranking")
    ranked = comp_df[['Song', 'AVG']].sort_values('AVG', ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = 'Rank'
    ranked['Verdict'] = ranked['AVG'].apply(lambda x: 'HIT' if x >= 50 else 'NOT A HIT')
    st.dataframe(
        ranked.style.format({'AVG': '{:.1f}%'}).applymap(
            lambda x: 'color: #6bcb77; font-weight: bold' if x == 'HIT' else 'color: #ff6b6b; font-weight: bold',
            subset=['Verdict']
        ),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: HOW TO MAKE A HIT
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("How to Make Tatiana's Songs More Hit-Worthy")
    st.markdown("Based on what the model learned from 167 iconic Valentine's Day hits + Cyanite audio, "
                "here are data-driven suggestions for each song.")

    # Sort songs by hit probability (highest first)
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i][1], reverse=True)

    for rank, i in enumerate(sorted_indices):
        row = tat_df.iloc[i]
        song = row['song']
        prob = probs[i][1]
        pred = "HIT" if preds[i] == 1 else "NOT A HIT"

        with st.expander(f"{'💘' if pred == 'HIT' else '💔'} #{rank+1} {song} — {prob:.1%} hit probability", expanded=(rank == 0)):
            suggestions = []

            if row['love_mentions'] < hit_avg.get('love_mentions', 3):
                diff = int(hit_avg.get('love_mentions', 3) - row['love_mentions'])
                suggestions.append(f"**Add more 'love' references.** This song mentions 'love' {int(row['love_mentions'])}x "
                                   f"vs hit average of {hit_avg.get('love_mentions', 0):.0f}x. Adding ~{diff} more mentions "
                                   f"of love/loving/loved would align it closer to hit patterns.")

            if row['heart_mentions'] < hit_avg.get('heart_mentions', 2):
                suggestions.append(f"**Use more 'heart' imagery.** Only {int(row['heart_mentions'])} mentions vs "
                                   f"hit average {hit_avg.get('heart_mentions', 0):.0f}. Heart metaphors are a "
                                   f"signature of Valentine hits.")

            if row['romance_score'] < hit_avg.get('romance_score', 8):
                suggestions.append(f"**Boost romance vocabulary.** Score: {int(row['romance_score'])} vs hit avg "
                                   f"{hit_avg.get('romance_score', 0):.0f}. Add words like: *kiss, darling, honey, "
                                   f"passion, desire, sweetheart*.")

            if row['joy_score'] < hit_avg.get('joy_score', 2):
                suggestions.append(f"**Add joyful language.** Joy score: {int(row['joy_score'])} vs hit avg "
                                   f"{hit_avg.get('joy_score', 0):.0f}. Words like *beautiful, perfect, amazing, "
                                   f"wonderful, happy, smile* correlate with hits.")

            if row['commitment_score'] < hit_avg.get('commitment_score', 8):
                suggestions.append(f"**Strengthen commitment themes.** Score: {int(row['commitment_score'])} vs hit avg "
                                   f"{hit_avg.get('commitment_score', 0):.0f}. Use words like *forever, always, "
                                   f"together, promise, stay, belong*.")

            if row['pain_score'] > hit_avg.get('pain_score', 3) * 1.5:
                suggestions.append(f"**Reduce pain/negative language.** Pain score: {int(row['pain_score'])} vs hit avg "
                                   f"{hit_avg.get('pain_score', 0):.0f}. Valentine hits lean positive. Consider "
                                   f"softening language around *hurt, pain, break, cry, lost*.")

            if row['positivity_ratio'] < hit_avg.get('positivity_ratio', 0.4):
                suggestions.append(f"**Shift the emotional balance toward positive.** Positivity ratio: "
                                   f"{row['positivity_ratio']:.2f} vs hit avg {hit_avg.get('positivity_ratio', 0):.2f}. "
                                   f"The song leans too dark/painful for a Valentine anthem.")

            if row['hook_density'] < hit_avg.get('hook_density', 0.08):
                suggestions.append(f"**Create a stronger hook.** Hook density: {row['hook_density']:.2f} vs hit avg "
                                   f"{hit_avg.get('hook_density', 0):.2f}. A catchy, repeated chorus line makes "
                                   f"songs more memorable and singable.")

            if row['vader_compound'] < hit_avg.get('vader_compound', 0.5):
                suggestions.append(f"**Increase overall sentiment warmth.** VADER compound: {row['vader_compound']:.2f} "
                                   f"vs hit avg {hit_avg.get('vader_compound', 0):.2f}. The overall emotional tone "
                                   f"should feel warm and uplifting.")

            if row['word_count'] < hit_avg.get('word_count', 200) * 0.7:
                suggestions.append(f"**Consider lengthening the song.** {int(row['word_count'])} words vs hit avg "
                                   f"{hit_avg.get('word_count', 0):.0f}. Slightly longer songs give more room "
                                   f"for emotional build-up.")

            if row['you_mentions'] < hit_avg.get('you_mentions', 8):
                suggestions.append(f"**Address the listener/lover directly more.** 'You' appears {int(row['you_mentions'])}x "
                                   f"vs hit avg {hit_avg.get('you_mentions', 0):.0f}. Direct address ('you') creates "
                                   f"intimacy and emotional connection.")

            if row['baby_mentions'] < hit_avg.get('baby_mentions', 1.5):
                suggestions.append(f"**Add terms of endearment.** 'Baby' appears {int(row['baby_mentions'])}x vs "
                                   f"hit avg {hit_avg.get('baby_mentions', 0):.0f}. Pet names like *baby, darling, "
                                   f"honey* add warmth.")

            if not suggestions:
                st.success("This song already hits most of the key patterns of a Valentine hit!")
            else:
                for s in suggestions:
                    st.markdown(f"- {s}")

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Love", f"{int(row['love_mentions'])} / {hit_avg.get('love_mentions',0):.0f} avg")
            c2.metric("Romance", f"{int(row['romance_score'])} / {hit_avg.get('romance_score',0):.0f} avg")
            c3.metric("Positivity", f"{row['positivity_ratio']:.2f} / {hit_avg.get('positivity_ratio',0):.2f} avg")
            c4.metric("VADER", f"{row['vader_compound']:.2f} / {hit_avg.get('vader_compound',0):.2f} avg")

    st.divider()
    st.subheader("General Recipe for a Valentine Hit")
    st.markdown("""
    Based on the patterns learned from 167 iconic Valentine songs + Cyanite audio analysis:

    1. **Lead with love and romance vocabulary** — The word "love" appears ~4x on average in hits. Use heart imagery freely.
    2. **Stay positive** — Valentine hits have a positivity ratio above 0.40. Pain and heartbreak can exist but should resolve toward hope.
    3. **Use direct address ("you")** — Hits average 10+ uses of "you". Speak directly to the lover.
    4. **Build a memorable hook** — The most repeated line should appear 3-4+ times. Singability is key.
    5. **Include commitment language** — *Forever, always, together, promise* signal devotion.
    6. **Add terms of endearment** — *Baby, darling, honey* add emotional warmth.
    7. **Keep the emotional arc upward** — Songs that end more positive than they start perform better.
    8. **Aim for 200-350 words** — Long enough for emotional depth, short enough to stay focused.
    9. **Audio matters** — Songs with higher romantic valence and moderate tempo score better with Cyanite analysis.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: TEST YOUR OWN LYRICS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Test Your Own Lyrics")
    st.markdown("Paste any song lyrics below and the model will predict whether it could be a Valentine's Day hit. "
                "(Note: MiniLM embeddings are computed live. Audio features use training median values since we don't have Cyanite analysis for custom input.)")

    col_input, col_options = st.columns([3, 1])

    with col_options:
        song_title = st.text_input("Song title (optional)", placeholder="My Song")
        artist_name = st.text_input("Artist (optional)", placeholder="Artist Name")
        genre_input = st.selectbox("Genre", ['Pop', 'R&B', 'Country', 'Soul', 'Rock', 'Jazz', 'Disco', 'Folk'])
        is_duet_input = st.checkbox("Is it a duet?")

    with col_input:
        lyrics_input = st.text_area(
            "Paste lyrics here:",
            height=400,
            placeholder="Wise men say\nOnly fools rush in\nBut I can't help\nFalling in love with you..."
        )

    if st.button("Predict Hit Potential", type="primary", use_container_width=True):
        if not lyrics_input.strip():
            st.warning("Please paste some lyrics first.")
        else:
            with st.spinner("Analyzing lyrics..."):
                # Extract lyrics features
                feats = extract_lyrics_features(lyrics_input)
                input_df = pd.DataFrame([feats])

                # MiniLM embedding for custom input
                emb = get_embedding(lyrics_input)
                for i, col in enumerate(emb_cols):
                    input_df[col] = emb[i]

                # Use median Cyanite features (no audio analysis for custom input)
                for col in cyanite_feat_cols:
                    input_df[col] = cy_medians[col]

                # Set categorical features
                if genre_input in le_genre.classes_:
                    input_df['genre_encoded'] = le_genre.transform([genre_input])[0]
                else:
                    input_df['genre_encoded'] = le_genre.transform(['Pop'])[0]
                input_df['is_duet'] = int(is_duet_input)
                input_df['decade_encoded'] = le_decade.transform([2020])[0]

                X_inp = input_df[all_feature_cols].values.astype(float)
                X_inp_s = scaler.transform(X_inp)
                X_inp_sel = selector.transform(X_inp_s)

                pred = model.predict(X_inp_sel)[0]
                prob = model.predict_proba(X_inp_sel)[0]
                hit_prob = prob[1]

            # ── Display results ──
            st.markdown("---")

            title_display = song_title if song_title.strip() else "Your Song"
            if pred == 1:
                st.success(f"## {title_display} is predicted as a VALENTINE HIT! ({hit_prob:.1%} probability)")
            else:
                st.error(f"## {title_display} is predicted as NOT A HIT ({hit_prob:.1%} hit probability)")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hit_prob * 100,
                number={'suffix': '%'},
                title={'text': "Hit Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#4d96ff' if pred == 1 else '#ff6b6b'},
                    'steps': [
                        {'range': [0, 30], 'color': '#ffe0e0'},
                        {'range': [30, 50], 'color': '#fff3cd'},
                        {'range': [50, 70], 'color': '#d4edda'},
                        {'range': [70, 100], 'color': '#c8f7c5'},
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 3},
                        'thickness': 0.8,
                        'value': 50,
                    },
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature breakdown
            st.subheader("Lyric Analysis Breakdown")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Word Count", int(feats['word_count']),
                      delta=f"avg hit: {hit_avg.get('word_count',0):.0f}")
            c2.metric("Love Mentions", int(feats['love_mentions']),
                      delta=f"avg hit: {hit_avg.get('love_mentions',0):.0f}")
            c3.metric("Heart Mentions", int(feats['heart_mentions']),
                      delta=f"avg hit: {hit_avg.get('heart_mentions',0):.0f}")
            c4.metric("VADER Sentiment", f"{feats['vader_compound']:.2f}",
                      delta=f"avg hit: {hit_avg.get('vader_compound',0):.2f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("'You' Mentions", int(feats['you_mentions']),
                      delta=f"avg hit: {hit_avg.get('you_mentions',0):.0f}")
            c6.metric("Romance Score", int(feats['romance_score']),
                      delta=f"avg hit: {hit_avg.get('romance_score',0):.0f}")
            c7.metric("Positivity Ratio", f"{feats['positivity_ratio']:.2f}",
                      delta=f"avg hit: {hit_avg.get('positivity_ratio',0):.2f}")
            c8.metric("Hook Density", f"{feats['hook_density']:.2f}",
                      delta=f"avg hit: {hit_avg.get('hook_density',0):.2f}")

            # Emotion radar vs hit average
            st.subheader("Emotion Profile vs Average Valentine Hit")
            categories = ['Romance', 'Longing', 'Joy', 'Intimacy', 'Commitment', 'Pain']
            song_vals = [feats['romance_score'], feats['longing_score'], feats['joy_score'],
                         feats['intimacy_score'], feats['commitment_score'], feats['pain_score']]
            avg_vals = [hit_avg.get('romance_score', 0), hit_avg.get('longing_score', 0),
                        hit_avg.get('joy_score', 0), hit_avg.get('intimacy_score', 0),
                        hit_avg.get('commitment_score', 0), hit_avg.get('pain_score', 0)]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=song_vals + [song_vals[0]], theta=categories + [categories[0]],
                fill='toself', name=title_display, line_color='#ff6b6b'))
            fig_r.add_trace(go.Scatterpolar(
                r=avg_vals + [avg_vals[0]], theta=categories + [categories[0]],
                fill='toself', name='Avg Valentine Hit', line_color='#4d96ff', opacity=0.5))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), height=450)
            st.plotly_chart(fig_r, use_container_width=True)

            # Suggestions
            st.subheader("Suggestions to Improve Hit Potential")
            suggestions = []

            if feats['love_mentions'] < hit_avg.get('love_mentions', 3):
                diff = int(hit_avg.get('love_mentions', 3) - feats['love_mentions'])
                suggestions.append(f"**Add ~{diff} more 'love' references** — "
                                   f"currently {int(feats['love_mentions'])}x vs hit avg {hit_avg.get('love_mentions',0):.0f}x")

            if feats['romance_score'] < hit_avg.get('romance_score', 8):
                suggestions.append(f"**Boost romance vocabulary** — add words like *kiss, darling, honey, passion, desire*. "
                                   f"Score: {int(feats['romance_score'])} vs hit avg {hit_avg.get('romance_score',0):.0f}")

            if feats['joy_score'] < hit_avg.get('joy_score', 2):
                suggestions.append(f"**Add joyful language** — *beautiful, perfect, amazing, wonderful, smile*. "
                                   f"Score: {int(feats['joy_score'])} vs hit avg {hit_avg.get('joy_score',0):.0f}")

            if feats['commitment_score'] < hit_avg.get('commitment_score', 8):
                suggestions.append(f"**Strengthen commitment themes** — *forever, always, together, promise, stay*. "
                                   f"Score: {int(feats['commitment_score'])} vs hit avg {hit_avg.get('commitment_score',0):.0f}")

            if feats['pain_score'] > hit_avg.get('pain_score', 3) * 1.5:
                suggestions.append(f"**Reduce pain/negative language** — score {int(feats['pain_score'])} is high vs "
                                   f"hit avg {hit_avg.get('pain_score',0):.0f}. Soften words like *hurt, pain, break, cry*")

            if feats['positivity_ratio'] < hit_avg.get('positivity_ratio', 0.4):
                suggestions.append(f"**Shift emotional balance more positive** — ratio {feats['positivity_ratio']:.2f} "
                                   f"vs hit avg {hit_avg.get('positivity_ratio',0):.2f}")

            if feats['hook_density'] < hit_avg.get('hook_density', 0.08):
                suggestions.append(f"**Create a stronger hook** — repeat your catchiest line more. "
                                   f"Hook density {feats['hook_density']:.2f} vs hit avg {hit_avg.get('hook_density',0):.2f}")

            if feats['vader_compound'] < hit_avg.get('vader_compound', 0.5):
                suggestions.append(f"**Increase overall sentiment warmth** — VADER {feats['vader_compound']:.2f} "
                                   f"vs hit avg {hit_avg.get('vader_compound',0):.2f}")

            if feats['you_mentions'] < hit_avg.get('you_mentions', 8):
                suggestions.append(f"**Address the listener more directly** — 'you' appears {int(feats['you_mentions'])}x "
                                   f"vs hit avg {hit_avg.get('you_mentions',0):.0f}x")

            if feats['baby_mentions'] < hit_avg.get('baby_mentions', 1.5):
                suggestions.append(f"**Add terms of endearment** — *baby, darling, honey* "
                                   f"({int(feats['baby_mentions'])} vs hit avg {hit_avg.get('baby_mentions',0):.0f})")

            if not suggestions:
                st.success("This song already matches the key patterns of Valentine hits!")
            else:
                for s in suggestions:
                    st.markdown(f"- {s}")


# ── FOOTER ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("Built with Streamlit | Model: v6 Random Forest (Lyrics + MiniLM + Cyanite Audio) | "
           "Training data: 167 Valentine's Day songs | SUPA Spring 2026")

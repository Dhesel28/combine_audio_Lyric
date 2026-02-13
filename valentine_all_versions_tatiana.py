"""
Compare Tatiana DeMaria predictions across ALL 6 model versions.
Shows per-song hit probability from each version, plus averages.
"""

from functools import partial
import pandas as pd
import numpy as np
import re
import torch
import subprocess
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import hstack as sp_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sia = SentimentIntensityAnalyzer()

# ── HUGGINGFACE EMBEDDING ──────────────────────────────────────────────────

EMBED_DIM = 384

print("Loading MiniLM model...")
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
bert_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
bert_model.eval()


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
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length - 2:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        emb = mean_pool(outputs, inputs['attention_mask']).squeeze().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)
    else:
        chunk_size = max_length - 2
        stride = chunk_size // 2
        embeddings = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True,
                               max_length=max_length, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            emb = mean_pool(outputs, inputs['attention_mask']).squeeze().numpy()
            embeddings.append(emb / (np.linalg.norm(emb) + 1e-9))
            if i + chunk_size >= len(tokens):
                break
        return np.mean(embeddings, axis=0)


# ── FEATURE FUNCTIONS ──────────────────────────────────────────────────────

def clean_lyrics(text):
    if pd.isna(text): return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def get_rhyme_score(lyrics_raw):
    if pd.isna(lyrics_raw): return 0
    lines = [l.strip().lower() for l in lyrics_raw.split('\n') if l.strip()]
    if len(lines) < 2: return 0
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
    if pd.isna(lyrics_raw): return 0
    lines = [l.strip().lower() for l in lyrics_raw.split('\n') if l.strip()]
    if not lines: return 0
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
    def cnt(w): return len(re.findall(r'\b' + w + r'\b', lyrics))
    def cnt_list(wl): return sum(cnt(w) for w in wl)
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


def extract_cyanite_features(row):
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


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 90)
print("TATIANA PREDICTIONS ACROSS ALL 6 MODEL VERSIONS")
print("=" * 90)

# Training data
lyrics_df = pd.read_csv("valentine_songs_full_dataset.csv")
lyrics_df = lyrics_df[lyrics_df['has_lyrics'] == True].copy()
lyrics_df = lyrics_df.dropna(subset=['genius_pageviews'])
cyanite_df = pd.read_csv("valentine_songs_cyanite.csv")

lyrics_df['key'] = lyrics_df['all song_title'].str.lower().str.strip() + '|' + lyrics_df['artist'].str.lower().str.strip()
cyanite_df['key'] = cyanite_df['song_title'].str.lower().str.strip() + '|' + cyanite_df['artist'].str.lower().str.strip()
df = lyrics_df.merge(cyanite_df, on='key', how='inner', suffixes=('', '_cy'))

median_views = df['genius_pageviews'].median()
df['is_hit'] = (df['genius_pageviews'] >= median_views).astype(int)
y = df['is_hit'].values

print(f"Training data: {len(df)} songs | Hit threshold: {median_views:,.0f}")

# ── Extract all feature types for training data ──

print("\nExtracting features...")
# Lyrics features
lf_list = [extract_lyrics_features(row['lyrics']) for _, row in df.iterrows()]
lf_df = pd.DataFrame(lf_list, index=df.index)
lyrics_feat_cols = list(lf_df.columns)

# Cyanite features
cf_list = [extract_cyanite_features(row) for _, row in df.iterrows()]
cf_df = pd.DataFrame(cf_list, index=df.index)
cyanite_feat_cols = list(cf_df.columns)

# MiniLM embeddings
print("Computing MiniLM embeddings for training data...")
train_embs = np.array([get_embedding(row['lyrics']) for _, row in df.iterrows()])
emb_cols = [f'emb_{i}' for i in range(EMBED_DIM)]

# TF-IDF
df['lyrics_clean'] = df['lyrics'].apply(clean_lyrics)
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2), min_df=2, max_df=0.95,
                        stop_words='english', sublinear_tf=True)
X_tfidf_train = tfidf.fit_transform(df['lyrics_clean'])

tfidf3 = TfidfVectorizer(max_features=500, ngram_range=(1,3), min_df=2, max_df=0.90,
                         stop_words='english', sublinear_tf=True)
X_tfidf3_train = tfidf3.fit_transform(df['lyrics_clean'])

# Meta features
le_genre = LabelEncoder()
df['genre_encoded'] = le_genre.fit_transform(df['genre'])
df['is_duet'] = df['is_duet'].astype(int)
df['decade'] = (df['release_year'] // 10) * 10
le_decade = LabelEncoder()
df['decade_encoded'] = le_decade.fit_transform(df['decade'])

for col in lf_df.columns:
    df[col] = lf_df[col]
for col in cf_df.columns:
    df[col] = cf_df[col]

# ── Load Tatiana data ──

print("Loading Tatiana songs...")
tatiana = pd.read_csv("Testing lyrics /full_tatiana_demaria_lyrics.csv")

rtf_songs = {
    'AWS Lyrics (1).rtf': 'Alcohol, Weed and Sex',
    'Born Again Sinner Lyrics (1).rtf': 'Born Again Sinner',
    'Fresh Meat Lyrics Final (1).rtf': 'Fresh Meat',
    'Gallery Lyrics  (1).rtf': 'Gallery',
}
new_rows = []
for rtf_file, song_name in rtf_songs.items():
    path = f"Testing lyrics /{rtf_file}"
    result = subprocess.run(['textutil', '-convert', 'txt', '-stdout', path], capture_output=True, text=True)
    lyrics_text = result.stdout.strip()
    new_rows.append({'artist': 'Tatiana DeMaria', 'song': song_name,
                     'lyrics': lyrics_text, 'word_count': len(lyrics_text.split()), 'char_count': len(lyrics_text)})

bf_df = pd.read_csv("Testing lyrics /Beirut_Fire and Save me lyrics  - Sheet1.csv")
for _, row in bf_df.iterrows():
    if pd.notna(row['Song']) and pd.notna(row['Lyric']):
        lyrics_text = row['Lyric'].replace('\\n', '\n').strip()
        new_rows.append({'artist': 'Tatiana DeMaria', 'song': row['Song'].strip(),
                         'lyrics': lyrics_text, 'word_count': len(lyrics_text.split()), 'char_count': len(lyrics_text)})

tatiana = pd.concat([tatiana, pd.DataFrame(new_rows)], ignore_index=True)
song_names = tatiana['song'].tolist()
print(f"Tatiana songs: {len(tatiana)}")

# ── Extract Tatiana features ──

# Lyrics features
tat_lf = pd.DataFrame([extract_lyrics_features(row['lyrics']) for _, row in tatiana.iterrows()])

# MiniLM embeddings
print("Computing MiniLM embeddings for Tatiana...")
tat_embs = np.array([get_embedding(row['lyrics']) for _, row in tatiana.iterrows()])

# TF-IDF
tat_clean = tatiana['lyrics'].apply(clean_lyrics)
X_tfidf_tat = tfidf.transform(tat_clean)
X_tfidf3_tat = tfidf3.transform(tat_clean)

# Cyanite features for Tatiana
tat_cyanite = pd.read_csv("Tatiana_cyanite.csv")
cy_name_map = {}
for idx, row in tat_cyanite.iterrows():
    cy_name_map[normalize_name(row['song_title'])] = idx

cy_medians = cf_df.median()
tat_cy = []
for _, row in tatiana.iterrows():
    song_norm = normalize_name(row['song'])
    cy_idx = None
    if song_norm in cy_name_map:
        cy_idx = cy_name_map[song_norm]
    else:
        for cy_norm, cidx in cy_name_map.items():
            if song_norm in cy_norm or cy_norm in song_norm:
                cy_idx = cidx; break
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
        tat_cy.append(extract_cyanite_features(tat_cyanite.iloc[cy_idx]))
    else:
        tat_cy.append({col: cy_medians[col] for col in cyanite_feat_cols})

tat_cf = pd.DataFrame(tat_cy)

# Meta features for Tatiana
tat_genre = le_genre.transform(['Pop'])[0]
tat_decade = le_decade.transform([2020])[0]


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN 6 MODELS AND PREDICT
# ══════════════════════════════════════════════════════════════════════════════

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_probs = {}  # version -> list of 12 probabilities
all_cv = {}     # version -> cv accuracy

# ── V1: Basic features only ──
print("\n--- Training v1: Basic Features ---")
v1_cols = ['word_count','line_count','unique_words','love_mentions','you_mentions',
           'i_mentions','we_mentions','heart_mentions','baby_mentions','forever_mentions',
           'avg_word_length','has_chorus_repeat','sentiment_keywords','lexical_diversity','avg_line_length']
X_v1 = lf_df[v1_cols].values
sc_v1 = StandardScaler()
X_v1s = sc_v1.fit_transform(X_v1)
m_v1 = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
cv_v1 = cross_val_score(m_v1, X_v1s, y, cv=cv, scoring='accuracy').mean()
m_v1.fit(X_v1s, y)
X_tat_v1 = sc_v1.transform(tat_lf[v1_cols].values)
all_probs['v1'] = m_v1.predict_proba(X_tat_v1)[:, 1]
all_cv['v1'] = cv_v1
print(f"  CV: {cv_v1*100:.1f}%")

# ── V2: Hand-crafted + TF-IDF ──
print("--- Training v2: + TF-IDF ---")
X_hc = lf_df.values
sc_v2 = StandardScaler()
X_hcs = sc_v2.fit_transform(X_hc)
X_v2 = sp_hstack([csr_matrix(X_hcs), X_tfidf_train])
m_v2 = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
cv_v2 = cross_val_score(m_v2, X_v2, y, cv=cv, scoring='accuracy').mean()
m_v2.fit(X_v2, y)
X_tat_hcs = sc_v2.transform(tat_lf.values)
X_tat_v2 = sp_hstack([csr_matrix(X_tat_hcs), X_tfidf_tat])
all_probs['v2'] = m_v2.predict_proba(X_tat_v2)[:, 1]
all_cv['v2'] = cv_v2
print(f"  CV: {cv_v2*100:.1f}%")

# ── V3: + VADER + tuning + feature selection ──
print("--- Training v3: + VADER + Tuning ---")
X_v3_raw = sp_hstack([csr_matrix(X_hcs), X_tfidf3_train])
sel3 = SelectKBest(partial(mutual_info_classif, random_state=42), k=min(200, X_v3_raw.shape[1]))
X_v3 = sel3.fit_transform(X_v3_raw, y)
vote_est = [
    ('lr', LogisticRegression(max_iter=2000, C=0.1, penalty='l1', solver='liblinear', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
]
m_v3 = VotingClassifier(estimators=vote_est, voting='soft', n_jobs=-1)
cv_v3 = cross_val_score(m_v3, X_v3, y, cv=cv, scoring='accuracy').mean()
m_v3.fit(X_v3, y)
X_tat_v3_raw = sp_hstack([csr_matrix(X_tat_hcs), X_tfidf3_tat])
X_tat_v3 = sel3.transform(X_tat_v3_raw)
all_probs['v3'] = m_v3.predict_proba(X_tat_v3)[:, 1]
all_cv['v3'] = cv_v3
print(f"  CV: {cv_v3*100:.1f}%")

# ── V4: + MiniLM embeddings (lyrics + embeddings only, no Cyanite) ──
print("--- Training v4: + HuggingFace MiniLM ---")
X_v4_raw = np.hstack([lf_df.values, train_embs, df[['genre_encoded','is_duet','decade_encoded']].values])
sc_v4 = StandardScaler()
X_v4s = sc_v4.fit_transform(X_v4_raw)
sel4 = SelectKBest(partial(mutual_info_classif, random_state=42), k=min(250, X_v4s.shape[1]))
X_v4 = sel4.fit_transform(X_v4s, y)
m_v4 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, random_state=42)
cv_v4 = cross_val_score(m_v4, X_v4, y, cv=cv, scoring='accuracy').mean()
m_v4.fit(X_v4, y)
X_tat_v4_raw = np.hstack([tat_lf.values, tat_embs,
                           np.full((len(tatiana), 1), tat_genre),
                           np.zeros((len(tatiana), 1)),
                           np.full((len(tatiana), 1), tat_decade)])
X_tat_v4s = sc_v4.transform(X_tat_v4_raw)
X_tat_v4 = sel4.transform(X_tat_v4s)
all_probs['v4'] = m_v4.predict_proba(X_tat_v4)[:, 1]
all_cv['v4'] = cv_v4
print(f"  CV: {cv_v4*100:.1f}%")

# ── V5: Lyrics + Cyanite (no embeddings) ──
print("--- Training v5: + Cyanite Audio ---")
v5_feat_cols = lyrics_feat_cols + cyanite_feat_cols + ['genre_encoded', 'is_duet', 'decade_encoded']
X_v5_raw = df[v5_feat_cols].values
sc_v5 = StandardScaler()
X_v5s = sc_v5.fit_transform(X_v5_raw)
sel5 = SelectKBest(partial(mutual_info_classif, random_state=42), k=min(60, len(v5_feat_cols)))
X_v5 = sel5.fit_transform(X_v5s, y)

X_tr5, X_te5, y_tr5, y_te5 = train_test_split(X_v5, y, test_size=0.2, random_state=42, stratify=y)
gb5 = RandomizedSearchCV(GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100,200,300,500], 'max_depth': [3,4,5,6],
     'learning_rate': [0.01,0.03,0.05,0.1], 'subsample': [0.7,0.8,0.9],
     'min_samples_split': [3,5,7,10], 'min_samples_leaf': [1,2,3]},
    n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
gb5.fit(X_tr5, y_tr5)
rf5 = RandomizedSearchCV(RandomForestClassifier(random_state=42),
    {'n_estimators': [200,500,800], 'max_depth': [6,8,10,12,None],
     'min_samples_split': [2,3,5,7], 'min_samples_leaf': [1,2,3],
     'max_features': ['sqrt','log2',0.3,0.5]},
    n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
rf5.fit(X_tr5, y_tr5)
svm5 = RandomizedSearchCV(SVC(probability=True, random_state=42),
    {'C': [0.1,0.5,1,5,10,50], 'gamma': ['scale','auto',0.01,0.001], 'kernel': ['rbf','poly']},
    n_iter=40, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
svm5.fit(X_tr5, y_tr5)
lr5 = RandomizedSearchCV(LogisticRegression(max_iter=5000, random_state=42),
    {'C': [0.001,0.01,0.05,0.1,0.5,1,5,10], 'penalty': ['l1','l2'], 'solver': ['liblinear']},
    n_iter=16, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
lr5.fit(X_tr5, y_tr5)

est5 = [('lr', lr5.best_estimator_), ('rf', rf5.best_estimator_),
        ('gb', gb5.best_estimator_), ('svm', svm5.best_estimator_),
        ('et', ExtraTreesClassifier(n_estimators=500, max_depth=12, random_state=42))]
m_v5 = VotingClassifier(estimators=est5, voting='soft', n_jobs=-1)
from sklearn.model_selection import RepeatedStratifiedKFold
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_v5 = cross_val_score(m_v5, X_v5, y, cv=rcv, scoring='accuracy').mean()
m_v5.fit(X_v5, y)

# Tatiana v5
tat_v5_df = tat_lf.copy()
for col in cyanite_feat_cols:
    tat_v5_df[col] = tat_cf[col].values
tat_v5_df['genre_encoded'] = tat_genre
tat_v5_df['is_duet'] = 0
tat_v5_df['decade_encoded'] = tat_decade
X_tat_v5 = sel5.transform(sc_v5.transform(tat_v5_df[v5_feat_cols].values.astype(float)))
all_probs['v5'] = m_v5.predict_proba(X_tat_v5)[:, 1]
all_cv['v5'] = cv_v5
print(f"  CV: {cv_v5*100:.1f}%")

# ── V6: ALL Combined (lyrics + MiniLM + Cyanite) ──
print("--- Training v6: ALL Combined ---")
v6_feat_cols = lyrics_feat_cols + cyanite_feat_cols + emb_cols + ['genre_encoded', 'is_duet', 'decade_encoded']

# Build full training matrix
emb_df_train = pd.DataFrame(train_embs, index=df.index, columns=emb_cols)
for col in emb_cols:
    df[col] = emb_df_train[col]
X_v6_raw = df[v6_feat_cols].values
sc_v6 = StandardScaler()
X_v6s = sc_v6.fit_transform(X_v6_raw)
sel6 = SelectKBest(partial(mutual_info_classif, random_state=42), k=100)
X_v6 = sel6.fit_transform(X_v6s, y)

X_tr6, X_te6, y_tr6, y_te6 = train_test_split(X_v6, y, test_size=0.2, random_state=42, stratify=y)
rf6 = RandomizedSearchCV(RandomForestClassifier(random_state=42),
    {'n_estimators': [200,500,800], 'max_depth': [6,8,10,12,None],
     'min_samples_split': [2,3,5,7], 'min_samples_leaf': [1,2,3],
     'max_features': ['sqrt','log2',0.3,0.5]},
    n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
rf6.fit(X_tr6, y_tr6)
m_v6 = rf6.best_estimator_
cv_v6 = cross_val_score(m_v6, X_v6, y, cv=rcv, scoring='accuracy').mean()
m_v6.fit(X_v6, y)

# Tatiana v6
tat_v6_df = tat_lf.copy()
for col in cyanite_feat_cols:
    tat_v6_df[col] = tat_cf[col].values
for i, col in enumerate(emb_cols):
    tat_v6_df[col] = tat_embs[:, i]
tat_v6_df['genre_encoded'] = tat_genre
tat_v6_df['is_duet'] = 0
tat_v6_df['decade_encoded'] = tat_decade
X_tat_v6 = sel6.transform(sc_v6.transform(tat_v6_df[v6_feat_cols].values.astype(float)))
all_probs['v6'] = m_v6.predict_proba(X_tat_v6)[:, 1]
all_cv['v6'] = cv_v6
print(f"  CV: {cv_v6*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("TATIANA PREDICTIONS: ALL 6 VERSIONS SIDE BY SIDE")
print("=" * 90)

versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# Header
header = f"{'Song':<28s}"
for v in versions:
    header += f" {v:>7s}"
header += f" {'AVG':>7s}"
print(header)
print("-" * 90)

# Per-song rows
song_avgs = []
for i, song in enumerate(song_names):
    row = f"{song:<28s}"
    probs_row = []
    for v in versions:
        p = all_probs[v][i] * 100
        probs_row.append(p)
        row += f" {p:>6.1f}%"
    avg = np.mean(probs_row)
    song_avgs.append(avg)
    row += f" {avg:>6.1f}%"
    print(row)

print("-" * 90)

# Version averages row
row = f"{'VERSION AVG':<28s}"
for v in versions:
    avg_v = np.mean(all_probs[v]) * 100
    row += f" {avg_v:>6.1f}%"
overall_avg = np.mean(song_avgs)
row += f" {overall_avg:>6.1f}%"
print(row)

# CV accuracy row
row = f"{'CV ACCURACY':<28s}"
for v in versions:
    row += f" {all_cv[v]*100:>6.1f}%"
row += f"       "
print(row)

print("=" * 90)

# Summary
print(f"\nOVERALL AVERAGE across all songs and all versions: {overall_avg:.1f}%")

# Best song per version
print(f"\nBEST SONG PER VERSION:")
for v in versions:
    best_idx = np.argmax(all_probs[v])
    print(f"  {v}: {song_names[best_idx]:<28s} ({all_probs[v][best_idx]*100:.1f}%)")

# Average ranking
print(f"\nFINAL RANKING (averaged across all 6 versions):")
ranked = sorted(zip(song_names, song_avgs), key=lambda x: x[1], reverse=True)
for rank, (song, avg) in enumerate(ranked, 1):
    label = "HIT" if avg >= 50 else "NOT A HIT"
    print(f"  {rank:>2d}. {song:<28s} {avg:>5.1f}%  {label}")

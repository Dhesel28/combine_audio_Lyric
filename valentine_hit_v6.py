"""
Valentine's Day Hit Song Predictor v6 — Lyrics + HuggingFace MiniLM + Cyanite Audio
Combines all three signal types for maximum accuracy:
  - 39 hand-crafted lyrics features (VADER/TextBlob/emotion scores)
  - 384 MiniLM embedding dimensions (semantic meaning)
  - 42 Cyanite audio features (valence, arousal, tempo, mood)
  - 3 meta features (genre, duet, decade)
  → ~468 total features → SelectKBest reduces to ~80-100
"""

from functools import partial
import pandas as pd
import numpy as np
import re
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RepeatedStratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

sia = SentimentIntensityAnalyzer()


# ── HUGGINGFACE EMBEDDING ──────────────────────────────────────────────────

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

print("Loading all-MiniLM-L6-v2 from local cache...")
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
bert_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
bert_model.eval()
print("Model loaded.\n")


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (torch.sum(token_embeddings * mask_expanded, 1) /
            torch.clamp(mask_expanded.sum(1), min=1e-9))


def get_embedding(text, max_length=256):
    """Get a 384-dim embedding using all-MiniLM-L6-v2 with chunking for long texts."""
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
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
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
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            embeddings.append(emb)
            if i + chunk_size >= len(tokens):
                break
        return np.mean(embeddings, axis=0)


# ── LYRICS FEATURE EXTRACTION ───────────────────────────────────────────────

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

    # Direct numeric features
    feats['cy_valence'] = row.get('cyanite_valence', 0) or 0
    feats['cy_arousal'] = row.get('cyanite_arousal', 0) or 0
    feats['cy_tempo'] = row.get('cyanite_tempo_bpm', 0) or 0

    # m2e features
    feats['m2e_valence'] = row.get('m2e_valence', 0) if pd.notna(row.get('m2e_valence')) else 0
    feats['m2e_arousal'] = row.get('m2e_arousal', 0) if pd.notna(row.get('m2e_arousal')) else 0
    feats['m2e_valentine_fit'] = row.get('m2e_valentine_fit_score', 0) if pd.notna(row.get('m2e_valentine_fit_score')) else 0
    feats['m2e_best_valence'] = row.get('m2e_best_valence', 0) if pd.notna(row.get('m2e_best_valence')) else 0
    feats['m2e_best_arousal'] = row.get('m2e_best_arousal', 0) if pd.notna(row.get('m2e_best_arousal')) else 0

    # Energy level encoding
    energy_map = {'low': 0, 'medium': 1, 'high': 2, 'variable': 1.5}
    feats['cy_energy'] = energy_map.get(str(row.get('cyanite_energy_level', '')).lower(), 1)

    # Energy dynamics
    dyn_map = {'low': 0, 'medium': 1, 'high': 2, 'variable': 1.5}
    feats['cy_energy_dynamics'] = dyn_map.get(str(row.get('cyanite_energy_dynamics', '')).lower(), 1)

    # Voice gender encoding
    vg = str(row.get('cyanite_voice_gender', '')).lower()
    feats['cy_voice_male'] = 1 if 'male' in vg and 'female' not in vg else 0
    feats['cy_voice_female'] = 1 if 'female' in vg else 0

    # Voice presence
    vp_map = {'low': 0, 'medium': 1, 'high': 2}
    feats['cy_voice_presence'] = vp_map.get(str(row.get('cyanite_voice_presence', '')).lower(), 1)

    # Mood tag features
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

    # Count total romantic moods
    rom_moods = ['romantic','loving','tender','passionate','sentimental','warm','sexy','seductive','intimate','dreamy']
    feats['cy_romantic_mood_count'] = sum(1 for m in rom_moods if m in all_moods)

    # Emotional profile
    ep = str(row.get('cyanite_emotional_profile', '')).lower()
    feats['cy_emotion_positive'] = 1 if 'positive' in ep else 0
    feats['cy_emotion_negative'] = 1 if 'negative' in ep else 0
    feats['cy_emotion_balanced'] = 1 if 'balanced' in ep else 0

    # Movement tags
    mv = str(row.get('cyanite_movement_tags', '')).lower()
    feats['cy_flowing'] = 1 if 'flowing' in mv else 0
    feats['cy_swaying'] = 1 if 'swaying' in mv or 'groovy' in mv else 0
    feats['cy_bouncy'] = 1 if 'bouncy' in mv or 'driving' in mv else 0

    # Character tags
    ch = str(row.get('cyanite_character_tags', '')).lower()
    feats['cy_warm'] = 1 if 'warm' in ch else 0
    feats['cy_intimate'] = 1 if 'intimate' in ch else 0
    feats['cy_retro'] = 1 if 'retro' in ch or 'vintage' in ch else 0

    # Tempo categories
    tempo = feats['cy_tempo']
    feats['cy_tempo_slow'] = 1 if tempo < 85 else 0
    feats['cy_tempo_mid'] = 1 if 85 <= tempo <= 120 else 0
    feats['cy_tempo_fast'] = 1 if tempo > 120 else 0

    # Interaction features (audio x mood)
    feats['cy_valence_x_romantic'] = feats['cy_valence'] * feats['cy_mood_romantic']
    feats['cy_arousal_x_energy'] = feats['cy_arousal'] * feats['cy_energy']

    return feats


# ── LOAD AND MERGE DATA ─────────────────────────────────────────────────────

print("=" * 70)
print("VALENTINE'S DAY HIT SONG PREDICTOR v6")
print("Lyrics + HuggingFace MiniLM Embeddings + Cyanite Audio")
print("=" * 70)

# Load datasets
lyrics_df = pd.read_csv("valentine_songs_full_dataset.csv")
lyrics_df = lyrics_df[lyrics_df['has_lyrics'] == True].copy()
lyrics_df = lyrics_df.dropna(subset=['genius_pageviews'])

cyanite_df = pd.read_csv("valentine_songs_cyanite.csv")

# Merge on song title + artist
lyrics_df['key'] = lyrics_df['all song_title'].str.lower().str.strip() + '|' + lyrics_df['artist'].str.lower().str.strip()
cyanite_df['key'] = cyanite_df['song_title'].str.lower().str.strip() + '|' + cyanite_df['artist'].str.lower().str.strip()

df = lyrics_df.merge(cyanite_df, on='key', how='inner', suffixes=('', '_cy'))

print(f"\nMerged dataset: {len(df)} songs")
print(f"Pageview range: {df['genius_pageviews'].min():,.0f} - {df['genius_pageviews'].max():,.0f}")

# Target
median_views = df['genius_pageviews'].median()
df['is_hit'] = (df['genius_pageviews'] >= median_views).astype(int)
print(f"Hit threshold (median): {median_views:,.0f}")
print(f"Hit: {df['is_hit'].sum()} | Not hit: {(df['is_hit']==0).sum()}")

# ── EXTRACT ALL FEATURES ────────────────────────────────────────────────────

print("\nExtracting lyrics features...")
lyrics_feats = [extract_lyrics_features(row['lyrics']) for _, row in df.iterrows()]
lf_df = pd.DataFrame(lyrics_feats, index=df.index)
lyrics_feat_cols = list(lf_df.columns)

print("Extracting Cyanite audio features...")
cy_feats = [extract_cyanite_features(row) for _, row in df.iterrows()]
cf_df = pd.DataFrame(cy_feats, index=df.index)
cyanite_feat_cols = list(cf_df.columns)

print("Extracting MiniLM embeddings (this may take a minute)...")
embeddings = []
for idx, row in df.iterrows():
    emb = get_embedding(row['lyrics'])
    embeddings.append(emb)
emb_array = np.array(embeddings)
emb_cols = [f'emb_{i}' for i in range(EMBED_DIM)]
emb_df = pd.DataFrame(emb_array, index=df.index, columns=emb_cols)
print(f"  Embeddings shape: {emb_array.shape}")

# Combine into df
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
print(f"\nTotal features: {len(all_feature_cols)} "
      f"({len(lyrics_feat_cols)} lyrics + {len(cyanite_feat_cols)} audio + {len(emb_cols)} embeddings + 3 meta)")

X = df[all_feature_cols].values
y = df['is_hit'].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection — higher k to accommodate the richer feature set
k_best = min(100, len(all_feature_cols))
selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=k_best)
X_selected = selector.fit_transform(X_scaled, y)
print(f"Feature selection: {len(all_feature_cols)} -> {k_best} best features")

# Show selected features
mask = selector.get_support()
selected_names = [all_feature_cols[i] for i in range(len(all_feature_cols)) if mask[i]]
scores = selector.scores_
feat_scores = sorted(zip(all_feature_cols, scores), key=lambda x: x[1], reverse=True)

# Count selected by source
n_lyrics_sel = sum(1 for n in selected_names if n in lyrics_feat_cols)
n_audio_sel = sum(1 for n in selected_names if n in cyanite_feat_cols)
n_emb_sel = sum(1 for n in selected_names if n.startswith('emb_'))
n_meta_sel = sum(1 for n in selected_names if n in ['genre_encoded', 'is_duet', 'decade_encoded'])
print(f"Selected breakdown: {n_lyrics_sel} lyrics, {n_audio_sel} audio, {n_emb_sel} embedding, {n_meta_sel} meta")

print("\nTop 20 most informative features:")
for fname, fscore in feat_scores[:20]:
    if fname.startswith('emb_'):
        source = "EMBED"
    elif fname.startswith('cy_') or fname.startswith('m2e_'):
        source = "AUDIO"
    elif fname in ['genre_encoded', 'is_duet', 'decade_encoded']:
        source = "META"
    else:
        source = "LYRICS"
    print(f"  [{source:6s}] {fname:30s} MI={fscore:.4f}")


# ── TRAIN & EVALUATE ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("MODEL TRAINING & EVALUATION")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Tune models
print("\nTuning models...")
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100,200,300,500], 'max_depth': [3,4,5,6],
     'learning_rate': [0.01,0.03,0.05,0.1], 'subsample': [0.7,0.8,0.9],
     'min_samples_split': [3,5,7,10], 'min_samples_leaf': [1,2,3]},
    n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
gb_search.fit(X_train, y_train)
print(f"  Gradient Boosting best CV: {gb_search.best_score_:.3f}")

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [200,500,800], 'max_depth': [6,8,10,12,None],
     'min_samples_split': [2,3,5,7], 'min_samples_leaf': [1,2,3],
     'max_features': ['sqrt','log2',0.3,0.5]},
    n_iter=80, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
print(f"  Random Forest best CV: {rf_search.best_score_:.3f}")

svm_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    {'C': [0.1,0.5,1,5,10,50], 'gamma': ['scale','auto',0.01,0.001],
     'kernel': ['rbf','poly']},
    n_iter=40, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
svm_search.fit(X_train, y_train)
print(f"  SVM best CV: {svm_search.best_score_:.3f}")

lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=5000, random_state=42),
    {'C': [0.001,0.01,0.05,0.1,0.5,1,5,10], 'penalty': ['l1','l2'],
     'solver': ['liblinear']},
    n_iter=16, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
lr_search.fit(X_train, y_train)
print(f"  Logistic Regression best CV: {lr_search.best_score_:.3f}")

# Evaluate with repeated CV
print(f"\n{'─'*70}")
print("Repeated 5-Fold CV (x3 = 15 evaluations)")
print(f"{'─'*70}")

models = {
    'Logistic Regression': lr_search.best_estimator_,
    'Random Forest': rf_search.best_estimator_,
    'Gradient Boosting': gb_search.best_estimator_,
    'SVM': svm_search.best_estimator_,
    'Extra Trees': ExtraTreesClassifier(n_estimators=500, max_depth=12, random_state=42),
}

best_model, best_score, best_name = None, 0, ""
results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_selected, y, cv=rcv, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ta = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'cv': cv_scores.mean(), 'std': cv_scores.std(), 'test': ta, 'f1': f1, 'model': model}
    print(f"\n  {name}")
    print(f"    CV Accuracy:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"    Test Accuracy: {ta:.3f} | F1: {f1:.3f}")
    if cv_scores.mean() > best_score:
        best_score, best_model, best_name = cv_scores.mean(), model, name

# Ensembles
estimators = [
    ('lr', lr_search.best_estimator_),
    ('rf', rf_search.best_estimator_),
    ('gb', gb_search.best_estimator_),
    ('svm', svm_search.best_estimator_),
    ('et', ExtraTreesClassifier(n_estimators=500, max_depth=12, random_state=42)),
]

for ens_name, ens_model in [
    ("Stacking Ensemble", StackingClassifier(estimators=estimators,
        final_estimator=LogisticRegression(max_iter=5000, C=1.0), cv=5, n_jobs=-1)),
    ("Soft Voting", VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)),
]:
    cv_e = cross_val_score(ens_model, X_selected, y, cv=rcv, scoring='accuracy')
    ens_model.fit(X_train, y_train)
    y_pe = ens_model.predict(X_test)
    ta_e = accuracy_score(y_test, y_pe)
    f1_e = f1_score(y_test, y_pe, average='weighted')
    print(f"\n  {ens_name}")
    print(f"    CV Accuracy:   {cv_e.mean():.3f} (+/- {cv_e.std():.3f})")
    print(f"    Test Accuracy: {ta_e:.3f} | F1: {f1_e:.3f}")
    if cv_e.mean() > best_score:
        best_score, best_model, best_name = cv_e.mean(), ens_model, ens_name

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"  CV Accuracy: {best_score:.3f}")
print(f"{'='*70}")

# Retrain on all data
best_model.fit(X_selected, y)

# Classification report
print(f"\nClassification Report ({best_name} on test set):")
if best_name in results:
    results[best_name]['model'].fit(X_train, y_train)
    y_final = results[best_name]['model'].predict(X_test)
else:
    from copy import deepcopy
    eval_model = deepcopy(best_model)
    eval_model.fit(X_train, y_train)
    y_final = eval_model.predict(X_test)
print(classification_report(y_test, y_final, target_names=['Not a Hit', 'Hit']))


# ── PREDICT ON TATIANA DEMARIA ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("PREDICTIONS: TATIANA DEMARIA'S SONGS")
print("=" * 70)

# Load existing lyrics CSV
tatiana = pd.read_csv("Testing lyrics /full_tatiana_demaria_lyrics.csv")

# Add new songs from RTF files
import subprocess
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
    words = lyrics_text.split()
    new_rows.append({
        'artist': 'Tatiana DeMaria',
        'song': song_name,
        'lyrics': lyrics_text,
        'word_count': len(words),
        'char_count': len(lyrics_text),
    })

# Add Save Me and Beirut Fire from CSV
bf_df = pd.read_csv("Testing lyrics /Beirut_Fire and Save me lyrics  - Sheet1.csv")
for _, row in bf_df.iterrows():
    if pd.notna(row['Song']) and pd.notna(row['Lyric']):
        lyrics_text = row['Lyric'].replace('\\n', '\n').strip()
        words = lyrics_text.split()
        new_rows.append({
            'artist': 'Tatiana DeMaria',
            'song': row['Song'].strip(),
            'lyrics': lyrics_text,
            'word_count': len(words),
            'char_count': len(lyrics_text),
        })

# Add You Make Me from CSV (take first valid lyrics row only)
ym_df = pd.read_csv("Testing lyrics /You  make me  - Sheet1.csv")
ym_added = False
for _, row in ym_df.iterrows():
    lyric_val = row.get('Lyric') or row.get(row.keys()[2]) if len(row.keys()) > 2 else None
    if not ym_added and lyric_val is not None and pd.notna(lyric_val) and len(str(lyric_val)) > 50:
        lyrics_text = str(lyric_val).replace('\\n', '\n').strip()
        words = lyrics_text.split()
        new_rows.append({
            'artist': 'Tatiana DeMaria',
            'song': 'You Make Me',
            'lyrics': lyrics_text,
            'word_count': len(words),
            'char_count': len(lyrics_text),
        })
        ym_added = True

tatiana = pd.concat([tatiana, pd.DataFrame(new_rows)], ignore_index=True)
print(f"\nTotal Tatiana songs: {len(tatiana)}")

# Load Tatiana's Cyanite features
tat_cyanite = pd.read_csv("Tatiana_cyanite.csv")

# Build name mapping between lyrics and Cyanite
def normalize_name(s):
    s = str(s).lower().strip().strip("'\"")
    for suffix in ['official audio', 'acoustic', 'bp mix 2', 'master',
                   'rough boucnce tdm britain 25th nov 2024', '090920']:
        s = s.replace(suffix, '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

cy_name_map = {}
for idx, row in tat_cyanite.iterrows():
    norm = normalize_name(row['song_title'])
    cy_name_map[norm] = idx

# Lyrics features for Tatiana
tat_feats = []
for _, row in tatiana.iterrows():
    f = extract_lyrics_features(row['lyrics'])
    f['song'] = row['song']
    tat_feats.append(f)
tat_df = pd.DataFrame(tat_feats)

# MiniLM embeddings for Tatiana
print("Computing MiniLM embeddings for Tatiana's songs...")
tat_embs = []
for _, row in tatiana.iterrows():
    tat_embs.append(get_embedding(row['lyrics']))
tat_emb_array = np.array(tat_embs)

# Match each Tatiana song to its Cyanite data
cy_medians = cf_df.median()  # fallback for unmatched songs
matched = 0
unmatched_songs = []

for idx, row in tat_df.iterrows():
    song_norm = normalize_name(row['song'])
    cy_idx = None

    # Direct match
    if song_norm in cy_name_map:
        cy_idx = cy_name_map[song_norm]
    else:
        # Fuzzy: check if any Cyanite name contains or is contained by lyrics name
        for cy_norm, cidx in cy_name_map.items():
            if song_norm in cy_norm or cy_norm in song_norm:
                cy_idx = cidx
                break
            if song_norm == 'anxiety' and 'anxiety' in cy_norm:
                cy_idx = cidx
                break
            if song_norm == 'without you' and 'without you' in cy_norm:
                cy_idx = cidx
                break
            if song_norm == 'make me feel' and cy_norm == 'make me feel':
                cy_idx = cidx
                break
            if '04 you make me' in cy_norm and song_norm == 'you make me':
                cy_idx = cidx
                break
            if 'fresh meat' in cy_norm and 'fresh meat' in song_norm:
                cy_idx = cidx
                break
            if 'gallery' in cy_norm and 'gallery' in song_norm:
                cy_idx = cidx
                break
            if 'open wide' in cy_norm and 'open wide' in song_norm:
                cy_idx = cidx
                break

    if cy_idx is not None:
        cy_row = tat_cyanite.iloc[cy_idx]
        cy_f = extract_cyanite_features(cy_row)
        for col in cyanite_feat_cols:
            tat_df.at[idx, col] = cy_f.get(col, cy_medians[col])
        matched += 1
    else:
        unmatched_songs.append(row['song'])
        for col in cyanite_feat_cols:
            tat_df.at[idx, col] = cy_medians[col]

print(f"Cyanite matched: {matched}/{len(tat_df)} songs")
if unmatched_songs:
    print(f"Unmatched (using median values): {unmatched_songs}")

# Add embedding columns for Tatiana
for i, col in enumerate(emb_cols):
    tat_df[col] = tat_emb_array[:, i]

tat_df['genre_encoded'] = le_genre.transform(['Pop'])[0]
tat_df['is_duet'] = 0
tat_df['decade_encoded'] = le_decade.transform([2020])[0]

X_tat = tat_df[all_feature_cols].values.astype(float)
X_tat_s = scaler.transform(X_tat)
X_tat_sel = selector.transform(X_tat_s)

preds = best_model.predict(X_tat_sel)
probs = best_model.predict_proba(X_tat_sel)

print(f"\nUsing model: {best_name}")
print(f"Hit threshold: >= {median_views:,.0f} Genius pageviews")
print(f"Using actual Cyanite audio features + MiniLM embeddings for Tatiana's songs\n")

print(f"{'Song':<35s} {'Prediction':<15s} {'Hit Prob':>10s}")
print("-" * 63)
for i, row in tat_df.iterrows():
    pred = "HIT" if preds[i] == 1 else "NOT A HIT"
    print(f"{row['song']:<35s} {pred:<15s} {probs[i][1]:>9.1%}")

hit_count = preds.sum()
print(f"\n{'='*70}")
print(f"VERDICT: {hit_count}/{len(preds)} songs predicted as Valentine hits")
print(f"Average hit probability: {probs[:, 1].mean():.1%}")
print(f"Most likely hit: {tat_df.iloc[probs[:, 1].argmax()]['song']} ({probs[:, 1].max():.1%})")
print(f"{'='*70}")

# ── VERSION COMPARISON ───────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("ACCURACY PROGRESSION")
print(f"{'='*70}")
print(f"  v1 (Basic features):          49.1% CV")
print(f"  v2 (+TF-IDF, emotions):       58.1% CV")
print(f"  v3 (+VADER, tuning):          49.1% CV")
print(f"  v4 (+HuggingFace MiniLM):     64.0% CV")
print(f"  v5 (+Cyanite audio):          67.4% CV")
print(f"  v6 (ALL: lyrics+MiniLM+audio):{best_score*100:.1f}% CV  <-- NEW")
print(f"{'='*70}")

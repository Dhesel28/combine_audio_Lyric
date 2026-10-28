"""
Add Lyrics to TikTok Virality Dataset
======================================
Phase 1 — Fetch Genius lyrics for every labeled song (concurrent, resumable)
Phase 2 — Extract 45 NLP features per song
Phase 3 — Retrain model with audio + lyrics combined

Usage:
    python add_lyrics.py                 # full pipeline
    python add_lyrics.py --fetch-only    # just fetch & cache lyrics
    python add_lyrics.py --retrain-only  # skip fetch, retrain from cache
    python add_lyrics.py --no-embed      # skip MiniLM embeddings (faster)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import lyricsgenius
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import joblib

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "raw"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

EXPANDED   = DATA_DIR / "tiktok_expanded.csv"
CACHE_CSV  = DATA_DIR / "lyrics_cache.csv"          # song → lyrics
WITH_LYR   = DATA_DIR / "tiktok_with_lyrics.csv"    # final combined dataset

# ── Genius credentials ────────────────────────────────────────────────────────
GENIUS_TOKEN = "TnjMuLq_hkr02I5tjjU6JYkBt95FxRNtFtPpjpoQA7gQtYlkjXZDQa6PBUriaV-v"
GENIUS_HEADERS = {"Authorization": f"Bearer {GENIUS_TOKEN}"}

# lyricsgenius client (thread-local to avoid race conditions)
import threading
_genius_local = threading.local()

def _get_genius_client():
    if not hasattr(_genius_local, "client"):
        _genius_local.client = lyricsgenius.Genius(
            GENIUS_TOKEN,
            timeout=10,
            sleep_time=0.3,
            verbose=False,
            remove_section_headers=True,
            skip_non_songs=True,
        )
    return _genius_local.client

# ── Audio feature columns (from train_model.py) ───────────────────────────────
AUDIO_FEATURES = [
    "danceability","energy","key","loudness","mode",
    "speechiness","acousticness","instrumentalness","liveness","valence","tempo",
]
DERIVED_FEATURES = ["dance_energy","mood_score","tempo_norm","acoustic_anti","vocal_content"]

sia = SentimentIntensityAnalyzer()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — FETCH LYRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_lyrics(raw: str) -> Optional[str]:
    """Strip contributor headers, translations, and Genius boilerplate."""
    # Remove leading contributor/translation lines up to actual lyrics
    raw = re.sub(r"^\d+\s*Contributors?.*?\n", "", raw, flags=re.S)
    raw = re.split(r"You might also like|Embed\s*$", raw, flags=re.I)[0]
    raw = raw.strip()
    return raw if len(raw) > 100 else None


def fetch_one(title: str, artist: str) -> tuple[Optional[str], int, Optional[str]]:
    """Returns (lyrics_text, genius_pageviews, genius_url) using lyricsgenius."""
    try:
        genius = _get_genius_client()
        song = genius.search_song(title, artist)
        if song is None:
            return None, 0, None
        lyrics = _clean_lyrics(song.lyrics) if song.lyrics else None
        url = song.url or None
        views = 0
        return lyrics, views, url
    except Exception:
        return None, 0, None


def load_cache() -> dict[str, dict]:
    """Load existing lyrics cache. Returns {_key: {lyrics, pageviews, url}}."""
    if not CACHE_CSV.exists():
        return {}
    df = pd.read_csv(CACHE_CSV)
    return {
        row["_key"]: {
            "lyrics": row.get("lyrics") if pd.notna(row.get("lyrics")) else None,
            "pageviews": row.get("pageviews", 0),
            "url": row.get("url"),
        }
        for _, row in df.iterrows()
        if pd.notna(row.get("_key"))
    }


def save_cache(cache: dict[str, dict]) -> None:
    rows = [{"_key": k, **v} for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(CACHE_CSV, index=False)


def fetch_lyrics(df: pd.DataFrame, workers: int = 6) -> dict[str, dict]:
    """Fetch lyrics for all songs in df. Resumes from existing cache."""
    cache = load_cache()
    todo  = df[~df["_key"].isin(cache)][["_key", "track_name", "artist_name"]].drop_duplicates("_key")

    if todo.empty:
        print(f"  All {len(df['_key'].unique())} songs already cached.")
        return cache

    print(f"  {len(cache)} already cached  |  {len(todo)} to fetch  |  {workers} workers")
    done = 0

    def worker(row):
        lyrics, views, url = fetch_one(row["track_name"], row["artist_name"])
        time.sleep(0.3)          # polite rate limit per worker
        return row["_key"], lyrics, views, url

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, row): row["_key"]
                   for _, row in todo.iterrows()}
        for fut in as_completed(futures):
            key, lyrics, views, url = fut.result()
            cache[key] = {"lyrics": lyrics, "pageviews": int(views), "url": url}
            done += 1
            found = sum(1 for v in cache.values() if v.get("lyrics"))
            if done % 50 == 0 or done == len(todo):
                pct = done / len(todo) * 100
                print(f"  [{done:4d}/{len(todo)}  {pct:5.1f}%]  "
                      f"lyrics found: {found}/{done+len(cache)-len(todo)}")
                save_cache(cache)   # checkpoint

    save_cache(cache)
    found = sum(1 for v in cache.values() if v.get("lyrics"))
    print(f"\n  Lyrics cache: {found}/{len(cache)} songs have lyrics")
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _rhyme_score(raw: str) -> float:
    lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    ends = []
    for line in lines:
        ws = re.findall(r"[a-z]+", line)
        ends.append(ws[-1][-3:] if ws and len(ws[-1]) >= 3 else (ws[-1] if ws else ""))
    rhymes = sum(
        (1 if ends[i] == ends[i+1] else 0.5 if ends[i][-2:] == ends[i+1][-2:] else 0)
        for i in range(len(ends)-1)
        if ends[i] and ends[i+1]
    )
    return rhymes / max(len(ends)-1, 1)


def _repetition_score(raw: str) -> float:
    lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    return sum(c-1 for c in counts.values() if c > 1) / len(lines)


def _syllable_count(word: str) -> int:
    """Rough syllable counter."""
    word = word.lower().rstrip("e")
    return max(1, len(re.findall(r"[aeiou]+", word)))


_HOOK_WORDS = {"oh","ooh","yeah","yea","hey","whoa","woah","uh","ah","na",
               "la","bam","boom","woo","yo","ay","aye","mmm","hmm"}
_DANCE_WORDS = {"move","shake","dance","body","groove","bounce","sway","swing",
                "twerk","vibe","feel","hit","drop","floor","beat"}
_VIRAL_PHRASES = {"i know","you know","like that","right now","all night",
                  "feel it","can't stop","don't stop","let's go","come on"}


def extract_lyrics_features(raw: str | None) -> dict:
    """Extract 45 NLP features from raw lyrics text."""
    zero = {
        "lyr_word_count":0,"lyr_line_count":0,"lyr_unique_ratio":0,
        "lyr_avg_line_len":0,"lyr_avg_word_len":0,"lyr_has_chorus":0,
        "lyr_hook_density":0,"lyr_repetition":0,"lyr_rhyme_score":0,
        "lyr_total_syllables":0,"lyr_syllable_rate":0,
        "lyr_hook_words":0,"lyr_dance_words":0,"lyr_viral_phrases":0,
        "lyr_love":0,"lyr_you":0,"lyr_i":0,"lyr_we":0,
        "lyr_romance":0,"lyr_joy":0,"lyr_energy_words":0,
        "lyr_vader_compound":0,"lyr_vader_pos":0,"lyr_vader_neg":0,"lyr_vader_neu":0,
        "lyr_tb_polarity":0,"lyr_tb_subjectivity":0,
        "lyr_sent_variance":0,"lyr_sent_range":0,"lyr_emotional_arc":0,
        "lyr_positivity_ratio":0,"lyr_you_ratio":0,"lyr_lex_diversity":0,
        "lyr_short_lines":0,"lyr_quotable_score":0,
        "lyr_has_lyrics":0,
    }
    if not raw or not isinstance(raw, str) or len(raw.strip()) < 50:
        return zero

    cleaned = _clean(raw)
    words   = cleaned.split()
    wc      = len(words)
    if wc == 0:
        return zero

    lines_raw = [l.strip() for l in raw.split("\n") if l.strip()]
    lines_clean = [_clean(l) for l in lines_raw]
    lc = len(lines_raw)

    def cnt(w):
        return len(re.findall(r"\b" + re.escape(w) + r"\b", cleaned))
    def cntl(ws):
        return sum(cnt(w) for w in ws)

    uw = len(set(words))
    avg_wl = np.mean([len(w) for w in words])
    avg_ll = np.mean([len(l.split()) for l in lines_clean]) if lines_clean else 0

    # Section tags
    has_chorus = int(bool(re.search(r"\[chorus\]", raw.lower())))

    # Hook density (most repeated line / total lines)
    line_counts = Counter(lines_clean)
    hook_density = line_counts.most_common(1)[0][1] / lc if lc else 0

    # Repetition
    rep = _repetition_score(raw)

    # Rhyme
    rhyme = _rhyme_score(raw)

    # Syllables
    all_syl = sum(_syllable_count(w) for w in words)
    syl_rate = all_syl / wc  # syllables per word

    # TikTok hook words
    hook_w   = sum(1 for w in words if w in _HOOK_WORDS)
    dance_w  = cntl(_DANCE_WORDS)
    viral_ph = sum(1 for p in _VIRAL_PHRASES if p in cleaned)

    # Love/you/i/we
    love_cnt = cnt("love")
    you_cnt  = cnt("you")
    i_cnt    = cnt("i")
    we_cnt   = cnt("we")

    # Emotion clusters
    romance = cntl(["love","heart","kiss","darling","baby","passion"])
    joy     = cntl(["happy","smile","laugh","beautiful","amazing","wonderful"])
    energy  = cntl(["fire","wild","hot","electric","loud","crazy","hype"])

    # VADER
    vd = sia.polarity_scores(cleaned)

    # TextBlob
    blob = TextBlob(cleaned)

    # Per-line sentiment for arc/variance
    if lines_clean and len(lines_clean) > 1:
        ls  = [sia.polarity_scores(l)["compound"] for l in lines_clean]
        sv  = float(np.var(ls))
        sr  = max(ls) - min(ls)
        mid = len(ls) // 2
        ea  = float(np.mean(ls[mid:]) - np.mean(ls[:mid])) if mid > 0 else 0.0
    else:
        sv = sr = ea = 0.0

    # Positivity ratio
    total_emo = romance + joy + energy + 1
    pos_r = (romance + joy) / total_emo

    # you ratio
    you_r = you_cnt / max(you_cnt + i_cnt + we_cnt, 1)

    # Short (quotable) lines: lines with 2–8 words
    short = sum(1 for l in lines_clean if 2 <= len(l.split()) <= 8)
    quotable = short / lc if lc else 0

    return {
        "lyr_word_count":     wc,
        "lyr_line_count":     lc,
        "lyr_unique_ratio":   uw / wc,
        "lyr_avg_line_len":   avg_ll,
        "lyr_avg_word_len":   avg_wl,
        "lyr_has_chorus":     has_chorus,
        "lyr_hook_density":   hook_density,
        "lyr_repetition":     rep,
        "lyr_rhyme_score":    rhyme,
        "lyr_total_syllables":all_syl,
        "lyr_syllable_rate":  syl_rate,
        "lyr_hook_words":     hook_w,
        "lyr_dance_words":    dance_w,
        "lyr_viral_phrases":  viral_ph,
        "lyr_love":           love_cnt,
        "lyr_you":            you_cnt,
        "lyr_i":              i_cnt,
        "lyr_we":             we_cnt,
        "lyr_romance":        romance,
        "lyr_joy":            joy,
        "lyr_energy_words":   energy,
        "lyr_vader_compound": vd["compound"],
        "lyr_vader_pos":      vd["pos"],
        "lyr_vader_neg":      vd["neg"],
        "lyr_vader_neu":      vd["neu"],
        "lyr_tb_polarity":    float(blob.sentiment.polarity),
        "lyr_tb_subjectivity":float(blob.sentiment.subjectivity),
        "lyr_sent_variance":  sv,
        "lyr_sent_range":     sr,
        "lyr_emotional_arc":  ea,
        "lyr_positivity_ratio": pos_r,
        "lyr_you_ratio":      you_r,
        "lyr_lex_diversity":  uw / wc,
        "lyr_short_lines":    short,
        "lyr_quotable_score": quotable,
        "lyr_has_lyrics":     1,
    }


# ── MiniLM embeddings (optional) ─────────────────────────────────────────────

EMBED_DIM = 384
_tokenizer = None
_bert_model = None


def _load_minilm():
    global _tokenizer, _bert_model
    if _tokenizer is None:
        print("  Loading MiniLM embeddings model...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            _bert_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            _bert_model.eval()
            print("  MiniLM loaded from local cache.")
        except Exception as e:
            print(f"  MiniLM not available locally ({e}) — skipping embeddings.")
            _tokenizer = False   # sentinel: tried and failed


def _mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def get_embedding(text: str) -> np.ndarray:
    _load_minilm()
    if not _tokenizer:
        return np.zeros(EMBED_DIM)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return np.zeros(EMBED_DIM)
    inputs = _tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256, padding=True)
    with torch.no_grad():
        out = _bert_model(**inputs)
    emb = _mean_pool(out, inputs["attention_mask"]).squeeze().numpy()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-9) if norm > 0 else emb


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — RETRAIN WITH LYRICS + AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

LYRICS_FEATURES = [
    "lyr_word_count","lyr_line_count","lyr_unique_ratio","lyr_avg_line_len",
    "lyr_avg_word_len","lyr_has_chorus","lyr_hook_density","lyr_repetition",
    "lyr_rhyme_score","lyr_total_syllables","lyr_syllable_rate",
    "lyr_hook_words","lyr_dance_words","lyr_viral_phrases",
    "lyr_love","lyr_you","lyr_i","lyr_we",
    "lyr_romance","lyr_joy","lyr_energy_words",
    "lyr_vader_compound","lyr_vader_pos","lyr_vader_neg","lyr_vader_neu",
    "lyr_tb_polarity","lyr_tb_subjectivity",
    "lyr_sent_variance","lyr_sent_range","lyr_emotional_arc",
    "lyr_positivity_ratio","lyr_you_ratio","lyr_lex_diversity",
    "lyr_short_lines","lyr_quotable_score","lyr_has_lyrics",
]


def engineer_audio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dance_energy"]  = df["danceability"] * df["energy"]
    df["mood_score"]    = df["valence"] * df["energy"]
    df["tempo_norm"]    = df["tempo"] / 200.0
    df["acoustic_anti"] = 1.0 - df["acousticness"]
    df["vocal_content"] = 1.0 - df["instrumentalness"]
    return df


def train_and_evaluate(X: np.ndarray, y: np.ndarray, tag: str) -> dict:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pos_neg  = (y == 0).sum() / max((y == 1).sum(), 1)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": (
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42),
            True),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1), False),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1), False),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                               scale_pos_weight=pos_neg, eval_metric="logloss",
                               random_state=42, verbosity=0), False)
    if HAS_LGB:
        models["LightGBM"] = (
            lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                class_weight="balanced", random_state=42, verbose=-1), False)

    print(f"\n{'─'*65}")
    print(f"  {tag}")
    print(f"  {'Model':<25} {'AUC':>8} {'±':>6} {'AvgPrec':>10}")
    print(f"{'─'*65}")

    results = {}
    for name, (model, needs_scale) in models.items():
        X_in  = X_scaled if needs_scale else X
        scores = cross_validate(model, X_in, y, cv=cv,
                                scoring=["roc_auc","average_precision"],
                                return_train_score=False, n_jobs=1)
        results[name] = {
            "auc": float(scores["test_roc_auc"].mean()),
            "auc_std": float(scores["test_roc_auc"].std()),
            "avg_precision": float(scores["test_average_precision"].mean()),
            "needs_scale": needs_scale,
        }
        r = results[name]
        print(f"  {name:<25} {r['auc']:>8.3f} {r['auc_std']:>6.3f} {r['avg_precision']:>10.3f}")

    print(f"{'─'*65}")

    # Save best model
    best_name  = max(results, key=lambda k: results[k]["auc"])
    best_model = models[best_name][0]
    X_fit = X_scaled if results[best_name]["needs_scale"] else X
    best_model.fit(X_fit, y)

    suffix = "_lyrics" if "lyrics" in tag.lower() else "_audio"
    out = MODEL_DIR / f"song_virality_model{suffix}.pkl"
    joblib.dump({
        "model": best_model, "scaler": scaler,
        "model_name": best_name, "results": results,
        "training_shape": X.shape,
    }, out)
    print(f"  Saved → {out}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-only",   action="store_true")
    parser.add_argument("--retrain-only", action="store_true")
    parser.add_argument("--no-embed",     action="store_true")
    parser.add_argument("--workers",      type=int, default=6)
    args = parser.parse_args()

    # ── Load base dataset ──────────────────────────────────────────────────────
    print("=" * 65)
    print("TikTok Viral Predictor  —  Adding Lyrics Features")
    print("=" * 65)

    df = pd.read_csv(EXPANDED)
    df = df[df["viral"] != -1].dropna(subset=AUDIO_FEATURES).copy()
    print(f"\nLabeled songs with audio features: {len(df):,}")

    # ── Phase 1: Fetch lyrics ──────────────────────────────────────────────────
    if not args.retrain_only:
        print(f"\n{'─'*65}")
        print("PHASE 1 — Fetching Genius lyrics")
        print(f"{'─'*65}")
        cache = fetch_lyrics(df, workers=args.workers)
    else:
        cache = load_cache()
        print(f"\nUsing cached lyrics: {len(cache)} entries")

    if args.fetch_only:
        print("--fetch-only: done.")
        return

    # ── Phase 2: Extract lyrics features ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print("PHASE 2 — Extracting lyrics features")
    print(f"{'─'*65}")

    lyr_feats = []
    for _, row in df.iterrows():
        entry = cache.get(row["_key"], {})
        lyrics = entry.get("lyrics") if entry else None
        feats  = extract_lyrics_features(lyrics)
        feats["genius_pageviews"] = entry.get("pageviews", 0) if entry else 0
        lyr_feats.append(feats)

    lyr_df = pd.DataFrame(lyr_feats, index=df.index)
    songs_with_lyrics = lyr_df["lyr_has_lyrics"].sum()
    print(f"  Songs with lyrics: {int(songs_with_lyrics)}/{len(df)}")

    # MiniLM embeddings (optional)
    emb_cols = []
    if not args.no_embed:
        print("\n  Computing MiniLM embeddings...")
        _load_minilm()
        if _tokenizer:
            embs = []
            for i, (_, row) in enumerate(df.iterrows()):
                entry  = cache.get(row["_key"], {})
                lyrics = entry.get("lyrics") if entry else None
                embs.append(get_embedding(lyrics or ""))
                if (i+1) % 200 == 0:
                    print(f"    {i+1}/{len(df)} embeddings done")
            emb_arr  = np.array(embs)
            emb_cols = [f"emb_{i}" for i in range(EMBED_DIM)]
            emb_df   = pd.DataFrame(emb_arr, index=df.index, columns=emb_cols)
            for c in emb_cols:
                lyr_df[c] = emb_df[c]
            print(f"  Embeddings shape: {emb_arr.shape}")

    # Merge lyrics into main df
    df = pd.concat([df, lyr_df], axis=1)
    df.to_csv(WITH_LYR, index=False)
    print(f"\n  Saved → {WITH_LYR}  ({len(df):,} rows)")

    # ── Phase 3: Train audio-only vs audio+lyrics ─────────────────────────────
    print(f"\n{'─'*65}")
    print("PHASE 3 — Model Training Comparison")
    print(f"{'─'*65}")
    print(f"  Total training songs : {len(df):,}")
    print(f"  Viral / Not viral    : {(df['viral']==1).sum():,} / {(df['viral']==0).sum():,}")

    df = engineer_audio(df)

    # A) Audio only (baseline)
    audio_cols = AUDIO_FEATURES + ["dance_energy","mood_score","tempo_norm",
                                    "acoustic_anti","vocal_content"]
    X_audio = df[audio_cols].fillna(0).values
    y       = df["viral"].values

    results_audio = train_and_evaluate(X_audio, y,
        tag="AUDIO ONLY (baseline)  —  16 features")

    # B) Audio + lyrics (SelectKBest)
    all_lyr_cols = LYRICS_FEATURES + emb_cols
    available    = [c for c in all_lyr_cols if c in df.columns]
    combined_cols = audio_cols + available

    X_comb = df[combined_cols].fillna(0).values

    # SelectKBest to manage dimensionality
    k = min(60, X_comb.shape[1])
    scaler_tmp = StandardScaler()
    X_scaled_tmp = scaler_tmp.fit_transform(X_comb)
    from functools import partial
    selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=k)
    X_sel = selector.fit_transform(X_scaled_tmp, y)

    print(f"\n  Features: {X_comb.shape[1]} total → top {k} selected")
    sel_mask  = selector.get_support()
    sel_names = [combined_cols[i] for i in range(len(combined_cols)) if sel_mask[i]]
    n_audio_sel = sum(1 for n in sel_names if not n.startswith("lyr_") and not n.startswith("emb_"))
    n_lyr_sel   = sum(1 for n in sel_names if n.startswith("lyr_"))
    n_emb_sel   = sum(1 for n in sel_names if n.startswith("emb_"))
    print(f"  Selected: {n_audio_sel} audio  +  {n_lyr_sel} lyrics  +  {n_emb_sel} embeddings")

    results_comb = train_and_evaluate(X_sel, y,
        tag=f"AUDIO + LYRICS  —  {k} selected features")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")

    best_audio = max(results_audio, key=lambda k: results_audio[k]["auc"])
    best_comb  = max(results_comb,  key=lambda k: results_comb[k]["auc"])
    auc_a = results_audio[best_audio]["auc"]
    auc_c = results_comb[best_comb]["auc"]
    delta = auc_c - auc_a

    print(f"  Audio only   → {best_audio:<22}  AUC: {auc_a:.3f}")
    print(f"  Audio+Lyrics → {best_comb:<22}  AUC: {auc_c:.3f}  "
          f"({'+'if delta>=0 else ''}{delta:.3f})")
    print(f"\n  Songs with lyrics : {int(songs_with_lyrics):,}/{len(df):,} "
          f"({int(songs_with_lyrics)/len(df)*100:.1f}%)")
    print(f"  Lyrics coverage lifts AUC from {auc_a:.3f} → {auc_c:.3f}")
    print(f"\n  Model saved → models/song_virality_model_lyrics.pkl")
    print(f"  Dataset saved → data/raw/tiktok_with_lyrics.csv")
    print("=" * 65)

    # Save comparison JSON for presentation
    import json
    comparison = {
        "audio_only":   {"best": best_audio, "auc": auc_a, "results": results_audio},
        "audio_lyrics": {"best": best_comb,  "auc": auc_c, "results": results_comb},
        "songs_with_lyrics": int(songs_with_lyrics),
        "total_songs": len(df),
        "lyrics_features_selected": n_lyr_sel,
        "embedding_features_selected": n_emb_sel,
    }
    (MODEL_DIR / "lyrics_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"  Comparison saved → models/lyrics_comparison.json")


if __name__ == "__main__":
    main()

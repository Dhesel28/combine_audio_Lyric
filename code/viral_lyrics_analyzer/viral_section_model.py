"""
Train and evaluate ML models to predict which lyric section goes viral.

Models:
  - LogisticRegression (+ StandardScaler)       — baseline
  - RandomForestClassifier(n_estimators=500)     — ensemble
  - XGBoostClassifier                            — gradient boosting
  - LightGBMClassifier                           — fast gradient boosting
  - MLPClassifier(128→64→32)                    — neural net
  - StackingClassifier (LR meta-learner)         — ensemble of best models

Validation: GroupKFold(n_splits=N_SONGS) — leave-one-song-out

Metrics: P@1 (precision at rank 1), MRR, AUC-ROC

Outputs:
  models/viral_section_model.pkl       — best inference model
  models/viral_section_model_rf.pkl    — RF for feature importances
  models/feature_names.json
  models/model_metadata.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from nlp_viral_detector import WEIGHTS, _parse_sections, _section_type, score_section as _nlp_score

# 12 NLP signals used as ML features (no Genius API needed at inference time)
NLP_FEATURE_NAMES: list = list(WEIGHTS.keys())
FEATURE_NAMES = NLP_FEATURE_NAMES  # backwards-compat alias

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Load training data ─────────────────────────────────────────────────────────

def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, groups) for GroupKFold."""
    path = DATA_DIR / "training_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run build_training_data.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} sections, {df['song'].nunique()} songs")

    X = df[NLP_FEATURE_NAMES].fillna(0).values.astype(np.float32)
    y = df["is_viral"].values.astype(int)
    songs = df["song"].values
    unique_songs = list(dict.fromkeys(songs))
    song_to_id = {s: i for i, s in enumerate(unique_songs)}
    groups = np.array([song_to_id[s] for s in songs])
    return X, y, groups


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _precision_at_1(y_true: np.ndarray, proba: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    return float(y_true[np.argmax(proba)] == 1)


def _reciprocal_rank(y_true: np.ndarray, proba: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    for rank, idx in enumerate(np.argsort(proba)[::-1], 1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


def evaluate_model(
    name: str,
    model_fn,           # callable returning a fresh unfitted model
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_sample_weight: bool = False,
) -> dict:
    """Leave-one-song-out CV. model_fn() is called each fold to get a fresh model."""
    n_songs = len(np.unique(groups))
    print(f"  [{name}] evaluating ({n_songs} folds)...", end=" ", flush=True)

    p1, mrr, auc = [], [], []
    gkf = GroupKFold(n_splits=n_songs)

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if y_te.sum() == 0:
            continue

        mdl = model_fn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_sample_weight:
                sw = compute_sample_weight("balanced", y_tr)
                mdl.fit(X_tr, y_tr, sample_weight=sw)
            else:
                mdl.fit(X_tr, y_tr)

        proba = mdl.predict_proba(X_te)[:, 1]
        p1.append(_precision_at_1(y_te, proba))
        mrr.append(_reciprocal_rank(y_te, proba))
        if len(np.unique(y_te)) > 1:
            auc.append(roc_auc_score(y_te, proba))

    result = {
        "model": name,
        "P@1":     round(np.mean(p1),  4) if p1  else 0.0,
        "MRR":     round(np.mean(mrr), 4) if mrr else 0.0,
        "AUC-ROC": round(np.mean(auc), 4) if auc else 0.0,
        "n_eval":  len(p1),
    }
    print(f"P@1={result['P@1']:.3f}  MRR={result['MRR']:.3f}  AUC={result['AUC-ROC']:.3f}")
    return result


# ── Model factories ────────────────────────────────────────────────────────────

def _lr():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=1000, random_state=42,
        )),
    ])

def _lr_l1():
    """L1 (Lasso) logistic regression — automatic feature selection."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear",
            class_weight="balanced", C=0.5, max_iter=1000, random_state=42,
        )),
    ])

def _rf():
    return RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        max_features="sqrt", min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )

def _et():
    return ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced",
        max_features="sqrt", min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )

def _xgb():
    # scale_pos_weight ≈ neg/pos ratio handles imbalance natively
    neg_pos_ratio = (979 - 185) / max(185, 1)
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        min_child_weight=3, gamma=0.1,   # extra regularization for small data
        eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )

def _lgbm():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=20,           # reduced from 31 — less overfitting on 180 songs
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5,     # require at least 5 samples per leaf
            reg_alpha=0.1, reg_lambda=0.1,   # L1 + L2 regularization
            class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])

def _hgb():
    """HistGradientBoosting — sklearn native GBDT, handles small data well."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=5,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=42,
        )),
    ])

def _mlp():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            alpha=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=42,
        )),
    ])

def _stack():
    """Stacking: XGB + LGBM(pipeline) + ET as base; LR as meta."""
    estimators = [
        ("xgb",  _xgb()),
        ("lgbm", _lgbm()),   # already a Pipeline with scaler
        ("et",   _et()),
    ]
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=5, passthrough=True, n_jobs=1,  # n_jobs=1 avoids pickling issues
        )),
    ])

def _voting():
    """Soft-vote ensemble: LR + LR_L1 + HGB — all strong on small data."""
    return VotingClassifier(
        estimators=[
            ("lr",   _lr()),
            ("lr_l1", _lr_l1()),
            ("hgb",  _hgb()),
        ],
        voting="soft",
        n_jobs=1,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def save_model(model, metadata: dict) -> None:
    joblib.dump(model, MODELS_DIR / "viral_section_model.pkl")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(NLP_FEATURE_NAMES, f, indent=2)
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved → {MODELS_DIR}/viral_section_model.pkl")


def load_model():
    """Load saved model and feature names. Returns (model, feat_names)."""
    model = joblib.load(MODELS_DIR / "viral_section_model.pkl")
    with open(MODELS_DIR / "feature_names.json") as f:
        feat_names = json.load(f)
    return model, feat_names


def train_final_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    mdl = _rf()
    mdl.fit(X, y)
    return mdl


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_from_lyrics(
    raw_lyrics: str,
    model=None,
    feat_names: Optional[list] = None,
) -> Optional[pd.DataFrame]:
    """
    Parse pasted raw lyrics (with [Section] headers), compute 12 NLP features,
    and rank sections by viral probability. No Genius API call required.

    Parameters
    ----------
    raw_lyrics : full lyrics text with [Chorus], [Verse 1], etc. headers
    model      : trained model (loaded from disk if None)
    feat_names : feature name list (loaded from disk if None)

    Returns
    -------
    DataFrame sorted by viral_prob descending, or None if no sections parsed.
    """
    if model is None or feat_names is None:
        model, feat_names = load_model()

    sections = _parse_sections(raw_lyrics)
    if not sections:
        return None

    labels = list(sections.keys())
    n = len(labels)
    rows = []

    for idx, label in enumerate(labels):
        lines = sections[label]
        stype = _section_type(label)
        scores = _nlp_score(lines, idx, n, raw_lyrics)
        feat_vec = np.array(
            [scores.get(f, 0.0) for f in feat_names], dtype=np.float32
        ).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viral_prob = model.predict_proba(feat_vec)[0, 1]
        rows.append({
            "section_label":   label,
            "section_type":    stype,
            "viral_prob":      round(float(viral_prob), 4),
            "lyrics_preview":  " / ".join(lines[:3]),
            "full_lyrics":     "\n".join(lines),
            **scores,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("viral_prob", ascending=False)
        .reset_index(drop=True)
    )
    result["rank"] = result.index + 1
    return result


# Keep old name as alias for any callers that used predict_song()
def predict_song(
    title: str,
    artist: str,
    model=None,
    feat_names: Optional[list] = None,
) -> Optional[pd.DataFrame]:
    """Legacy: fetch lyrics from Genius then call predict_from_lyrics()."""
    import lyricsgenius
    from nlp_viral_detector import GENIUS_TOKEN

    if model is None or feat_names is None:
        model, feat_names = load_model()

    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        verbose=False,
    )
    genius.remove_section_headers = False
    song = genius.search_song(title, artist)
    if not song:
        print(f"Song not found on Genius: {title} by {artist}")
        return None

    return predict_from_lyrics(song.lyrics, model=model, feat_names=feat_names)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y, groups = load_training_data()
    pos_rate = y.mean()
    print(f"NLP features ({X.shape[1]}): {NLP_FEATURE_NAMES}")
    print(f"Positives: {y.sum()}/{len(y)}  ({pos_rate:.1%})\n")

    MODELS = [
        ("LogisticReg",    _lr,      False),
        ("LogisticReg_L1", _lr_l1,   False),
        ("RandomForest",   _rf,      False),
        ("ExtraTrees",     _et,      False),
        ("XGBoost",        _xgb,     False),  # scale_pos_weight handles imbalance
        ("LightGBM",       _lgbm,    False),  # built-in class_weight
        ("HGB",            _hgb,     False),  # HistGradientBoosting
        ("MLP(128-64-32)", _mlp,     False),
        ("Stacking",       _stack,   False),
        ("Voting(LR+L1+HGB)", _voting, False),
    ]

    print("=" * 65)
    print("Leave-one-song-out evaluation")
    print("=" * 65)

    results = []
    for name, fn, sw in MODELS:
        metrics = evaluate_model(name, fn, X, y, groups, use_sample_weight=sw)
        results.append(metrics)

    print("\n" + "=" * 65)
    print("Final Rankings")
    print("=" * 65)
    df_res = pd.DataFrame(results).sort_values("P@1", ascending=False)
    print(df_res[["model", "P@1", "MRR", "AUC-ROC", "n_eval"]].to_string(index=False))

    # ── Train final models on all data ─────────────────────────────────────────
    best = df_res.iloc[0]
    best_name = best["model"]
    print(f"\nBest model: {best_name}  (P@1={best['P@1']:.3f})")

    # RF always trained for feature importances
    print("\nTraining RandomForest on all data (for feature importances)...")
    final_rf = train_final_rf(X, y)
    joblib.dump(final_rf, MODELS_DIR / "viral_section_model_rf.pkl")
    importances = final_rf.feature_importances_
    top10 = sorted(zip(NLP_FEATURE_NAMES, importances), key=lambda x: -x[1])[:10]
    print("Top NLP features:")
    for feat, imp in top10:
        print(f"  {feat:35s}  {imp:.4f}")

    # Train the best model fresh on all data
    name_to_fn = {n: fn for n, fn, _ in MODELS}
    name_to_sw = {n: sw for n, _, sw in MODELS}
    print(f"\nTraining {best_name} on all data...")
    final_best = name_to_fn[best_name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name_to_sw[best_name]:
            sw_all = compute_sample_weight("balanced", y)
            final_best.fit(X, y, sample_weight=sw_all)
        else:
            final_best.fit(X, y)

    metadata = {
        "model_type": best_name,
        "feature_count": len(NLP_FEATURE_NAMES),
        "feature_names": NLP_FEATURE_NAMES,
        "training_sections": int(len(y)),
        "training_songs": int(len(np.unique(groups))),
        "positives": int(y.sum()),
        "best_model_cv": best.to_dict(),
        "all_cv_results": df_res.to_dict(orient="records"),
        "feature_importances": {
            f: round(float(i), 6) for f, i in zip(NLP_FEATURE_NAMES, importances)
        },
    }

    save_model(final_best, metadata)
    print("\nDone!")

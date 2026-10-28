"""
TikTok Viral Song Predictor — Model Training
=============================================
Data sources (all inside tiktok_virality/data/raw/):
  - tiktok_trending_tracks/tiktok.csv       6,746 TikTok songs + Spotify audio features
  - tiktok_popular_2020/TikTok_songs_2020.csv  292 songs from 2020
  - spotify_tiktok/spotify_tiktok_viral.parquet  509 confirmed viral songs

Viral label strategy:
  1 (viral)    : in Spotify viral playlist  OR  popularity >= 70
  0 (not viral): popularity < 50  AND  not in viral playlist
  drop          : ambiguous middle zone 50-69 not in viral list

Outputs saved to models/:
  song_virality_model.pkl   — best model + scaler + metadata
  feature_importances.csv   — RF feature importances
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Audio feature columns ─────────────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]

DERIVED_FEATURES = [
    "dance_energy", "mood_score", "tempo_norm",
    "acoustic_anti", "vocal_content",
]

FEATURE_COLS = AUDIO_FEATURES + DERIVED_FEATURES


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(use_expanded: bool = True) -> pd.DataFrame:
    print("=" * 60)
    print("Loading TikTok virality datasets...")

    expanded_path = DATA_DIR / "tiktok_expanded.csv"

    if use_expanded and expanded_path.exists():
        df = pd.read_csv(expanded_path)
        df = df[df["viral"] != -1].copy()
        df = df.dropna(subset=AUDIO_FEATURES)
        print(f"  tiktok_expanded.csv : {len(df):,} labeled tracks with audio features")
        v  = (df["viral"] == 1).sum()
        nv = (df["viral"] == 0).sum()
        print(f"  Viral (1)    : {v:,}  ({v/len(df)*100:.1f}%)")
        print(f"  Not viral (0): {nv:,}  ({nv/len(df)*100:.1f}%)")
        sources = df["source"].value_counts() if "source" in df.columns else None
        if sources is not None:
            for src, cnt in sources.items():
                print(f"    [{src}] {cnt:,}")
        return df
    else:
        # Legacy path — original 3-dataset approach
        tiktok = pd.read_csv(DATA_DIR / "tiktok_trending_tracks" / "tiktok.csv")
        tiktok = tiktok.drop_duplicates(subset="track_id")
        print(f"  tiktok.csv          : {len(tiktok):,} unique tracks")

        tiktok2020 = pd.read_csv(DATA_DIR / "tiktok_popular_2020" / "TikTok_songs_2020.csv")
        tiktok2020 = tiktok2020.rename(columns={"track_pop": "popularity", "duration_ms": "duration"})
        tiktok2020["genre"] = "TIKTOK_2020"
        print(f"  TikTok_songs_2020   : {len(tiktok2020):,} tracks")

        viral = pd.read_parquet(DATA_DIR / "spotify_tiktok" / "spotify_tiktok_viral.parquet")
        print(f"  spotify_viral       : {len(viral):,} confirmed viral tracks")

        # Legacy label creation
        viral_ids   = set(viral["track_id"].dropna())
        viral_names = set(viral["track_name"].str.lower().dropna())
        df = pd.concat([tiktok, tiktok2020], ignore_index=True)
        df["in_viral_playlist"] = (
            df.get("track_id", pd.Series(dtype=str)).isin(viral_ids) |
            df["track_name"].str.lower().isin(viral_names)
        )
        import numpy as np
        df["viral"] = np.select(
            [df["in_viral_playlist"], df["popularity"] >= 70, df["popularity"] < 50],
            [1, 1, 0], default=-1
        )
        df = df[df["viral"] != -1].copy()
        v, nv = (df["viral"] == 1).sum(), (df["viral"] == 0).sum()
        print(f"  Viral (1)    : {v:,}  Not viral (0): {nv:,}")
        return df


# keep create_labels as a no-op shim so app.py doesn't break
def create_labels(tiktok, viral=None):
    return tiktok


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Interaction features
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["mood_score"] = df["valence"] * df["energy"]          # happy + energetic
    df["tempo_norm"] = df["tempo"] / 200.0

    # How "electronic" vs acoustic
    df["acoustic_anti"] = 1.0 - df["acousticness"]

    # How vocal (not instrumental)
    df["vocal_content"] = 1.0 - df["instrumentalness"]

    return df


# ── Model training ────────────────────────────────────────────────────────────
def build_model_zoo(pos_neg_ratio: float) -> dict:
    models: dict = {
        "Logistic Regression": (
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42),
            True,   # needs scaling
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, max_features="sqrt",
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1
            ),
            False,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                scale_pos_weight=pos_neg_ratio,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            False,
        )

    if HAS_LGB:
        models["LightGBM"] = (
            lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                class_weight="balanced", random_state=42, verbose=-1,
            ),
            False,
        )

    return models


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> tuple[dict, dict, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pos_neg_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    model_zoo = build_model_zoo(pos_neg_ratio)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict = {}

    print("\nTraining models (5-fold stratified CV):")
    print("-" * 65)
    print(f"  {'Model':<25} {'AUC':>8} {'±':>6} {'AvgPrec':>10}")
    print("-" * 65)

    for name, (model, needs_scale) in model_zoo.items():
        X_in = X_scaled if needs_scale else X
        scores = cross_validate(
            model, X_in, y, cv=cv,
            scoring=["roc_auc", "average_precision"],
            return_train_score=False, n_jobs=1,
        )
        results[name] = {
            "auc": float(scores["test_roc_auc"].mean()),
            "auc_std": float(scores["test_roc_auc"].std()),
            "avg_precision": float(scores["test_average_precision"].mean()),
            "needs_scale": needs_scale,
        }
        r = results[name]
        print(f"  {name:<25} {r['auc']:>8.3f} {r['auc_std']:>6.3f} {r['avg_precision']:>10.3f}")

    print("-" * 65)
    return results, {k: v[0] for k, v in model_zoo.items()}, scaler


# ── Save best model ────────────────────────────────────────────────────────────
def save_best(
    X: np.ndarray,
    y: np.ndarray,
    results: dict,
    models: dict,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> dict:
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\nBest model → {best_name}  (AUC={results[best_name]['auc']:.3f})")

    X_scaled = scaler.transform(X)
    best_model = models[best_name]
    X_fit = X_scaled if results[best_name]["needs_scale"] else X
    best_model.fit(X_fit, y)

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "model_name": best_name,
        "feature_cols": FEATURE_COLS,
        "results": results,
        "training_shape": X.shape,
        "positive_rate": float(y.mean()),
    }
    out = MODEL_DIR / "song_virality_model.pkl"
    joblib.dump(artifact, out)
    print(f"Saved  → {out}")

    # Also save RF for feature importances
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = MODEL_DIR / "feature_importances.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved  → {imp_path}")

    print("\nTop features (Random Forest):")
    for _, row in imp_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<30} {row['importance']*100:5.1f}%  {bar}")

    # Save results JSON for dashboard
    results_path = MODEL_DIR / "cv_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    return artifact


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data(use_expanded=True)
    df = engineer_features(df)

    # Drop rows with missing features
    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values
    y = df["viral"].values

    print(f"\nFinal training set: {len(df):,} songs  |  {len(FEATURE_COLS)} features")

    results, models, scaler = train_and_evaluate(X, y)
    save_best(X, y, results, models, scaler, df)

    print("\nDone! Run: streamlit run app.py")


if __name__ == "__main__":
    main()

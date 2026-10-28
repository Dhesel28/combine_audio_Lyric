"""
TikTok Viral Predictor — Streamlit Dashboard
============================================
6 tabs:
  1. Song Virality Checker  — enter a song → viral probability (audio features)
  2. Viral Snippet Finder   — enter a song → which section will trend on TikTok
  3. Data Explorer          — visualise the 6,163-song training dataset
  4. Model Performance      — CV results + audio vs lyrics comparison
  5. Feature Analysis       — top audio feature importances
  6. Lyrics Analysis        — NLP features + before/after AUC lift

Run:
  cd tiktok_virality
  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TikTok Viral Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #1a1a2e; }
    .metric-card {
        background: #16213e; border-radius: 12px; padding: 20px;
        text-align: center; border: 1px solid #0f3460;
    }
    .metric-num { font-size: 2.4rem; font-weight: 700; }
    .section-bar { border-radius: 8px; padding: 10px 16px; margin: 4px 0; }
    h1, h2, h3 { color: #e94560 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .reason-text { color: #a8dadc; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths & cache ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

# Load .env from viral_lyrics_analyzer sibling
_env_path = BASE_DIR.parent / "viral_lyrics_analyzer" / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass


@st.cache_resource
def load_model_artifact():
    # Prefer lyrics model (AUC 0.751), fall back to audio-only
    for name in ["song_virality_model_lyrics.pkl", "song_virality_model.pkl"]:
        path = MODEL_DIR / name
        if path.exists():
            import joblib
            return joblib.load(path)
    return None


@st.cache_data
def load_tiktok_data():
    # Use expanded dataset if available
    expanded = DATA_DIR / "tiktok_with_lyrics.csv"
    if expanded.exists():
        df = pd.read_csv(expanded)
        return df
    fallback = DATA_DIR / "tiktok_expanded.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    df = pd.read_csv(DATA_DIR / "tiktok_trending_tracks" / "tiktok.csv")
    return df.drop_duplicates(subset="track_id")


@st.cache_data
def load_viral_data():
    path = DATA_DIR / "spotify_tiktok" / "spotify_tiktok_viral.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_cv_results():
    path = MODEL_DIR / "cv_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_lyrics_comparison():
    path = MODEL_DIR / "lyrics_comparison.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_feature_importances():
    path = MODEL_DIR / "feature_importances.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Helper: predict song virality ─────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]
DERIVED = [
    "dance_energy", "mood_score", "tempo_norm",
    "acoustic_anti", "vocal_content",
]
ALL_FEATURES = AUDIO_FEATURES + DERIVED

FEATURE_DISPLAY_NAMES = {
    "danceability": "Danceability",
    "energy": "Energy",
    "key": "Key",
    "loudness": "Loudness (dB)",
    "mode": "Mode (Major/Minor)",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "valence": "Valence (Mood)",
    "tempo": "Tempo (BPM)",
    "dance_energy": "Dance×Energy",
    "mood_score": "Mood Score",
    "tempo_norm": "Tempo (norm)",
    "acoustic_anti": "Electronic-ness",
    "vocal_content": "Vocal Content",
}


def engineer_row(d: dict) -> dict:
    d = dict(d)
    d["dance_energy"] = d.get("danceability", 0) * d.get("energy", 0)
    d["mood_score"] = d.get("valence", 0) * d.get("energy", 0)
    d["tempo_norm"] = d.get("tempo", 120) / 200.0
    d["acoustic_anti"] = 1.0 - d.get("acousticness", 0)
    d["vocal_content"] = 1.0 - d.get("instrumentalness", 0)
    return d


def predict_virality(audio_dict: dict, artifact) -> float:
    if artifact is None:
        return 0.5
    row = engineer_row(audio_dict)
    X = np.array([[row.get(f, 0.0) for f in ALL_FEATURES]])
    model = artifact["model"]
    scaler = artifact["scaler"]
    if artifact["model_name"] == "Logistic Regression":
        X = scaler.transform(X)
    return float(model.predict_proba(X)[0][1])


def prob_color(p: float) -> str:
    if p >= 0.70:
        return "#00c896"
    if p >= 0.50:
        return "#ffd166"
    return "#ef4444"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎵 TikTok Viral Predictor")
st.markdown("*Which songs go viral on TikTok? — ML on 6,163 songs · Genius lyrics · MiniLM embeddings*")

artifact      = load_model_artifact()
tiktok_df     = load_tiktok_data()
viral_df      = load_viral_data()
lyr_comp      = load_lyrics_comparison()

model_trained = artifact is not None

if not model_trained:
    st.warning(
        "⚠️ Model not trained yet. Run `python train_model.py` then `python add_lyrics.py` first.",
        icon="⚠️",
    )

# ── Summary cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

# Best AUC: prefer lyrics comparison data
if lyr_comp:
    best_auc = lyr_comp["audio_lyrics"]["auc"]
    auc_str  = f"{best_auc:.3f}"
elif artifact:
    auc_val = artifact["results"].get(artifact["model_name"], {}).get("auc", "—")
    auc_str = f"{auc_val:.3f}" if isinstance(auc_val, float) else "—"
else:
    auc_str = "—"

songs_with_lyrics = int(lyr_comp["songs_with_lyrics"]) if lyr_comp else 0
total_songs = len(tiktok_df)

with c1:
    st.metric("Training Songs", f"{total_songs:,}")
with c2:
    st.metric("Songs with Lyrics", f"{songs_with_lyrics:,}")
with c3:
    st.metric("AUC (Audio+Lyrics)", auc_str)
with c4:
    if lyr_comp:
        lift = lyr_comp["audio_lyrics"]["auc"] - lyr_comp["audio_only"]["auc"]
        st.metric("AUC Lift vs Audio-Only", f"+{lift:.3f}")
    else:
        st.metric("Features", "16 audio + lyrics")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎵 Song Virality Checker",
    "✂️ Viral Snippet Finder",
    "📊 Data Explorer",
    "🏆 Model Performance",
    "🔬 Feature Analysis",
    "📝 Lyrics Analysis",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — SONG VIRALITY CHECKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("### 🎵 Song Virality Checker")
    st.markdown(
        "Adjust the audio feature sliders to match a song's Spotify profile, "
        "or search for a song in our training data."
    )

    col_search, col_space = st.columns([2, 1])
    with col_search:
        search_query = st.text_input(
            "Search training data (song or artist)",
            placeholder="e.g. Blinding Lights",
        )

    if search_query:
        mask = (
            tiktok_df["track_name"].str.lower().str.contains(search_query.lower(), na=False) |
            tiktok_df["artist_name"].str.lower().str.contains(search_query.lower(), na=False)
        )
        found = tiktok_df[mask].head(5)
        if found.empty:
            st.info("No matching song found in training data.")
        else:
            st.markdown("**Results from training data:**")
            chosen = st.selectbox(
                "Select to auto-fill sliders",
                options=["— select —"] + [
                    f"{r['track_name']} — {r['artist_name']}" for _, r in found.iterrows()
                ],
            )
            if chosen != "— select —":
                name, art = chosen.split(" — ", 1)
                row = found[
                    (found["track_name"] == name) & (found["artist_name"] == art)
                ].iloc[0]
                st.session_state["prefill"] = row.to_dict()

    prefill = st.session_state.get("prefill", {})

    st.markdown("#### Audio Feature Sliders")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        danceability = st.slider("Danceability", 0.0, 1.0, float(prefill.get("danceability", 0.70)), 0.01)
        energy = st.slider("Energy", 0.0, 1.0, float(prefill.get("energy", 0.65)), 0.01)
        valence = st.slider("Valence (mood)", 0.0, 1.0, float(prefill.get("valence", 0.55)), 0.01)
        tempo = st.slider("Tempo (BPM)", 60.0, 200.0, float(prefill.get("tempo", 120.0)), 1.0)

    with col_b:
        speechiness = st.slider("Speechiness", 0.0, 1.0, float(prefill.get("speechiness", 0.06)), 0.01)
        acousticness = st.slider("Acousticness", 0.0, 1.0, float(prefill.get("acousticness", 0.15)), 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, float(prefill.get("instrumentalness", 0.0)), 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, float(prefill.get("liveness", 0.12)), 0.01)

    with col_c:
        loudness = st.slider("Loudness (dB)", -40.0, 0.0, float(prefill.get("loudness", -6.0)), 0.5)
        mode = st.selectbox("Mode", ["Major (1)", "Minor (0)"], index=0)
        key = st.slider("Key (0=C … 11=B)", 0, 11, int(prefill.get("key", 5)))

    mode_val = 1 if "Major" in mode else 0

    audio_input = {
        "danceability": danceability, "energy": energy, "valence": valence,
        "tempo": tempo, "speechiness": speechiness, "acousticness": acousticness,
        "instrumentalness": instrumentalness, "liveness": liveness,
        "loudness": loudness, "mode": mode_val, "key": key,
    }

    prob = predict_virality(audio_input, artifact)
    color = prob_color(prob)

    st.divider()
    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div style='color:#aaa'>Viral Probability</div>"
            f"<div class='metric-num' style='color:{color}'>{prob*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with r2:
        label = "🔥 HIGH VIRAL POTENTIAL" if prob >= 0.70 else ("⚡ MODERATE" if prob >= 0.50 else "❄️ LOW VIRAL POTENTIAL")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div style='color:#aaa'>Verdict</div>"
            f"<div class='metric-num' style='font-size:1.1rem; color:{color}'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with r3:
        mood = danceability * energy
        mood_label = "Party / Dance" if mood > 0.5 else ("Chill / Emotional" if valence < 0.4 else "Pop / Upbeat")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div style='color:#aaa'>Vibe</div>"
            f"<div class='metric-num' style='font-size:1.1rem; color:#a8dadc'>{mood_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Radar chart
    cats = ["Danceability", "Energy", "Valence", "Speechiness", "Acousticness", "Liveness"]
    vals = [danceability, energy, valence, speechiness, acousticness, liveness]
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor=f"rgba(233,69,96,0.2)",
        line=dict(color="#e94560", width=2),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, height=350,
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    if not model_trained:
        st.info("Probability shown is a placeholder (0.5). Train the model to see real predictions.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — VIRAL SNIPPET FINDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### ✂️ Viral Snippet Finder")
    st.markdown(
        "Enter any song. The engine fetches its lyrics, parses every section, "
        "and ranks them by TikTok viral potential using 25 NLP features."
    )

    col_song, col_artist = st.columns(2)
    with col_song:
        snip_song = st.text_input("Song title", value="Espresso", key="snip_song")
    with col_artist:
        snip_artist = st.text_input("Artist", value="Sabrina Carpenter", key="snip_artist")

    # Optional audio features
    with st.expander("🎛️ Optional: provide Spotify audio features to improve prediction"):
        sa, sb, sc = st.columns(3)
        with sa:
            s_dance = st.slider("Danceability", 0.0, 1.0, 0.70, 0.01, key="s_d")
            s_energy = st.slider("Energy", 0.0, 1.0, 0.65, 0.01, key="s_e")
        with sb:
            s_valence = st.slider("Valence", 0.0, 1.0, 0.80, 0.01, key="s_v")
            s_tempo = st.slider("Tempo (BPM)", 60.0, 200.0, 110.0, 1.0, key="s_t")
        with sc:
            s_speech = st.slider("Speechiness", 0.0, 1.0, 0.06, 0.01, key="s_s")
            s_acoustic = st.slider("Acousticness", 0.0, 1.0, 0.05, 0.01, key="s_a")
        s_audio = {
            "danceability": s_dance, "energy": s_energy, "valence": s_valence,
            "tempo": s_tempo, "speechiness": s_speech, "acousticness": s_acoustic,
            "instrumentalness": 0.0, "liveness": 0.12, "loudness": -6.0,
            "mode": 1, "key": 5,
        }
        use_audio = True

    analyze_btn = st.button("🔍 Find Viral Snippet", type="primary")

    if analyze_btn:
        if not os.getenv("GENIUS_ACCESS_TOKEN"):
            st.error("GENIUS_ACCESS_TOKEN not set. Add it to viral_lyrics_analyzer/.env")
        else:
            with st.spinner(f"Fetching lyrics for '{snip_song}' by {snip_artist}..."):
                try:
                    from snippet_engine import find_viral_snippet
                    audio_feats = s_audio if use_audio else None
                    results = find_viral_snippet(snip_song, snip_artist, audio_features=audio_feats)
                except Exception as e:
                    st.error(f"Error: {e}")
                    results = []

            if not results:
                st.error("Could not fetch or parse lyrics. Check the song name / artist or verify Genius API token.")
            else:
                st.success(f"Found {len(results)} sections! Top viral clip ↓")

                # Section probability bars
                st.markdown("#### Section Viral Scores")
                max_score = max(r["viral_score"] for r in results)
                for i, r in enumerate(results):
                    pct = r["viral_score"] / max(max_score, 0.001) * 100
                    color = "#e94560" if i == 0 else ("#ffd166" if i <= 2 else "#4cc9f0")
                    star = " ← #1 ★" if i == 0 else ""
                    st.markdown(
                        f"<div class='section-bar' style='background:{color}22; border-left: 4px solid {color}'>"
                        f"<b style='color:{color}'>[{r['label']}]{star}</b> &nbsp;"
                        f"<span style='color:#eee'>{r['viral_score']*100:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Bar chart
                bar_df = pd.DataFrame([
                    {"Section": r["label"], "Viral Score (%)": r["viral_score"] * 100}
                    for r in results
                ])
                fig_bar = px.bar(
                    bar_df, x="Viral Score (%)", y="Section", orientation="h",
                    color="Viral Score (%)", color_continuous_scale="RdYlGn",
                    title="Section Viral Probability",
                )
                fig_bar.update_layout(
                    paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                    font=dict(color="white"), height=max(300, len(results) * 45),
                    coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Top section detail
                top = results[0]
                st.markdown(f"#### Why **[{top['label']}]** scores highest:")
                for reason in top["top_reasons"]:
                    st.markdown(f"- {reason}")

                st.markdown("**Predicted viral clip lyrics:**")
                preview = top["text"][:300]
                st.code(preview + ("..." if len(top["text"]) > 300 else ""), language=None)

                # NLP feature breakdown
                with st.expander("📐 Full NLP feature breakdown (top section)"):
                    feat_df = pd.DataFrame.from_dict(
                        {k: [round(v, 4)] for k, v in top["features"].items()},
                        orient="columns",
                    ).T.rename(columns={0: "value"})
                    st.dataframe(feat_df, use_container_width=True)

                # Score decomposition
                st.markdown("#### Score Decomposition")
                decomp_df = pd.DataFrame([{
                    "Component": "Song Audio Score (35%)",
                    "Value": top["song_prob"],
                    "Weight": 0.35,
                    "Contribution": top["song_prob"] * 0.35,
                }, {
                    "Component": "Section NLP Score (55%)",
                    "Value": top["nlp_score"],
                    "Weight": 0.55,
                    "Contribution": top["nlp_score"] * 0.55,
                }, {
                    "Component": "Position Boost (10%)",
                    "Value": top["pos_boost"],
                    "Weight": 0.10,
                    "Contribution": top["pos_boost"] * 0.10,
                }])
                fig_decomp = px.bar(
                    decomp_df, x="Contribution", y="Component", orientation="h",
                    color="Component",
                    color_discrete_sequence=["#00c896", "#e94560", "#ffd166"],
                )
                fig_decomp.update_layout(
                    paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                    font=dict(color="white"), showlegend=False, height=200,
                )
                st.plotly_chart(fig_decomp, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — DATA EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### 📊 TikTok Data Explorer")

    df = tiktok_df.copy()

    # Genre filter
    genres = ["All"] + sorted(df["genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Filter by genre", genres)
    if selected_genre != "All":
        df = df[df["genre"] == selected_genre]

    st.caption(f"Showing {len(df):,} songs")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Songs", f"{len(df):,}")
    e2.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
    e3.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")
    e4.metric("Avg Energy", f"{df['energy'].mean():.2f}")

    # Popularity distribution
    fig_pop = px.histogram(
        df, x="popularity", nbins=30,
        title="Popularity Score Distribution",
        color_discrete_sequence=["#e94560"],
    )
    fig_pop.add_vline(x=70, line_dash="dash", line_color="#00c896",
                      annotation_text="Viral threshold (70)")
    fig_pop.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="white"), height=350,
    )
    st.plotly_chart(fig_pop, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        # Danceability vs Energy scatter
        fig_scatter = px.scatter(
            df.sample(min(1000, len(df))), x="danceability", y="energy",
            color="popularity", color_continuous_scale="RdYlGn",
            title="Danceability vs Energy (coloured by popularity)",
            opacity=0.6,
        )
        fig_scatter.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font=dict(color="white"), height=380,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        # Valence vs Popularity
        fig_val = px.scatter(
            df.sample(min(1000, len(df))), x="valence", y="popularity",
            color="energy", color_continuous_scale="Viridis",
            title="Valence vs Popularity",
            opacity=0.6,
        )
        fig_val.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font=dict(color="white"), height=380,
        )
        st.plotly_chart(fig_val, use_container_width=True)

    # Audio feature correlation with popularity
    corr_cols = ["danceability", "energy", "valence", "tempo",
                 "speechiness", "acousticness", "loudness", "liveness"]
    corr_vals = [df[c].corr(df["popularity"]) for c in corr_cols]
    corr_df = pd.DataFrame({"Feature": corr_cols, "Correlation with Popularity": corr_vals})
    corr_df = corr_df.sort_values("Correlation with Popularity")

    fig_corr = px.bar(
        corr_df, x="Correlation with Popularity", y="Feature", orientation="h",
        title="Audio Feature Correlation with Popularity",
        color="Correlation with Popularity",
        color_continuous_scale="RdBu",
    )
    fig_corr.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="white"), height=380, coloraxis_showscale=False,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Top songs table
    st.markdown("#### Top 20 Most Popular Songs in Dataset")
    top_songs = (
        df.nlargest(20, "popularity")[
            ["track_name", "artist_name", "popularity", "danceability", "energy", "valence", "tempo", "genre"]
        ]
        .reset_index(drop=True)
    )
    top_songs.index = top_songs.index + 1
    st.dataframe(top_songs, use_container_width=True)

    # Confirmed viral songs from Spotify
    st.markdown("#### Spotify-Confirmed Viral Songs (509 tracks)")
    vshow = viral_df[["track_name", "artist_name", "popularity", "playlist_source"]].head(20)
    st.dataframe(vshow.reset_index(drop=True), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### 🏆 Model Performance — Stratified 5-Fold CV")

    lyr_comp = load_lyrics_comparison()

    if lyr_comp:
        # ── Before/After headline ──────────────────────────────────────────────
        audio_auc  = lyr_comp["audio_only"]["auc"]
        lyrics_auc = lyr_comp["audio_lyrics"]["auc"]
        lift       = lyrics_auc - audio_auc

        m1, m2, m3 = st.columns(3)
        m1.metric("Audio-Only Best AUC",   f"{audio_auc:.3f}",  delta=None)
        m2.metric("Audio+Lyrics Best AUC", f"{lyrics_auc:.3f}", delta=f"+{lift:.3f}")
        m3.metric("Lyrics Coverage",
                  f"{lyr_comp['songs_with_lyrics']:,}/{lyr_comp['total_songs']:,}",
                  delta=f"{lyr_comp['songs_with_lyrics']/lyr_comp['total_songs']*100:.1f}%")

        st.divider()

        # ── Side-by-side grouped bar chart ────────────────────────────────────
        models = list(lyr_comp["audio_only"]["results"].keys())
        rows = []
        for m in models:
            rows.append({"Model": m, "Type": "Audio Only",
                         "AUC": lyr_comp["audio_only"]["results"][m]["auc"],
                         "std": lyr_comp["audio_only"]["results"][m]["auc_std"]})
            rows.append({"Model": m, "Type": "Audio + Lyrics",
                         "AUC": lyr_comp["audio_lyrics"]["results"][m]["auc"],
                         "std": lyr_comp["audio_lyrics"]["results"][m]["auc_std"]})
        cmp_df = pd.DataFrame(rows)

        fig_cmp = px.bar(
            cmp_df, x="AUC", y="Model", color="Type", orientation="h",
            barmode="group",
            title="Audio-Only vs Audio+Lyrics — AUC per Model",
            color_discrete_map={"Audio Only": "#6c5ce7", "Audio + Lyrics": "#00c896"},
            range_x=[0.5, 0.85],
        )
        fig_cmp.add_vline(x=0.5, line_dash="dash", line_color="gray",
                          annotation_text="Random")
        fig_cmp.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font=dict(color="white"), height=400, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Detailed tables ────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        def _make_table(result_dict, label):
            rows = []
            for name, r in result_dict.items():
                rows.append({"Model": name, "AUC": r["auc"],
                             "± std": r["auc_std"], "Avg Prec": r["avg_precision"]})
            df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
            df.index += 1
            return df

        with col_a:
            st.markdown("#### Audio Only")
            df_a = _make_table(lyr_comp["audio_only"]["results"], "audio")
            st.dataframe(df_a.style.format({"AUC": "{:.3f}", "± std": "{:.3f}",
                                            "Avg Prec": "{:.3f}"})
                         .background_gradient(subset=["AUC"], cmap="RdYlGn"),
                         use_container_width=True)

        with col_b:
            st.markdown("#### Audio + Lyrics")
            df_b = _make_table(lyr_comp["audio_lyrics"]["results"], "lyrics")
            st.dataframe(df_b.style.format({"AUC": "{:.3f}", "± std": "{:.3f}",
                                            "Avg Prec": "{:.3f}"})
                         .background_gradient(subset=["AUC"], cmap="RdYlGn"),
                         use_container_width=True)

        feat_note = (
            f"SelectKBest selected **{lyr_comp.get('lyrics_features_selected', 5)} lyrics features** "
            f"+ **{lyr_comp.get('embedding_features_selected', 55)} MiniLM embedding dims** "
            f"(0 audio features — embeddings dominated)"
        )
        st.info(feat_note)

    else:
        cv_results = load_cv_results()
        if cv_results is None:
            st.info("Train the model first: `python train_model.py` then `python add_lyrics.py`")
        else:
            rows = []
            for name, r in cv_results.items():
                rows.append({"Model": name, "AUC-ROC": r["auc"],
                             "AUC ± std": r["auc_std"], "Avg Precision": r["avg_precision"]})
            perf_df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
            perf_df.index += 1
            st.success(f"🥇 Best model: **{perf_df.iloc[0]['Model']}**  (AUC = {perf_df.iloc[0]['AUC-ROC']:.3f})")
            st.dataframe(perf_df.style.format({"AUC-ROC": "{:.3f}", "AUC ± std": "{:.3f}",
                                               "Avg Precision": "{:.3f}"})
                         .background_gradient(subset=["AUC-ROC"], cmap="RdYlGn"),
                         use_container_width=True)

    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | What it measures | Our interpretation |
|--------|-----------------|-------------------|
| **AUC-ROC** | How well model separates viral from non-viral | > 0.80 = strong; 0.5 = random |
| **Avg Precision** | Precision across all recall thresholds (handles imbalance) | Higher = fewer false positives |
| **± std** | Stability across 5 CV folds | Lower = more consistent |
        """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — FEATURE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("### 🔬 Feature Analysis")

    imp_df = load_feature_importances()
    if imp_df is None:
        st.info("Train the model first: `python train_model.py`")
    else:
        # Feature importance bar chart
        fig_imp = px.bar(
            imp_df.head(15), x="importance", y="feature", orientation="h",
            title="Top Audio Feature Importances (Random Forest)",
            color="importance", color_continuous_scale="Reds",
        )
        fig_imp.update_layout(
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font=dict(color="white"), height=480, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### What drives TikTok virality in audio?")
        insights = {
            "danceability": "Songs you can move to → high shareability on TikTok",
            "energy": "High-energy songs dominate TikTok feeds",
            "dance_energy": "Dance × Energy interaction — the 'bop' factor",
            "mood_score": "Happy + energetic = viral combo",
            "valence": "Positive mood songs trend more on TikTok",
            "tempo_norm": "Mid-to-high BPM sweet spot (~100-140 BPM)",
            "vocal_content": "Vocal songs outperform instrumentals on TikTok",
            "loudness_norm": "Louder mastered songs cut through on mobile",
            "acousticness": "Electronic > acoustic for TikTok trends",
        }
        for feat, insight in list(insights.items())[:6]:
            match = imp_df[imp_df["feature"] == feat]
            if not match.empty:
                imp_val = match.iloc[0]["importance"]
                st.markdown(
                    f"- **{FEATURE_DISPLAY_NAMES.get(feat, feat)}** "
                    f"({imp_val*100:.1f}%): {insight}"
                )

    # Section NLP weights (always visible — built into snippet engine)
    st.divider()
    st.markdown("#### Viral Snippet — Section Feature Weights")
    st.markdown("These weights power the **Viral Snippet Finder** (derived from viral chorus patterns):")

    from snippet_engine import _SECTION_WEIGHTS
    weights_df = pd.DataFrame([
        {"Feature": k, "Weight": v}
        for k, v in sorted(_SECTION_WEIGHTS.items(), key=lambda x: -x[1])
    ])
    fig_w = px.bar(
        weights_df, x="Weight", y="Feature", orientation="h",
        title="NLP Section Feature Weights (Viral Snippet Algorithm)",
        color="Weight", color_continuous_scale="Greens",
    )
    fig_w.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="white"), height=380, coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("""
**How the snippet algorithm works:**

1. **Hook keyword score** (22%) — presence of emotionally charged words ('love', 'baby', 'yeah', etc.)
2. **Repeat score** (18%) — how much the section repeats itself (earworm factor)
3. **Chorus/Hook type** (15%) — structural label is the strongest single signal
4. **Repetition pattern** (10%) — structural repetition detected
5. **VADER sentiment** (8%) — positive emotional charge
6. **Rhyme score** (7%) — end-rhyme density (singability)
7. **Short line score** (7%) — punchy lines ≤ 8 words fit TikTok's attention span
8. **Vowel density** (5%) — easy to sing along
9. **Pronoun density** (4%) — personal connection (I/you/we)
10. **Call-response** (4%) — crowd-participation phrasing
    """)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — LYRICS ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("### 📝 Lyrics Analysis")

    lyric_cols = [c for c in tiktok_df.columns if c.startswith("lyr_") and c != "lyr_has_lyrics"]
    has_lyrics_col = "lyr_has_lyrics" in tiktok_df.columns

    if not has_lyrics_col:
        st.info("Run `python add_lyrics.py` to fetch lyrics and compute NLP features.")
    else:
        ldf = tiktok_df.copy()

        # ── Coverage stats ─────────────────────────────────────────────────────
        n_with = int(ldf["lyr_has_lyrics"].sum())
        n_total = len(ldf)
        pct = n_with / n_total * 100

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Songs with Lyrics", f"{n_with:,}")
        s2.metric("Coverage", f"{pct:.1f}%")
        s3.metric("Avg Word Count", f"{ldf[ldf['lyr_has_lyrics']==1]['lyr_word_count'].mean():.0f}")
        s4.metric("Avg Hook Density", f"{ldf[ldf['lyr_has_lyrics']==1]['lyr_hook_density'].mean():.2f}")

        st.divider()

        # ── AUC lift bar ───────────────────────────────────────────────────────
        lyr_comp_t6 = load_lyrics_comparison()
        if lyr_comp_t6:
            st.markdown("#### AUC Lift from Adding Lyrics")
            audio_auc  = lyr_comp_t6["audio_only"]["auc"]
            lyrics_auc = lyr_comp_t6["audio_lyrics"]["auc"]

            lift_df = pd.DataFrame([
                {"Approach": "Audio Only (16 features)",            "AUC": audio_auc},
                {"Approach": "Audio + Lyrics + MiniLM (60 selected)", "AUC": lyrics_auc},
            ])
            fig_lift = px.bar(
                lift_df, x="AUC", y="Approach", orientation="h",
                color="Approach",
                color_discrete_sequence=["#6c5ce7", "#00c896"],
                range_x=[0.5, 0.85],
                title=f"AUC: {audio_auc:.3f} → {lyrics_auc:.3f}  (+{lyrics_auc-audio_auc:.3f})",
            )
            fig_lift.add_vline(x=0.5, line_dash="dash", line_color="gray")
            fig_lift.update_layout(
                paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                font=dict(color="white"), height=220, showlegend=False,
            )
            st.plotly_chart(fig_lift, use_container_width=True)

        st.divider()

        # ── Viral vs Non-viral lyric feature comparison ────────────────────────
        st.markdown("#### Viral vs Non-Viral — Lyric Feature Averages")

        if "viral" in ldf.columns:
            viral_ldf   = ldf[(ldf["viral"] == 1) & (ldf["lyr_has_lyrics"] == 1)]
            noviral_ldf = ldf[(ldf["viral"] == 0) & (ldf["lyr_has_lyrics"] == 1)]

            compare_feats = [
                "lyr_hook_density", "lyr_repetition", "lyr_rhyme_score",
                "lyr_hook_words", "lyr_dance_words", "lyr_quotable_score",
                "lyr_vader_compound", "lyr_positivity_ratio", "lyr_lex_diversity",
                "lyr_syllable_rate",
            ]
            feat_labels = {
                "lyr_hook_density":    "Hook Density",
                "lyr_repetition":      "Repetition",
                "lyr_rhyme_score":     "Rhyme Score",
                "lyr_hook_words":      "Hook Word Count",
                "lyr_dance_words":     "Dance Word Count",
                "lyr_quotable_score":  "Quotable Lines",
                "lyr_vader_compound":  "VADER Sentiment",
                "lyr_positivity_ratio":"Positivity Ratio",
                "lyr_lex_diversity":   "Lexical Diversity",
                "lyr_syllable_rate":   "Syllable Rate",
            }

            cmp_rows = []
            for f in compare_feats:
                if f in viral_ldf.columns:
                    cmp_rows.append({
                        "Feature":    feat_labels.get(f, f),
                        "Viral":      viral_ldf[f].mean(),
                        "Not Viral":  noviral_ldf[f].mean(),
                    })
            cmp_feat_df = pd.DataFrame(cmp_rows)

            # Normalize for display
            for col in ["Viral", "Not Viral"]:
                mx = cmp_feat_df[col].abs().max()
                if mx > 0:
                    cmp_feat_df[f"{col} (norm)"] = cmp_feat_df[col] / mx

            fig_vn = go.Figure()
            fig_vn.add_trace(go.Bar(
                name="Viral", y=cmp_feat_df["Feature"],
                x=cmp_feat_df["Viral (norm)"],
                orientation="h", marker_color="#e94560",
            ))
            fig_vn.add_trace(go.Bar(
                name="Not Viral", y=cmp_feat_df["Feature"],
                x=cmp_feat_df["Not Viral (norm)"],
                orientation="h", marker_color="#6c5ce7",
            ))
            fig_vn.update_layout(
                barmode="group", title="Lyric Features: Viral vs Not Viral (normalised)",
                paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                font=dict(color="white"), height=420,
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_vn, use_container_width=True)

        st.divider()

        # ── Song search with lyrics preview ────────────────────────────────────
        st.markdown("#### Search a Song's Lyric Features")

        cache_path = DATA_DIR / "lyrics_cache.csv"
        if cache_path.exists():
            cache_df = pd.read_csv(cache_path)
            cache_df = cache_df[cache_df["lyrics"].notna()]

            search_lyr = st.text_input("Search song or artist", placeholder="e.g. Blinding Lights")
            if search_lyr:
                merged_search = ldf.merge(cache_df[["_key","lyrics","url"]], on="_key", how="left")
                mask = (
                    merged_search["track_name"].str.lower().str.contains(search_lyr.lower(), na=False) |
                    merged_search["artist_name"].str.lower().str.contains(search_lyr.lower(), na=False)
                )
                hits = merged_search[mask & merged_search["lyrics"].notna()].head(5)

                if hits.empty:
                    st.info("No matching song with lyrics found.")
                else:
                    chosen_lyr = st.selectbox(
                        "Select song",
                        ["— select —"] + [f"{r['track_name']} — {r['artist_name']}" for _, r in hits.iterrows()],
                        key="lyr_select"
                    )
                    if chosen_lyr != "— select —":
                        name_l, art_l = chosen_lyr.split(" — ", 1)
                        song_row = hits[(hits["track_name"] == name_l) & (hits["artist_name"] == art_l)].iloc[0]

                        col_lyr, col_feat = st.columns([1, 1])
                        with col_lyr:
                            st.markdown(f"**Lyrics preview** — [{song_row['track_name']}]({song_row.get('url','')})")
                            st.text_area("", value=str(song_row["lyrics"])[:800], height=300, disabled=True)

                        with col_feat:
                            st.markdown("**NLP Feature Values**")
                            feat_vals = {feat_labels.get(f, f): round(float(song_row[f]), 4)
                                         for f in compare_feats if f in song_row}
                            feat_vals["Viral Label"] = "✅ Viral" if song_row.get("viral") == 1 else "❌ Not Viral"
                            feat_show_df = pd.DataFrame(list(feat_vals.items()), columns=["Feature", "Value"])
                            st.dataframe(feat_show_df, use_container_width=True, hide_index=True)
        else:
            st.info("Lyrics cache not found. Run `python add_lyrics.py` first.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8rem'>"
    "TikTok Viral Predictor · SUPA Spring 2026 · "
    "6,163 songs · Genius lyrics · MiniLM embeddings · AUC 0.751"
    "</div>",
    unsafe_allow_html=True,
)

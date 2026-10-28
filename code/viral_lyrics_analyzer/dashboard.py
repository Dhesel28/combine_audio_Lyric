"""
Streamlit dashboard: Viral Lyrics Section Analyzer
Run with: streamlit run dashboard.py

Tabs:
1. TikTok Trending Songs   - current trending songs list
2. Lyric Section Breakdown  - which section is most viral per song
3. Cross-Platform Comparison - TikTok vs YouTube Shorts
4. Analyze Any Song         - enter song + artist and analyze live
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from tiktok_trends import get_trending_songs
from genius_analyzer import analyze_song, parse_lyrics_into_sections
from youtube_shorts import cross_check_song, _has_yt_key


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Viral Lyrics Analyzer",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Viral Lyrics Section Analyzer")
st.caption("Which part of a song goes viral on TikTok & YouTube Shorts?")


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def cached_trending_songs():
    return get_trending_songs(count=50)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_analyze_song(title: str, artist: str):
    return analyze_song(title, artist)


def load_saved_results() -> pd.DataFrame:
    path = DATA_DIR / "cross_platform_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    path2 = DATA_DIR / "lyric_sections_analysis.csv"
    if path2.exists():
        return pd.read_csv(path2)
    return pd.DataFrame()


def section_color(section_type: str) -> str:
    colors = {
        "Chorus/Hook": "#FF6B6B",
        "Verse": "#4ECDC4",
        "Bridge": "#FFE66D",
        "Pre-Chorus": "#A8E6CF",
        "Intro": "#B0C4DE",
        "Outro": "#D3B8E0",
        "Other": "#CCCCCC",
    }
    return colors.get(section_type, "#CCCCCC")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 TikTok Trending",
    "🎤 Lyric Sections",
    "📊 Cross-Platform",
    "🔍 Analyze Any Song",
    "🤖 Model Prediction",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — TikTok Trending Songs
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Currently Trending on TikTok")

    if st.button("Refresh TikTok Charts", key="refresh_tiktok"):
        st.cache_data.clear()

    with st.spinner("Fetching TikTok trending songs..."):
        tiktok_df = cached_trending_songs()

    if tiktok_df.empty:
        st.warning("Could not fetch TikTok data. Check your RapidAPI key in .env")
        st.info("Showing curated fallback list instead:")
        fallback = pd.DataFrame([
            {"music_title": "Espresso", "music_author": "Sabrina Carpenter", "total_plays": "~5B"},
            {"music_title": "APT.", "music_author": "ROSÉ", "total_plays": "~3B"},
            {"music_title": "Die With A Smile", "music_author": "Lady Gaga", "total_plays": "~2B"},
            {"music_title": "Taste", "music_author": "Sabrina Carpenter", "total_plays": "~1.8B"},
            {"music_title": "Beautiful Things", "music_author": "Benson Boone", "total_plays": "~1.5B"},
        ])
        st.dataframe(fallback, use_container_width=True)
    else:
        # Display top 20
        display_df = tiktok_df.head(20).copy()
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(
            display_df[["music_title", "music_author", "total_plays", "total_likes", "video_count"]],
            use_container_width=True,
        )

        if "total_plays" in tiktok_df.columns and not tiktok_df.empty:
            fig = px.bar(
                tiktok_df.head(15),
                x="music_title",
                y="total_plays",
                color="total_likes",
                title="Top 15 Trending TikTok Songs by Total Plays",
                labels={"music_title": "Song", "total_plays": "Total Plays", "total_likes": "Likes"},
                color_continuous_scale="Reds",
            )
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Lyric Section Breakdown
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Lyric Section Viral Scores")
    st.caption("Based on Genius annotation count (community engagement) + text hook features")

    saved = load_saved_results()

    if saved.empty:
        st.info("No saved analysis found. Use Tab 4 to analyze songs, or run `pipeline.py`")
    else:
        songs_available = saved["song"].unique().tolist()
        selected_song = st.selectbox("Choose a song:", songs_available, key="song_select_tab2")

        song_data = saved[saved["song"] == selected_song].copy()

        col1, col2 = st.columns([2, 1])

        with col1:
            score_col = "cross_platform_score" if "cross_platform_score" in song_data.columns else "combined_viral_score"
            song_data_sorted = song_data.sort_values(score_col, ascending=True)

            fig = px.bar(
                song_data_sorted,
                x=score_col,
                y="section_label",
                orientation="h",
                color="section_type",
                title=f"Viral Score by Section — {selected_song}",
                labels={score_col: "Viral Score", "section_label": "Section"},
                color_discrete_map={
                    "Chorus/Hook": "#FF6B6B",
                    "Verse": "#4ECDC4",
                    "Bridge": "#FFE66D",
                    "Pre-Chorus": "#A8E6CF",
                    "Intro": "#B0C4DE",
                    "Outro": "#D3B8E0",
                    "Other": "#CCCCCC",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Section Scores**")
            for _, row in song_data.sort_values(score_col, ascending=False).iterrows():
                score_pct = int(row[score_col] * 100)
                st.markdown(
                    f"**{row['section_label']}** ({row['section_type']})  \n"
                    f"Score: `{row[score_col]:.3f}` | Ann: `{int(row.get('genius_annotation_count', 0))}`"
                )
                st.progress(score_pct / 100)

        # Show lyrics for top section
        top_section = song_data.loc[song_data[score_col].idxmax()]
        st.markdown("---")
        st.markdown(f"**Most Viral Section: {top_section['section_label']}**")
        st.text(top_section.get("full_lyrics", top_section.get("lyrics_preview", "")))


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Cross-Platform Comparison
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("TikTok vs YouTube Shorts: Same Lyric Sections?")

    saved = load_saved_results()

    if "yt_shorts_matches" not in (saved.columns if not saved.empty else []):
        st.info("No YouTube Shorts data yet. Add your YOUTUBE_API_KEY to .env, then run `pipeline.py`")
        st.markdown("""
        **How to get a free YouTube Data API key:**
        1. Go to [console.cloud.google.com](https://console.cloud.google.com)
        2. Create a new project (e.g., *SUPA-Viral-Lyrics*)
        3. Search **"YouTube Data API v3"** → Enable it
        4. Go to **Credentials → Create Credentials → API Key**
        5. Copy the key and add to `viral_lyrics_analyzer/.env`:
           ```
           YOUTUBE_API_KEY=your_key_here
           ```
        6. Run `python pipeline.py` to fetch YouTube data
        """)
    else:
        songs_available = saved["song"].unique().tolist()
        selected = st.selectbox("Choose a song:", songs_available, key="song_select_tab3")
        song_data = saved[saved["song"] == selected].copy()

        # Scatter: Genius annotations vs YT matches
        fig = px.scatter(
            song_data,
            x="genius_annotation_count",
            y="yt_shorts_matches",
            text="section_label",
            color="section_type",
            size="combined_viral_score",
            title=f"Cross-Platform Signal — {selected}",
            labels={
                "genius_annotation_count": "Genius Annotations (TikTok proxy)",
                "yt_shorts_matches": "YouTube Shorts Title Matches",
            },
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        cols = ["section_label", "section_type", "genius_annotation_count",
                "yt_shorts_matches", "cross_platform_score", "lyrics_preview"]
        available_cols = [c for c in cols if c in song_data.columns]
        st.dataframe(
            song_data.sort_values("cross_platform_score", ascending=False)[available_cols],
            use_container_width=True,
        )


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Analyze Any Song Live
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Analyze Any Song")
    st.caption("Enter a song title and artist to run a live analysis")

    col_a, col_b = st.columns(2)
    with col_a:
        input_title = st.text_input("Song Title", placeholder="e.g. Espresso")
    with col_b:
        input_artist = st.text_input("Artist", placeholder="e.g. Sabrina Carpenter")

    run_yt = st.checkbox(
        "Also check YouTube Shorts",
        value=_has_yt_key(),
        disabled=not _has_yt_key(),
        help="Requires YOUTUBE_API_KEY in .env",
    )

    if st.button("Analyze", key="analyze_btn") and input_title and input_artist:
        with st.spinner(f"Analyzing '{input_title}' by {input_artist}..."):
            result_df = cached_analyze_song(input_title, input_artist)

        if result_df is None or result_df.empty:
            st.error(f"Song not found on Genius: '{input_title}' by {input_artist}. Try a different spelling.")
        else:
            st.success(f"Found {len(result_df)} sections!")

            # YouTube cross-check if requested
            if run_yt and _has_yt_key():
                with st.spinner("Checking YouTube Shorts..."):
                    import lyricsgenius
                    from config import GENIUS_TOKEN
                    genius_client = lyricsgenius.Genius(GENIUS_TOKEN, verbose=False)
                    song = genius_client.search_song(input_title, input_artist)
                    if song:
                        sections = parse_lyrics_into_sections(song.lyrics)
                        yt_df = cross_check_song(input_title, input_artist, sections, max_results=30)
                        if not yt_df.empty:
                            result_df = result_df.merge(
                                yt_df[["section_label", "yt_shorts_matches", "unique_videos_matched"]],
                                on="section_label", how="left"
                            )

            score_col = "combined_viral_score"

            # Bar chart
            fig = px.bar(
                result_df.sort_values(score_col, ascending=False),
                x="section_label",
                y=score_col,
                color="section_type",
                title=f"Viral Score by Section — {input_title}",
                labels={"section_label": "Section", score_col: "Viral Score"},
                color_discrete_map={
                    "Chorus/Hook": "#FF6B6B",
                    "Verse": "#4ECDC4",
                    "Bridge": "#FFE66D",
                    "Pre-Chorus": "#A8E6CF",
                    "Intro": "#B0C4DE",
                    "Outro": "#D3B8E0",
                    "Other": "#CCCCCC",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature breakdown radar for top section
            top = result_df.iloc[0]
            radar_features = ["repeat_score", "rhyme_score", "hook_keyword_score",
                              "short_line_score", "annotation_score"]
            available_radar = [f for f in radar_features if f in top.index]

            if available_radar:
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[float(top[f]) for f in available_radar],
                    theta=[f.replace("_score", "").replace("_", " ").title() for f in available_radar],
                    fill="toself",
                    line_color="#FF6B6B",
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 1])),
                    title=f"Feature Breakdown: {top['section_label']}",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Show top section lyrics
            st.markdown(f"**Most Viral Section: `{top['section_label']}` ({top['section_type']})**")
            st.text(top.get("full_lyrics", top.get("lyrics_preview", "")))

            # Full table
            with st.expander("Full Analysis Table"):
                cols_to_show = ["section_label", "section_type", score_col,
                                "genius_annotation_count", "repeat_score",
                                "rhyme_score", "hook_keyword_score", "lyrics_preview"]
                available = [c for c in cols_to_show if c in result_df.columns]
                st.dataframe(result_df[available], use_container_width=True)

            # Save to CSV
            out = DATA_DIR / f"{input_title.replace(' ', '_')}_{input_artist.replace(' ', '_')}_analysis.csv"
            result_df.to_csv(out, index=False)
            st.caption(f"Saved to {out}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — ML Model Prediction
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🤖 ML Viral Section Predictor")
    st.caption("Trained on 180 songs · 32 features · Leave-one-song-out CV · P@1 = 0.77")

    MODEL_PATH = Path(__file__).parent / "models" / "viral_section_model.pkl"
    RF_MODEL_PATH = Path(__file__).parent / "models" / "viral_section_model_rf.pkl"
    META_PATH = Path(__file__).parent / "models" / "model_metadata.json"

    if not MODEL_PATH.exists():
        st.error(
            "Model not trained yet. Run `python viral_section_model.py` from the "
            "`viral_lyrics_analyzer/` directory first."
        )
    else:
        # ── Load model (cached) ──────────────────────────────────────────────
        @st.cache_resource(show_spinner=False)
        def _load_ml_model():
            import joblib, json
            mdl = joblib.load(MODEL_PATH)
            rf_mdl = joblib.load(RF_MODEL_PATH) if RF_MODEL_PATH.exists() else None
            with open(Path(__file__).parent / "models" / "feature_names.json") as f:
                feat_names = json.load(f)
            meta = {}
            if META_PATH.exists():
                with open(META_PATH) as f:
                    meta = json.load(f)
            return mdl, rf_mdl, feat_names, meta

        ml_model, rf_model, ml_feat_names, ml_meta = _load_ml_model()

        # ── Song input ───────────────────────────────────────────────────────
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            ml_title = st.text_input("Song Title", placeholder="e.g. Espresso", key="ml_title")
        with col_m2:
            ml_artist = st.text_input("Artist", placeholder="e.g. Sabrina Carpenter", key="ml_artist")

        if st.button("Predict Viral Section", key="ml_predict_btn") and ml_title and ml_artist:
            with st.spinner(f"Fetching and scoring '{ml_title}' by {ml_artist}..."):
                from viral_section_model import predict_song
                ml_result = predict_song(ml_title, ml_artist, ml_model, ml_feat_names)

            if ml_result is None or ml_result.empty:
                st.error(f"Song not found on Genius: '{ml_title}' by {ml_artist}.")
            else:
                st.success(f"Analyzed {len(ml_result)} sections!")

                # ── Viral probability bar chart ──────────────────────────────
                fig_prob = px.bar(
                    ml_result.sort_values("viral_prob", ascending=False),
                    x="section_label",
                    y="viral_prob",
                    color="section_type",
                    title=f"ML Viral Probability by Section — {ml_title}",
                    labels={"section_label": "Section", "viral_prob": "Viral Probability"},
                    color_discrete_map={
                        "Chorus/Hook": "#FF6B6B",
                        "Verse": "#4ECDC4",
                        "Bridge": "#FFE66D",
                        "Pre-Chorus": "#A8E6CF",
                        "Intro": "#B0C4DE",
                        "Outro": "#D3B8E0",
                        "Other": "#CCCCCC",
                    },
                )
                fig_prob.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig_prob, use_container_width=True)

                # ── Feature importances (top 15 from RF) ────────────────────
                if rf_model is not None and hasattr(rf_model, "feature_importances_"):
                    importances = rf_model.feature_importances_
                    fi_df = pd.DataFrame({
                        "feature": ml_feat_names,
                        "importance": importances,
                    }).sort_values("importance", ascending=True).tail(15)

                    fig_fi = px.bar(
                        fi_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 15 Feature Importances (RandomForest)",
                        labels={"importance": "Importance", "feature": "Feature"},
                        color="importance",
                        color_continuous_scale="Oranges",
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                elif ml_meta.get("feature_importances"):
                    fi_items = sorted(
                        ml_meta["feature_importances"].items(), key=lambda x: x[1]
                    )[-15:]
                    fi_df = pd.DataFrame(fi_items, columns=["feature", "importance"])
                    fig_fi = px.bar(
                        fi_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 15 Feature Importances",
                        labels={"importance": "Importance", "feature": "Feature"},
                        color="importance",
                        color_continuous_scale="Oranges",
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

                # ── Model vs Genius annotation scatter ───────────────────────
                if "annotation_score" in ml_result.columns or "text_virality_score" in ml_result.columns:
                    x_col = "annotation_score" if "annotation_score" in ml_result.columns else "text_virality_score"
                    fig_scatter = px.scatter(
                        ml_result,
                        x=x_col,
                        y="viral_prob",
                        text="section_label",
                        color="section_type",
                        size="viral_prob",
                        title="ML Score vs Genius Annotation Score",
                        labels={x_col: "Genius/Text Score", "viral_prob": "ML Viral Probability"},
                        color_discrete_map={
                            "Chorus/Hook": "#FF6B6B",
                            "Verse": "#4ECDC4",
                            "Bridge": "#FFE66D",
                            "Pre-Chorus": "#A8E6CF",
                        },
                    )
                    fig_scatter.update_traces(textposition="top center")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # ── Top predicted viral section lyrics ───────────────────────
                top_ml = ml_result.iloc[0]
                st.markdown("---")
                st.markdown(
                    f"**🏆 Predicted Viral Section: `{top_ml['section_label']}` "
                    f"({top_ml['section_type']}) — {top_ml['viral_prob']*100:.1f}% viral probability**"
                )
                st.text(top_ml.get("full_lyrics", top_ml.get("lyrics_preview", "")))

                # ── Full ranked table ────────────────────────────────────────
                with st.expander("Full Ranked Results"):
                    show_cols = ["rank", "section_label", "section_type", "viral_prob",
                                 "repeat_score", "rhyme_score", "hook_keyword_score",
                                 "has_repetition_pattern", "stype_chorus_hook"]
                    avail = [c for c in show_cols if c in ml_result.columns]
                    st.dataframe(ml_result[avail], use_container_width=True)

        # ── Model metadata sidebar ───────────────────────────────────────────
        with st.expander("Model Info"):
            if ml_meta:
                st.json({
                    "model_type": ml_meta.get("model_type", "Unknown"),
                    "training_songs": ml_meta.get("training_songs"),
                    "training_sections": ml_meta.get("training_sections"),
                    "feature_count": ml_meta.get("feature_count"),
                    "cv_results": ml_meta.get("all_cv_results", []),
                })

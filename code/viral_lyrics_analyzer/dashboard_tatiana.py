"""
Streamlit Dashboard — NLP Viral Section Predictor + Tatiana Analysis
=====================================================================
Run with: streamlit run dashboard_tatiana.py

Tabs:
1. 🔥 TikTok Trending     — live trending songs
2. 🎤 Lyric Sections      — Genius + text scores per section
3. 🤖 NLP Detector        — 12-signal NLP algorithm on any song
4. 🎸 Tatiana's Songs     — Cyanite audio features + NLP viral predictions
5. 📊 Model Prediction    — trained LogisticReg_L1 model
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from tiktok_trends import get_trending_songs

st.set_page_config(
    page_title="NLP Viral Section Predictor",
    page_icon="🎵",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;
    }
    .metric-card {
        background: #1e293b; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #6366f1; margin-bottom: 0.5rem;
    }
    .viral-badge {
        background: #ef4444; color: white; border-radius: 20px;
        padding: 2px 10px; font-size: 12px; font-weight: bold;
    }
    .signal-row { padding: 4px 0; border-bottom: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🎵 NLP Viral Section Predictor</h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0;">Which part of a song goes viral on TikTok?</p>
</div>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

SECTION_COLORS = {
    "Chorus/Hook": "#ef4444",
    "Verse": "#22c55e",
    "Bridge": "#f59e0b",
    "Pre-Chorus": "#3b82f6",
    "Intro": "#8b5cf6",
    "Outro": "#ec4899",
    "Other": "#6b7280",
}

SIGNAL_NAMES = {
    "hook_vocabulary":       "Hook Keywords",
    "cross_song_repetition": "Cross-Song Repetition",
    "internal_repetition":   "Internal N-gram Repeat",
    "sentiment_extremity":   "Emotional Extremity",
    "end_rhyme_density":     "End Rhyme Density",
    "line_conciseness":      "Line Conciseness",
    "syllable_consistency":  "Syllable Rhythm",
    "phonetic_openness":     "Phonetic Openness",
    "pronoun_relatability":  "Pronoun Relatability",
    "internal_rhyme":        "Internal Rhyme",
    "energy_markers":        "Energy Markers",
    "position_score":        "Song Position",
}

SIGNAL_WEIGHTS = {
    "hook_vocabulary": 0.15, "cross_song_repetition": 0.13,
    "internal_repetition": 0.12, "sentiment_extremity": 0.10,
    "end_rhyme_density": 0.09, "line_conciseness": 0.09,
    "syllable_consistency": 0.08, "phonetic_openness": 0.07,
    "pronoun_relatability": 0.07, "internal_rhyme": 0.05,
    "energy_markers": 0.03, "position_score": 0.02,
}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_trending_songs():
    return get_trending_songs(count=50)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_nlp_detect(title: str, artist: str):
    from nlp_viral_detector import detect_viral_section
    return detect_viral_section(title, artist)


def analyze_pasted_lyrics(raw_lyrics: str, song_name: str) -> pd.DataFrame:
    """
    Run 12 NLP signals on pasted raw lyrics without any API call.
    Returns a DataFrame sorted by viral_score (same schema as detect_viral_section).
    """
    from nlp_viral_detector import _parse_sections, _section_type, score_section
    sections = _parse_sections(raw_lyrics)
    if not sections:
        return pd.DataFrame()
    labels = list(sections.keys())
    total = len(labels)
    rows = []
    for idx, label in enumerate(labels):
        lines = sections[label]
        stype = _section_type(label)
        scores = score_section(lines, idx, total, raw_lyrics)
        rows.append({
            "section":     label,
            "type":        stype,
            "lines":       len(lines),
            "preview":     " / ".join(lines[:2]),
            "full_lyrics": "\n".join(lines),
            **scores,
        })
    if not rows:
        return pd.DataFrame()
    df = (
        pd.DataFrame(rows)
        .sort_values("viral_score", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def cached_tatiana_data():
    cache = DATA_DIR / "tatiana_sections.csv"
    if cache.exists():
        return pd.read_csv(cache)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def cached_full_lyrics_data():
    cache = DATA_DIR / "tatiana_full_lyrics_sections.csv"
    if cache.exists():
        return pd.read_csv(cache)
    # Compute on demand (no API call needed)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from tatiana_analyzer import analyze_full_lyrics_csv
        return analyze_full_lyrics_csv(verbose=False)
    except Exception:
        return pd.DataFrame()


def load_tatiana_csv():
    # Use resolve() so path works regardless of working directory
    here = Path(__file__).resolve().parent          # viral_lyrics_analyzer/
    root = here.parent                               # SUPA_Spring/
    path = root / "Tatiana_cyanite.csv"

    if not path.exists():
        st.warning(f"Tatiana_cyanite.csv not found at: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        def _clean(t: str) -> str:
            t = str(t).strip().strip("'\"")
            for sfx in ["Official Audio", "Official Video", "Official", "Lyric Video"]:
                t = re.sub(rf"\s*{sfx}\s*$", "", t, flags=re.IGNORECASE).strip()
            return t

        df["song_title_clean"] = df["song_title"].apply(_clean)
        return df
    except Exception as e:
        st.error(f"Error reading Tatiana_cyanite.csv: {e}")
        return pd.DataFrame()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 TikTok Trending",
    "🤖 NLP Detector",
    "🎸 Tatiana's Songs",
    "📊 Model Prediction",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TikTok Trending
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Currently Trending on TikTok")

    if st.button("🔄 Refresh Charts", key="refresh_tt"):
        st.cache_data.clear()

    with st.spinner("Fetching TikTok trends..."):
        tt_df = cached_trending_songs()

    if tt_df.empty:
        st.warning("Could not fetch live TikTok data — showing curated viral songs:")
        fallback = pd.DataFrame([
            {"Song": "Espresso", "Artist": "Sabrina Carpenter", "Est. Plays": "5B+"},
            {"Song": "APT.", "Artist": "ROSÉ", "Est. Plays": "3B+"},
            {"Song": "Die With A Smile", "Artist": "Lady Gaga", "Est. Plays": "2B+"},
            {"Song": "Taste", "Artist": "Sabrina Carpenter", "Est. Plays": "1.8B+"},
            {"Song": "Beautiful Things", "Artist": "Benson Boone", "Est. Plays": "1.5B+"},
            {"Song": "Birds of a Feather", "Artist": "Billie Eilish", "Est. Plays": "1.2B+"},
            {"Song": "Cruel Summer", "Artist": "Taylor Swift", "Est. Plays": "1B+"},
        ])
        st.dataframe(fallback, use_container_width=True)
    else:
        st.dataframe(
            tt_df.head(20)[["music_title", "music_author", "total_plays", "total_likes", "video_count"]],
            use_container_width=True
        )
        if "total_plays" in tt_df.columns:
            fig = px.bar(
                tt_df.head(15), x="music_title", y="total_plays",
                color="total_likes",
                title="Top 15 TikTok Songs by Plays",
                labels={"music_title": "Song", "total_plays": "Total Plays"},
                color_continuous_scale="Reds",
            )
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NLP Detector (12 signals)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 NLP Viral Section Detector")
    st.caption("12 NLP signals: hook vocabulary · cross-song repetition · rhyme · emotion · position · ...")

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NLP Signals", "12")
    c2.metric("Top Signal", "Hook Keywords (15%)")
    c3.metric("Tested P@1", "71.7%")
    c4.metric("AUC-ROC", "0.897")

    st.markdown("---")

    # ── Mode selector ─────────────────────────────────────────────────────────
    nlp_mode = st.radio(
        "Input method:",
        ["📋 Paste Lyrics (any song, no API)", "🔍 Search by Title (Genius API)"],
        horizontal=True,
        key="nlp_mode",
    )

    nlp_df = None   # will be set by whichever mode runs
    nlp_song_label = ""

    # ── MODE A: Paste Lyrics ──────────────────────────────────────────────────
    if nlp_mode == "📋 Paste Lyrics (any song, no API)":
        st.markdown(
            "Paste the full lyrics with **`[Section]` headers** "
            "(e.g. `[Chorus]`, `[Verse 1]`, `[Bridge]`). "
            "Works for any song — no internet needed."
        )

        paste_song_name = st.text_input(
            "Song name (for chart labels)", placeholder="e.g. My Song Title", key="paste_name"
        )
        paste_lyrics = st.text_area(
            "Paste lyrics here",
            height=300,
            placeholder=(
                "[Verse 1]\n"
                "I got my coffee and my sunshine\n"
                "Another perfect morning feeling fine\n\n"
                "[Pre-Chorus]\n"
                "Something about the way you move\n\n"
                "[Chorus]\n"
                "Espresso, baby, yeah you keep me up\n"
                "That's that me espresso\n"
                "Oh oh oh yeah"
            ),
            key="paste_lyrics",
        )

        if st.button("🔍 Detect Viral Section", key="nlp_paste_btn") and paste_lyrics.strip():
            with st.spinner("Running 12 NLP signals on pasted lyrics..."):
                nlp_df = analyze_pasted_lyrics(paste_lyrics, paste_song_name or "Pasted Song")
            nlp_song_label = paste_song_name or "Pasted Song"
            if nlp_df is None or nlp_df.empty:
                st.warning(
                    "No sections found. Make sure your lyrics include headers like "
                    "`[Chorus]`, `[Verse 1]`, `[Bridge]` etc."
                )
                nlp_df = None

    # ── MODE B: Search by Title (Genius API) ─────────────────────────────────
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            nlp_title = st.text_input("Song Title", placeholder="e.g. Espresso", key="nlp_title")
        with col_b:
            nlp_artist = st.text_input("Artist", placeholder="e.g. Sabrina Carpenter", key="nlp_artist")

        if st.button("🔍 Detect Viral Section", key="nlp_btn") and nlp_title and nlp_artist:
            with st.spinner(f"Running 12 NLP signals on '{nlp_title}' by {nlp_artist}..."):
                nlp_df = cached_nlp_detect(nlp_title, nlp_artist)
            nlp_song_label = f"{nlp_title} — {nlp_artist}"
            if nlp_df is None or nlp_df.empty:
                st.error(f"Song '{nlp_title}' not found on Genius. Try different spelling.")
                nlp_df = None

    # ── Shared results rendering ──────────────────────────────────────────────
    if nlp_df is not None and not nlp_df.empty:
        top = nlp_df.iloc[0]
        st.success(f"✅ Analyzed **{len(nlp_df)} sections** · Predicted viral: **{top['section']}**")

        # ── All-section viral score bar chart ─────────────────────────────────
        fig_score = px.bar(
            nlp_df.sort_values("viral_score", ascending=False),
            x="section", y="viral_score",
            color="type",
            text="viral_score",
            title=f"NLP Viral Score by Section — {nlp_song_label}",
            labels={"section": "Section", "viral_score": "NLP Viral Score"},
            color_discrete_map=SECTION_COLORS,
        )
        fig_score.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_score.update_layout(yaxis_range=[0, 1], xaxis_title="")
        st.plotly_chart(fig_score, use_container_width=True)

        # ── Score progress bars for each section ──────────────────────────────
        st.markdown("#### Section Rankings")
        for _, row in nlp_df.sort_values("viral_score", ascending=False).iterrows():
            badge = "🏆 " if row["rank"] == 1 else f"#{int(row['rank'])} "
            col_lbl, col_bar = st.columns([2, 5])
            with col_lbl:
                st.markdown(
                    f"{badge}**{row['section']}**  "
                    f"<span style='color:#94a3b8;font-size:12px'>({row['type']})</span>",
                    unsafe_allow_html=True,
                )
            with col_bar:
                st.progress(float(row["viral_score"]))
                st.caption(f"Score: {row['viral_score']:.3f}")

        st.markdown("---")

        # ── Top section deep-dive ─────────────────────────────────────────────
        st.markdown(
            f"### 🏆 Predicted Viral: `{top['section']}` ({top['type']}) — "
            f"Score: **{top['viral_score']:.3f}**"
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Signal Breakdown (top section):**")
            signal_data = []
            for sig, name in SIGNAL_NAMES.items():
                val = float(top.get(sig, 0))
                w = SIGNAL_WEIGHTS[sig]
                level = "🟢 HIGH" if val >= 0.65 else ("🟡 MED" if val >= 0.35 else "🔴 LOW")
                signal_data.append({
                    "Signal": name,
                    "Score": round(val, 3),
                    "Weight": w,
                    "Contribution": round(w * val, 4),
                    "Level": level,
                })
            sig_df = pd.DataFrame(signal_data).sort_values("Contribution", ascending=False)
            st.dataframe(
                sig_df[["Signal", "Score", "Weight", "Contribution", "Level"]],
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            # Radar chart for top section
            radar_sigs = [
                "hook_vocabulary", "cross_song_repetition", "internal_repetition",
                "sentiment_extremity", "end_rhyme_density", "line_conciseness",
                "syllable_consistency", "phonetic_openness",
            ]
            radar_vals = [float(top.get(s, 0)) for s in radar_sigs]
            radar_labels = [SIGNAL_NAMES[s] for s in radar_sigs]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.3)",
                line=dict(color="#6366f1", width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                title=f"Signal Radar: {top['section']}",
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Signal bars for every section (comparison) ────────────────────────
        with st.expander("📊 Compare all sections — signal heatmap"):
            sig_keys = list(SIGNAL_NAMES.keys())
            sig_readable = list(SIGNAL_NAMES.values())
            heatmap_data = []
            for _, row in nlp_df.sort_values("viral_score", ascending=False).iterrows():
                heatmap_data.append([float(row.get(s, 0)) for s in sig_keys])
            section_labels = nlp_df.sort_values("viral_score", ascending=False)["section"].tolist()

            fig_heat = go.Figure(go.Heatmap(
                z=heatmap_data,
                x=sig_readable,
                y=section_labels,
                colorscale="Blues",
                zmin=0, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in heatmap_data],
                texttemplate="%{text}",
                hoverongaps=False,
            ))
            fig_heat.update_layout(
                title="NLP Signal Heatmap — All Sections",
                xaxis_tickangle=-35,
                height=max(300, len(section_labels) * 50 + 120),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Full ranked table ─────────────────────────────────────────────────
        with st.expander("📋 Full ranked table"):
            sig_cols = list(SIGNAL_NAMES.keys()) + ["viral_score"]
            show_cols = ["rank", "section", "type", "lines"] + sig_cols
            avail = [c for c in show_cols if c in nlp_df.columns]
            st.dataframe(nlp_df[avail], use_container_width=True, hide_index=True)

        # ── Predicted viral lyrics ────────────────────────────────────────────
        st.markdown(f"**Predicted viral lyrics — `{top['section']}`:**")
        lyrics = str(top.get("full_lyrics", top.get("preview", "")))
        for line in [l.strip() for l in lyrics.split("\n") if l.strip()][:10]:
            st.markdown(f"> {line}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tatiana's Songs
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎸 Tatiana DeMaria — Cyanite Audio Analysis + NLP Predictions")
    st.caption("16 songs analyzed with Cyanite.ai audio features + NLP viral section detection")

    tatiana_meta = load_tatiana_csv()
    tatiana_sections = cached_tatiana_data()

    if tatiana_meta.empty:
        st.error("Tatiana_cyanite.csv not found. Make sure it's in the SUPA_Spring folder.")
    else:
        # ── Song overview table ──────────────────────────────────────────────
        st.markdown("#### 🎵 Song Library — Cyanite Audio Features")
        overview_cols = ["song_title_clean", "cyanite_tempo_bpm", "cyanite_valence",
                         "cyanite_arousal", "cyanite_energy_level",
                         "cyanite_mood_tags", "recommended_clip"]
        avail_overview = [c for c in overview_cols if c in tatiana_meta.columns]
        display_meta = tatiana_meta[avail_overview].copy()
        display_meta.columns = [c.replace("cyanite_", "").replace("_", " ").title()
                                  for c in avail_overview]
        st.dataframe(display_meta, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Scatter: valence vs arousal ──────────────────────────────────────
        st.markdown("#### Valence vs Arousal (Cyanite Audio Embedding)")
        if "cyanite_valence" in tatiana_meta.columns and "cyanite_arousal" in tatiana_meta.columns:
            fig_va = px.scatter(
                tatiana_meta,
                x="cyanite_valence", y="cyanite_arousal",
                text="song_title_clean",
                color="cyanite_energy_level",
                size_max=18,
                title="Valence vs Arousal — Tatiana DeMaria Songs",
                labels={"cyanite_valence": "Valence (negative ← → positive)",
                        "cyanite_arousal": "Arousal (calm ← → energetic)"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_va.update_traces(textposition="top center", textfont_size=10)
            fig_va.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_va.add_vline(x=0.3, line_dash="dash", line_color="gray", opacity=0.5)
            fig_va.update_layout(height=450)
            st.plotly_chart(fig_va, use_container_width=True)

        st.markdown("---")

        # ── Full Lyrics NLP Analysis (6 songs, no Genius API) ─────────────────
        st.markdown("#### 🎤 Full Lyrics Viral Section Analysis")
        st.caption("6 songs analyzed from full lyrics CSV · sections auto-detected · no Genius API")

        full_df = cached_full_lyrics_data()

        if full_df.empty:
            st.info("Full lyrics data not yet computed. Generating now...")
            full_df = cached_full_lyrics_data.clear() or cached_full_lyrics_data()

        if not full_df.empty:
            songs_available = sorted(full_df["song"].unique().tolist())
            selected_song = st.selectbox(
                "Select a song", songs_available, key="full_lyrics_song_select"
            )

            song_df = full_df[full_df["song"] == selected_song].copy()
            song_df = song_df.sort_values("viral_score", ascending=False).reset_index(drop=True)
            song_df["rank"] = song_df.index + 1

            col_left, col_right = st.columns([3, 2])

            with col_left:
                # Bar chart: viral score per section
                colors = [
                    "#ef4444" if r == 1 else "#6366f1"
                    for r in song_df["rank"]
                ]
                fig_full = go.Figure(go.Bar(
                    x=song_df["section_label"],
                    y=song_df["viral_score"],
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in song_df["viral_score"]],
                    textposition="outside",
                ))
                fig_full.update_layout(
                    title=f"NLP Viral Score per Section — {selected_song}",
                    xaxis_title="Section",
                    yaxis_title="Viral Score",
                    yaxis_range=[0, min(1.0, song_df["viral_score"].max() * 1.2)],
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig_full, use_container_width=True)

            with col_right:
                top = song_df.iloc[0]
                st.markdown(f"**🏆 Top Viral Section: `{top['section_label']}`**")
                st.markdown(f"Type: `{top['section_type']}` · Score: `{top['viral_score']:.3f}`")
                preview_lines = str(top.get("lyrics_preview", "")).split(" / ")
                for ln in preview_lines[:4]:
                    if ln.strip():
                        st.markdown(f"> *{ln.strip()}*")

                st.markdown("**NLP Signals:**")
                sig_cols = list(SIGNAL_NAMES.keys())
                avail_sigs = [s for s in sig_cols if s in top.index]
                sig_data = pd.DataFrame({
                    "Signal": [SIGNAL_NAMES[s] for s in avail_sigs],
                    "Score": [round(float(top[s]), 3) for s in avail_sigs],
                }).sort_values("Score", ascending=False)
                st.dataframe(sig_data, use_container_width=True, hide_index=True)

            # Ranked table
            with st.expander("All sections ranked"):
                disp_cols = ["rank", "section_label", "section_type", "viral_score", "lyrics_preview"]
                avail_disp = [c for c in disp_cols if c in song_df.columns]
                st.dataframe(song_df[avail_disp], use_container_width=True, hide_index=True)
        else:
            st.warning("Full lyrics CSV not found. Place full_tatiana_demaria_lyrics.csv in the 'Testing lyrics' folder.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML Model Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 ML Viral Section Predictor")
    st.caption("LogisticReg_L1 · 12 NLP features · Leave-one-song-out CV · No Genius API needed")

    MODEL_PATH = Path(__file__).parent / "models" / "viral_section_model.pkl"
    RF_PATH    = Path(__file__).parent / "models" / "viral_section_model_rf.pkl"
    META_PATH  = Path(__file__).parent / "models" / "model_metadata.json"

    # Training data stats
    td_path = DATA_DIR / "training_data.csv"
    if td_path.exists():
        td = pd.read_csv(td_path)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Training Sections", len(td))
        sc2.metric("Training Songs", td["song"].nunique())
        sc3.metric("Sources", td["source"].nunique() if "source" in td.columns else "—")
        sc4.metric("Positive Rate", f"{td['is_viral'].mean():.1%}" if "is_viral" in td.columns else "—")

    st.markdown("---")

    if not MODEL_PATH.exists():
        st.error("Model not trained yet.")
        st.code("cd viral_lyrics_analyzer && python build_training_data.py && python viral_section_model.py", language="bash")
    else:
        @st.cache_resource(show_spinner=False)
        def _load_model():
            import joblib, json
            mdl = joblib.load(MODEL_PATH)
            rf = joblib.load(RF_PATH) if RF_PATH.exists() else None
            with open(Path(__file__).parent / "models" / "feature_names.json") as f:
                feat = json.load(f)
            meta = {}
            if META_PATH.exists():
                with open(META_PATH) as f:
                    meta = json.load(f)
            return mdl, rf, feat, meta

        ml_model, rf_model, ml_feats, ml_meta = _load_model()

        # ── All-model CV results table ─────────────────────────────────────────
        if ml_meta.get("all_cv_results"):
            st.markdown("### 🏆 All Model Results (Leave-One-Song-Out CV)")
            cv_df = pd.DataFrame(ml_meta["all_cv_results"])
            cv_df = cv_df.sort_values("P@1", ascending=False).reset_index(drop=True)
            cv_df["Rank"] = cv_df.index + 1
            cv_df = cv_df.rename(columns={"model": "Model", "n_eval": "Folds"})

            best_model_name = ml_meta.get("model_type", "")

            # Styled bar chart
            fig_cv = px.bar(
                cv_df,
                x="Model", y="P@1",
                color="P@1",
                color_continuous_scale="Teal",
                text="P@1",
                title="P@1 (Precision at Rank 1) — Higher is better",
            )
            fig_cv.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_cv.update_layout(
                yaxis_range=[0, 1], xaxis_title="",
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
            )
            # Highlight best model bar in gold
            best_idx = cv_df[cv_df["Model"] == best_model_name].index.tolist()
            if best_idx:
                colors = ["#2dd4bf"] * len(cv_df)
                colors[best_idx[0]] = "#fbbf24"
                fig_cv.update_traces(marker_color=colors)
            st.plotly_chart(fig_cv, use_container_width=True)

            # Side-by-side metric bars for AUC and MRR
            col_auc, col_mrr = st.columns(2)
            with col_auc:
                fig_auc = px.bar(
                    cv_df, x="Model", y="AUC-ROC",
                    color="AUC-ROC", color_continuous_scale="Blues",
                    text="AUC-ROC", title="AUC-ROC per Model",
                )
                fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_auc.update_layout(
                    yaxis_range=[0, 1], xaxis_title="",
                    coloraxis_showscale=False, xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_auc, use_container_width=True)
            with col_mrr:
                fig_mrr = px.bar(
                    cv_df, x="Model", y="MRR",
                    color="MRR", color_continuous_scale="Purples",
                    text="MRR", title="MRR (Mean Reciprocal Rank) per Model",
                )
                fig_mrr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_mrr.update_layout(
                    yaxis_range=[0, 1], xaxis_title="",
                    coloraxis_showscale=False, xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_mrr, use_container_width=True)

            # Full ranked table
            display_cols = ["Rank", "Model", "P@1", "MRR", "AUC-ROC", "Folds"]
            display_cols = [c for c in display_cols if c in cv_df.columns]
            st.dataframe(
                cv_df[display_cols].style.format({
                    "P@1": "{:.1%}", "MRR": "{:.3f}", "AUC-ROC": "{:.3f}",
                }).highlight_max(subset=["P@1", "MRR", "AUC-ROC"], color="#fbbf2480"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Best model: **{best_model_name}** · "
                f"P@1 {ml_meta['best_model_cv'].get('P@1', 0):.1%} · "
                f"AUC {ml_meta['best_model_cv'].get('AUC-ROC', 0):.3f} · "
                f"Trained on {ml_meta.get('training_songs', '?')} songs, "
                f"{ml_meta.get('training_sections', '?')} sections"
            )


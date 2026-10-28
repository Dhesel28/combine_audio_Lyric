# TikTok Viral Detector

A machine learning system that predicts whether a song or lyric snippet will go viral on TikTok.

## What It Does

- Predicts **song-level virality** on TikTok using audio and lyric features
- Detects the **specific lyric snippet** most likely to blow up
- Finds **similar trending songs** from the current TikTok chart
- Analyzes any artist's catalog against TikTok trending patterns
- Runs as an interactive **Streamlit web app**

## How It Works

1. Pulls songs from a tiered catalog (by popularity)
2. Fetches live **TikTok trending songs** via RapidAPI
3. Enriches songs with **lyrics from Genius API**
4. Trains ML models on lyric + trend features
5. Scores new songs against the trained models

## Project Structure

```
tiktok_trend/
├── app.py                  # Streamlit web app
├── pipeline.py             # Full data pipeline (sample → TikTok → lyrics → merge)
├── train.py                # Model training script
├── nlp_models.py           # NLP feature extraction models
├── song_viral_model.py     # Song-level virality predictor
├── similarity_engine.py    # Finds similar trending songs
├── snippet_detector.py     # Detects viral lyric snippets
├── trending_finder.py      # TikTok trending song fetcher
├── tiktok_fetcher.py       # TikTok API integration
├── genius_fetcher.py       # Genius lyrics API integration
├── tatiana_analyzer.py     # Artist-specific analysis
├── dataset.py              # Dataset utilities
├── config.py               # Config and API key loading
├── models/
│   ├── song_viral_model.pkl    # Trained song virality model
│   ├── viral_line_model.pkl    # Trained lyric snippet model
│   └── similarity_model.pkl    # Trained similarity model
└── data/
    ├── tiktok_trending.csv     # Current TikTok trending songs
    ├── balanced_tiers.csv      # Tiered song catalog
    └── tatiana_results.csv     # Sample artist analysis results
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
GENIUS_ACCESS_TOKEN=your_genius_token
RAPIDAPI_KEY=your_rapidapi_key
```

## Usage

### Run the web app
```bash
streamlit run app.py
```

### Build dataset from scratch
```bash
python pipeline.py --build
```

### Retrain models
```bash
python train.py
```

### Analyze a specific artist
```bash
python tatiana_analyzer.py
```

## Models

| Model | Description | Algorithm |
|---|---|---|
| `song_viral_model.pkl` | Predicts song-level TikTok virality | Random Forest |
| `viral_line_model.pkl` | Detects which lyric line goes viral | Gradient Boosting |
| `similarity_model.pkl` | Finds similar trending songs | TF-IDF + Cosine Similarity |

## Tech Stack

- **Python** — scikit-learn, pandas, numpy
- **NLP** — TF-IDF, sentiment analysis, lyric feature extraction
- **APIs** — Genius (lyrics), RapidAPI TikTok
- **App** — Streamlit

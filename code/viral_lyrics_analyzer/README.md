# Viral Lyrics Section Analyzer

Identifies **which portion of a song's lyrics is trending** on TikTok and YouTube Shorts.

## How It Works

1. **TikTok Trending Songs** → RapidAPI fetches current trending sounds
2. **Lyric Section Analysis** → Genius API fetches lyrics + annotation counts
   - High annotation count = that lyric fragment is most discussed = likely the viral hook
3. **Text Hook Scoring** → Repetition, rhyme, short lines, hook keywords
4. **YouTube Shorts Cross-check** → YouTube Data API searches Shorts for each song, counts lyric matches in titles/descriptions
5. **Combined Score** → Genius signal (50%) + text features (50%) | + YT signal (40%) when available

## Setup

```bash
pip install -r requirements.txt
```

Add your keys to `.env`:
```
GENIUS_ACCESS_TOKEN=your_token   # genius.com/api-clients
RAPIDAPI_KEY=your_key            # already set
YOUTUBE_API_KEY=your_key         # console.cloud.google.com (free)
```

## Usage

### Analyze a single song
```bash
cd viral_lyrics_analyzer
python pipeline.py --song "Espresso" "Sabrina Carpenter"
```

### Analyze trending TikTok songs
```bash
python pipeline.py --songs tiktok
```

### Use curated fallback list (if TikTok API is down)
```bash
python pipeline.py --songs manual
```

### Launch Streamlit dashboard
```bash
streamlit run dashboard.py
```

## Output Files (in `data/`)
| File | Description |
|------|-------------|
| `tiktok_trending.csv` | Trending TikTok songs |
| `lyric_sections_analysis.csv` | Section scores per song |
| `cross_platform_analysis.csv` | Merged TikTok + YouTube data |
| `{Song}_{Artist}_analysis.csv` | Per-song analysis from dashboard |

## Key Signal: Genius Annotations

Genius allows users to annotate specific lyric fragments. A fragment with 10+ annotations
means the community found it notable — this is the strongest proxy for identifying viral hooks
without direct TikTok timestamp data.

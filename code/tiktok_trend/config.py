"""Load API keys and shared config from .env"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Reuse the .env from the sibling viral_lyrics_analyzer project
_env_path = Path(__file__).parent.parent / "viral_lyrics_analyzer" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path(__file__).parent / ".env")

GENIUS_TOKEN               = os.getenv("GENIUS_ACCESS_TOKEN", "")
RAPIDAPI_KEY               = os.getenv("RAPIDAPI_KEY", "")
YOUTUBE_API_KEY            = os.getenv("YOUTUBE_API_KEY", "")
CHARTMETRIC_REFRESH_TOKEN  = os.getenv(
    "CHARTMETRIC_REFRESH_TOKEN",
    "E7gj0Vli9DG8PORYtKmHRuZSeGhcIRt5T0aAFnODG6wbqrIDBsm9OAZUyzuBWmTG",
)
TIKTOK_API_HOST = "tiktok-api23.p.rapidapi.com"

_WORK_DIR  = Path("/Volumes/Dhesel Driv /Work")
_DATA_DIR  = _WORK_DIR / "SUPA_Spring" / "data"
_LOCAL_DIR = Path(__file__).parent / "data"
_LOCAL_DIR.mkdir(exist_ok=True)

# Prefer external drive when mounted; fall back to local data/ directory
DATA_DIR = _WORK_DIR if _WORK_DIR.exists() else _LOCAL_DIR

def _resolve(filename: str) -> Path:
    """Search Work drive subfolders first, then fall back to local data/."""
    if _DATA_DIR.exists():
        # Check root of data dir
        candidate = _DATA_DIR / filename
        if candidate.exists():
            return candidate
        # Search all subfolders (tiktok/, spotify/, lyrics/, training/, models/, evaluation/)
        for subdir in _DATA_DIR.iterdir():
            if subdir.is_dir():
                candidate = subdir / filename
                if candidate.exists():
                    return candidate
    return _LOCAL_DIR / filename

SUMMER_DATASET = _resolve("Summer Dataset.csv")

CATALOG_CSV         = _resolve("catalog_with_lyrics.csv")
TIKTOK_TRENDING_CSV = _resolve("tiktok_trending.csv")
BALANCED_TIERS_CSV  = _resolve("balanced_tiers.csv")
BILLBOARD_CSV       = _resolve("billboard_2026.csv")
CM_TRAINING_CSV     = _resolve("cm_training_songs.csv")

SAMPLE_PER_TIER = 50

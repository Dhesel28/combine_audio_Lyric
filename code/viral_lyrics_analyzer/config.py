"""Load API keys and shared config from .env"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this directory
load_dotenv(Path(__file__).parent / ".env")

GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

TIKTOK_API_HOST = "tiktok-api23.p.rapidapi.com"

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

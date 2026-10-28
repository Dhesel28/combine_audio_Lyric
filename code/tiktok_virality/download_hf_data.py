"""Download Genius Lyrics from HuggingFace (no auth needed)"""

import os
from pathlib import Path

DATA_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Downloading Genius Lyrics from HuggingFace")
print("="*60)
print("\nThis contains ~480k English songs with lyrics and metadata...")
print("No authentication required.\n")

from datasets import load_dataset
import pandas as pd

hf_dir = DATA_DIR / "genius_lyrics_hf"
hf_dir.mkdir(parents=True, exist_ok=True)

parquet_file = hf_dir / "genius_lyrics.parquet"

if parquet_file.exists():
    print(f"File already exists: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Shape: {df.shape}")
else:
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("brunokreiner/genius-lyrics", split="train")

    print("Converting to pandas DataFrame...")
    df = dataset.to_pandas()

    print(f"Saving to {parquet_file}...")
    df.to_parquet(parquet_file, index=False)

    print(f"\n✓ Download complete!")
    print(f"  Shape: {df.shape}")

print(f"\nColumns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head(3))

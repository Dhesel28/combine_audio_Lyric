"""Download TikTok datasets from Kaggle using opendatasets (interactive auth)"""

import os
from pathlib import Path

DATA_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

import opendatasets as od

print("="*60)
print("Downloading TikTok Datasets from Kaggle")
print("="*60)
print("\nYou'll be prompted for your Kaggle credentials once.")
print("Get them from: https://www.kaggle.com/settings -> API -> Create New Token")
print("Enter your Kaggle username and API key when prompted.\n")

# Kaggle dataset URLs
DATASETS = [
    ("https://www.kaggle.com/datasets/sveta151/tiktok-popular-songs-2020", "tiktok-popular-songs-2020"),
    ("https://www.kaggle.com/datasets/yamqwe/tiktok-trending-tracks", "tiktok-trending-tracks"),
    ("https://www.kaggle.com/datasets/thedevastator/popular-tiktok-videos-authors-and-musics", "popular-tiktok-videos-authors-and-musics"),
]

for url, expected_folder in DATASETS:
    print(f"\n{'='*40}")
    print(f"Dataset: {expected_folder}")
    print(f"{'='*40}")

    # Check if already downloaded
    possible_dir = DATA_DIR / expected_folder
    if possible_dir.exists() and any(possible_dir.iterdir()):
        print(f"✓ Already downloaded, skipping...")
        continue

    try:
        od.download(url, data_dir=str(DATA_DIR))
        print(f"✓ Downloaded successfully")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*60)
print("Download Summary")
print("="*60)

# List all downloaded folders
for item in sorted(DATA_DIR.iterdir()):
    if item.is_dir() and not item.name.startswith('.'):
        files = list(item.glob("**/*"))
        file_count = len([f for f in files if f.is_file()])
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
        print(f"📁 {item.name}: {file_count} files, {total_size:.1f} MB")

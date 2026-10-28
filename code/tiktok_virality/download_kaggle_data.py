"""Download TikTok datasets from Kaggle"""

import os
from pathlib import Path

DATA_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle datasets to download
DATASETS = [
    ("sveta151/tiktok-popular-songs-2020", "tiktok_popular_2020"),
    ("yamqwe/tiktok-trending-tracks", "tiktok_trending_tracks"),
    ("thedevastator/popular-tiktok-videos-authors-and-musics", "tiktok_videos_music"),
]

print("="*60)
print("Downloading TikTok Datasets from Kaggle")
print("="*60)

try:
    import kaggle
    print("\n✓ Kaggle API authenticated successfully\n")
except Exception as e:
    print(f"\n✗ Kaggle authentication failed: {e}")
    print("\nPlease ensure your kaggle.json is in ~/.kaggle/")
    print("Download from: https://www.kaggle.com/settings -> API -> Create New Token")
    exit(1)

for dataset_slug, folder_name in DATASETS:
    target_dir = DATA_DIR / folder_name

    # Check if already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✓ {folder_name} already exists, skipping...")
        continue

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {dataset_slug}")
    print(f"To: {target_dir}")

    try:
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=str(target_dir),
            unzip=True
        )
        print(f"✓ Downloaded {folder_name}")

        # List downloaded files
        files = list(target_dir.glob("*"))
        for f in files[:5]:  # Show first 5 files
            print(f"  - {f.name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")

    except Exception as e:
        print(f"✗ Failed to download {folder_name}: {e}")

print("\n" + "="*60)
print("Download Complete!")
print("="*60)

# Summary
print("\nDownloaded datasets:")
for _, folder_name in DATASETS:
    target_dir = DATA_DIR / folder_name
    if target_dir.exists():
        files = list(target_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
        print(f"  📁 {folder_name}: {len(files)} files, {total_size:.1f} MB")

"""
Download datasets for TikTok Song Virality Prediction project.

This script downloads:
1. TikTok song datasets from Kaggle
2. Genius lyrics dataset from HuggingFace
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_kaggle_datasets():
    """Download TikTok datasets from Kaggle using opendatasets."""
    try:
        import opendatasets as od
    except ImportError:
        print("Installing opendatasets...")
        os.system("pip install opendatasets -q")
        import opendatasets as od

    print("\n" + "="*60)
    print("KAGGLE DATASET DOWNLOAD")
    print("="*60)
    print("\nYou'll need your Kaggle credentials.")
    print("Get them from: https://www.kaggle.com/settings -> API -> Create New Token")
    print("\n")

    kaggle_datasets = [
        # TikTok datasets
        ("https://www.kaggle.com/datasets/sveta151/tiktok-popular-songs-2020", "tiktok_popular_2020"),
        ("https://www.kaggle.com/datasets/yamqwe/tiktok-trending-tracks", "tiktok_trending_tracks"),
        ("https://www.kaggle.com/datasets/thedevastator/popular-tiktok-videos-authors-and-musics", "tiktok_videos_music"),
        # Lyrics dataset
        ("https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information", "genius_lyrics_kaggle"),
    ]

    for url, folder_name in kaggle_datasets:
        target_dir = DATA_DIR / folder_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"✓ {folder_name} already exists, skipping...")
            continue

        print(f"\nDownloading: {folder_name}")
        print(f"From: {url}")
        try:
            od.download(url, data_dir=str(DATA_DIR))
            print(f"✓ Downloaded {folder_name}")
        except Exception as e:
            print(f"✗ Failed to download {folder_name}: {e}")


def download_huggingface_datasets():
    """Download Genius lyrics from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    print("\n" + "="*60)
    print("HUGGINGFACE DATASET DOWNLOAD")
    print("="*60)

    hf_dir = DATA_DIR / "genius_lyrics_hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = hf_dir / "genius_lyrics.parquet"
    if parquet_file.exists():
        print(f"✓ HuggingFace Genius lyrics already exists, skipping...")
        return

    print("\nDownloading: brunokreiner/genius-lyrics from HuggingFace")
    print("This contains ~480k English songs with lyrics and metadata...")

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("brunokreiner/genius-lyrics", split="train")

        # Convert to pandas and save as parquet
        df = dataset.to_pandas()
        df.to_parquet(parquet_file, index=False)

        print(f"✓ Downloaded and saved to {parquet_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

    except Exception as e:
        print(f"✗ Failed to download HuggingFace dataset: {e}")


def download_5m_lyrics_sample():
    """Download a sample from the 5 Million Song Lyrics dataset."""
    try:
        import opendatasets as od
    except ImportError:
        os.system("pip install opendatasets -q")
        import opendatasets as od

    print("\n" + "="*60)
    print("5 MILLION SONG LYRICS (Kaggle)")
    print("="*60)

    target_dir = DATA_DIR / "5m_song_lyrics"
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✓ 5M Song Lyrics already exists, skipping...")
        return

    print("\nDownloading: 5 Million Song Lyrics Dataset")
    print("This is a large dataset - may take some time...")

    try:
        od.download(
            "https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset",
            data_dir=str(DATA_DIR)
        )
        print(f"✓ Downloaded 5M Song Lyrics")
    except Exception as e:
        print(f"✗ Failed to download: {e}")


def create_dataset_info():
    """Create a README with dataset information."""
    readme_content = """# Raw Data Directory

This directory contains the raw datasets downloaded for the TikTok Song Virality project.

## Datasets

### TikTok Song Datasets
1. **tiktok_popular_2020/** - TikTok Popular Songs 2020 (Kaggle)
2. **tiktok_trending_tracks/** - 7,000 TikTok Trending Tracks (Kaggle)
3. **tiktok_videos_music/** - Popular TikTok Videos, Authors, and Music (Kaggle)

### Lyrics Datasets
4. **genius_lyrics_hf/** - 480k English songs from HuggingFace (brunokreiner/genius-lyrics)
5. **genius_lyrics_kaggle/** - Genius Song Lyrics with Language Info (Kaggle)
6. **5m_song_lyrics/** - 5 Million Song Lyrics Dataset (Kaggle) [Optional - Large]

## Download Instructions

Run the download script:
```bash
python src/data/download_datasets.py
```

You'll need Kaggle credentials for Kaggle datasets.
Get them from: https://www.kaggle.com/settings -> API -> Create New Token
"""

    readme_path = DATA_DIR / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_content)
    print(f"\n✓ Created {readme_path}")


def main():
    """Main download function."""
    print("\n" + "="*60)
    print("TikTok Song Virality - Dataset Download")
    print("="*60)

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create dataset info
    create_dataset_info()

    # Download HuggingFace dataset first (no auth needed)
    download_huggingface_datasets()

    # Download Kaggle datasets
    print("\n" + "-"*60)
    print("Now downloading from Kaggle (requires authentication)")
    print("-"*60)
    download_kaggle_datasets()

    # Optional: Download 5M lyrics (large dataset)
    print("\n" + "-"*60)
    response = input("Download 5 Million Song Lyrics dataset? (large, ~1GB) [y/N]: ")
    if response.lower() == 'y':
        download_5m_lyrics_sample()

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nData saved to: {DATA_DIR}")

    # List downloaded files
    print("\nDownloaded datasets:")
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            files = list(item.glob("*"))
            print(f"  📁 {item.name}/ ({len(files)} files)")
        else:
            print(f"  📄 {item.name}")


if __name__ == "__main__":
    main()

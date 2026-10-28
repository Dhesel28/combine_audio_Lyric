"""Scrape TikTok trending sounds and videos using unofficial API"""

import asyncio
import pandas as pd
from pathlib import Path
from TikTokApi import TikTokApi

DATA_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/data/raw/tiktok_scraped")
DATA_DIR.mkdir(parents=True, exist_ok=True)

async def get_trending_videos(api, count=50):
    """Get trending videos from TikTok"""
    videos_data = []

    async for video in api.trending.videos(count=count):
        try:
            video_info = {
                'video_id': video.id,
                'description': video.desc,
                'create_time': video.create_time,
                'author_username': video.author.username if video.author else None,
                'author_nickname': video.author.nickname if video.author else None,
                'author_id': video.author.user_id if video.author else None,
                'music_id': video.sound.id if video.sound else None,
                'music_title': video.sound.title if video.sound else None,
                'music_author': video.sound.author if video.sound else None,
                'play_count': video.stats.play_count if video.stats else None,
                'like_count': video.stats.digg_count if video.stats else None,
                'comment_count': video.stats.comment_count if video.stats else None,
                'share_count': video.stats.share_count if video.stats else None,
                'duration': video.duration,
            }
            videos_data.append(video_info)
            print(f"  Fetched video: {video_info['description'][:50]}...")
        except Exception as e:
            print(f"  Error processing video: {e}")
            continue

    return videos_data

async def get_trending_sounds(api, count=30):
    """Get trending sounds/music from TikTok"""
    sounds_data = []

    # Get videos and extract unique sounds
    seen_sounds = set()
    async for video in api.trending.videos(count=count * 3):  # Get more videos to find unique sounds
        try:
            if video.sound and video.sound.id not in seen_sounds:
                sound_info = {
                    'sound_id': video.sound.id,
                    'title': video.sound.title,
                    'author': video.sound.author,
                    'duration': video.sound.duration if hasattr(video.sound, 'duration') else None,
                    'original': video.sound.original if hasattr(video.sound, 'original') else None,
                    'play_url': video.sound.play_url if hasattr(video.sound, 'play_url') else None,
                }
                sounds_data.append(sound_info)
                seen_sounds.add(video.sound.id)
                print(f"  Found sound: {sound_info['title'][:50] if sound_info['title'] else 'Unknown'}...")

                if len(sounds_data) >= count:
                    break
        except Exception as e:
            continue

    return sounds_data

async def main():
    print("=" * 60)
    print("Scraping TikTok Trending Data")
    print("=" * 60)
    print("\nNote: This uses an unofficial API and may not always work.")
    print("TikTok actively blocks scrapers.\n")

    async with TikTokApi() as api:
        # Create sessions (required for the API to work)
        await api.create_sessions(
            num_sessions=1,
            sleep_after=5,
            headless=True,
            browser="webkit",
            suppress_resource_load_types=["image", "media", "font", "stylesheet"]
        )

        # Get trending videos
        print("\n[1/2] Fetching trending videos...")
        try:
            videos = await get_trending_videos(api, count=100)
            print(f"  Got {len(videos)} videos")
        except Exception as e:
            print(f"  Error fetching videos: {e}")
            videos = []

        # Get trending sounds
        print("\n[2/2] Extracting trending sounds...")
        try:
            sounds = await get_trending_sounds(api, count=50)
            print(f"  Got {len(sounds)} unique sounds")
        except Exception as e:
            print(f"  Error fetching sounds: {e}")
            sounds = []

    # Save data
    print("\n" + "=" * 60)
    print("Saving Data")
    print("=" * 60)

    if videos:
        df_videos = pd.DataFrame(videos)
        videos_path = DATA_DIR / "tiktok_trending_videos.csv"
        df_videos.to_csv(videos_path, index=False)
        print(f"\nSaved {len(videos)} videos to: {videos_path}")
        print(f"Columns: {list(df_videos.columns)}")

    if sounds:
        df_sounds = pd.DataFrame(sounds)
        sounds_path = DATA_DIR / "tiktok_trending_sounds.csv"
        df_sounds.to_csv(sounds_path, index=False)
        print(f"\nSaved {len(sounds)} sounds to: {sounds_path}")
        print(f"Columns: {list(df_sounds.columns)}")

    if not videos and not sounds:
        print("\nNo data was scraped. TikTok may have blocked the request.")
        print("Try again later or use the Kaggle datasets instead.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

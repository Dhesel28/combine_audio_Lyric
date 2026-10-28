"""Download TikTok Viral Songs from Spotify Editorial Playlists"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from pathlib import Path

# Spotify credentials
CLIENT_ID = "f6173037527448209afec2264cd3f3da"
CLIENT_SECRET = "b5c9dc06a3b34dd2a6ae21a52db9dd1e"

DATA_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/data/raw/spotify_tiktok")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# TikTok-related Spotify editorial playlists
PLAYLISTS = {
    "tiktok_viral_2026": "57EG9lWmdn7HHofXuQVsow",  # TikTok Music 2026 Viral Trending
    "tiktok_viral_2025": "52Gos9MbHaGxeaiRZXMhAG",  # Viral TikTok Songs 2025
    "tiktok_hits_2024": "0NsRHOJbKs0GbOX2HeQrEk",   # TikTok Hits 2024
}

print("=" * 60)
print("Downloading TikTok Playlists from Spotify")
print("=" * 60)

# Authenticate
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))
print("\n Spotify API authenticated successfully\n")

def get_playlist_tracks(playlist_id, playlist_name):
    """Fetch all tracks from a playlist with audio features"""
    tracks_data = []

    # Get playlist info
    playlist = sp.playlist(playlist_id)
    print(f"\nPlaylist: {playlist['name']}")
    print(f"Total tracks: {playlist['tracks']['total']}")

    # Fetch tracks (handle pagination)
    results = playlist['tracks']
    while results:
        for item in results['items']:
            track = item.get('track')
            if track and track.get('id'):
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': ', '.join([a['name'] for a in track['artists']]),
                    'artist_ids': ', '.join([a['id'] for a in track['artists']]),
                    'album_name': track['album']['name'],
                    'album_id': track['album']['id'],
                    'release_date': track['album'].get('release_date', ''),
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'explicit': track['explicit'],
                    'external_url': track['external_urls'].get('spotify', ''),
                    'preview_url': track.get('preview_url', ''),
                    'playlist_source': playlist_name,
                    'added_at': item.get('added_at', '')
                }
                tracks_data.append(track_info)

        # Next page
        if results['next']:
            results = sp.next(results)
        else:
            results = None

    return tracks_data

# Collect all tracks
all_tracks = []

for name, playlist_id in PLAYLISTS.items():
    try:
        tracks = get_playlist_tracks(playlist_id, name)
        all_tracks.extend(tracks)
        print(f"  Fetched {len(tracks)} tracks")
    except Exception as e:
        print(f"  Error fetching {name}: {e}")

print(f"\nTotal tracks collected: {len(all_tracks)}")

# Note: Audio features API deprecated for new apps (Nov 2024)
# Skipping audio features, saving track metadata only

# Create DataFrame and save
df = pd.DataFrame(all_tracks)

# Remove duplicates (same track in multiple playlists)
df_unique = df.drop_duplicates(subset=['track_id'], keep='first')

# Save to CSV and Parquet
csv_path = DATA_DIR / "spotify_tiktok_viral.csv"
parquet_path = DATA_DIR / "spotify_tiktok_viral.parquet"

df.to_csv(csv_path, index=False)
df_unique.to_parquet(parquet_path, index=False)

print("\n" + "=" * 60)
print("Download Complete!")
print("=" * 60)
print(f"\nSaved {len(df)} tracks (with duplicates): {csv_path}")
print(f"Saved {len(df_unique)} unique tracks: {parquet_path}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample data:")
print(df[['track_name', 'artist_name', 'popularity', 'release_date']].head())

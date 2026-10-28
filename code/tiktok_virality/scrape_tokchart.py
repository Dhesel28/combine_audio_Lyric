"""Scrape trending TikTok songs from Tokchart.com"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path("/Users/dhekha/Desktop/SUPA_Spring/tiktok_virality/data/raw/tokchart")
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def scrape_tokchart():
    """Scrape trending songs from tokchart.com"""
    print("=" * 60)
    print("Scraping Tokchart.com - TikTok Trending Songs")
    print("=" * 60)

    url = "https://tokchart.com/"
    print(f"\nFetching: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error fetching page: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    songs = []

    # Find song entries - tokchart uses various div structures
    # Look for song cards/entries
    song_elements = soup.find_all(['div', 'article'], class_=lambda x: x and ('song' in x.lower() or 'track' in x.lower() or 'chart' in x.lower()))

    if not song_elements:
        # Try finding by looking for common patterns
        song_elements = soup.find_all('a', href=lambda x: x and '/song/' in x if x else False)

    print(f"\nFound {len(song_elements)} potential song elements")

    # Also try to find any table or list structure
    tables = soup.find_all('table')
    lists = soup.find_all(['ul', 'ol'], class_=lambda x: x and ('chart' in str(x).lower() or 'song' in str(x).lower() or 'track' in str(x).lower()) if x else False)

    print(f"Found {len(tables)} tables, {len(lists)} chart lists")

    # Extract text content for analysis
    main_content = soup.find('main') or soup.find('body')
    if main_content:
        # Look for rank patterns (1., 2., etc. or #1, #2)
        text_content = main_content.get_text()
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]

        # Save raw content for manual inspection
        raw_path = DATA_DIR / "tokchart_raw.txt"
        with open(raw_path, 'w') as f:
            f.write(response.text)
        print(f"\nSaved raw HTML to: {raw_path}")

    # Try alternative: look for JSON data in script tags
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and ('songs' in script.string.lower() or 'track' in script.string.lower()):
            json_path = DATA_DIR / "tokchart_scripts.txt"
            with open(json_path, 'a') as f:
                f.write(script.string + "\n\n---\n\n")

    return songs

def scrape_tokchart_api():
    """Try to find and use tokchart's API"""
    print("\n" + "=" * 60)
    print("Trying Tokchart API endpoints")
    print("=" * 60)

    # Common API endpoints to try
    api_endpoints = [
        "https://tokchart.com/api/songs",
        "https://tokchart.com/api/trending",
        "https://tokchart.com/api/charts",
        "https://tokchart.com/api/v1/songs",
        "https://tokchart.com/api/v1/trending",
    ]

    for endpoint in api_endpoints:
        try:
            print(f"\nTrying: {endpoint}")
            response = requests.get(endpoint, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                print(f"  Success! Status: {response.status_code}")
                try:
                    data = response.json()
                    print(f"  Got JSON data: {type(data)}")
                    return data
                except:
                    print(f"  Not JSON, content type: {response.headers.get('content-type')}")
            else:
                print(f"  Status: {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")

    return None

if __name__ == "__main__":
    # Try web scraping
    songs = scrape_tokchart()

    # Try API
    api_data = scrape_tokchart_api()

    if api_data:
        df = pd.DataFrame(api_data if isinstance(api_data, list) else [api_data])
        csv_path = DATA_DIR / "tokchart_trending.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved API data to: {csv_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nCheck the data/raw/tokchart/ folder for any scraped content.")

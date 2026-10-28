"""
Chartmetric API fetcher — TikTok trending sounds + tracks, Instagram sounds.

Authentication (token flow):
    POST /api/token  {"refreshtoken": "..."}  → access token (1-hour TTL)
    Token is cached to disk so we never re-exchange inside a single hour.

Endpoints used:
    GET /api/charts/tiktok/sounds          — TikTok Sound Chart
    GET /api/charts/tiktok/tracks          — TikTok Top-Track Chart
    GET /api/SNS/deepSocial/cm_artist/{id}/instagram  — IG top Reels per artist
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

from config import CHARTMETRIC_REFRESH_TOKEN

BASE_URL     = "https://api.chartmetric.com"
_DATA_DIR    = Path(__file__).parent / "data"
_TOKEN_CACHE = _DATA_DIR / "cm_token_cache.json"


# ── Auth ──────────────────────────────────────────────────────────────────────

def _load_token() -> tuple[str, float]:
    """Return (token, expires_at) from disk cache, or ('', 0) if missing."""
    if _TOKEN_CACHE.exists():
        try:
            d = json.loads(_TOKEN_CACHE.read_text(encoding="utf-8"))
            return d.get("token", ""), float(d.get("expires_at", 0))
        except Exception:
            pass
    return "", 0.0


def _save_token(token: str, expires_at: float) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    _TOKEN_CACHE.write_text(
        json.dumps({"token": token, "expires_at": expires_at}),
        encoding="utf-8",
    )


def get_access_token() -> str:
    """Return a valid Bearer token, refreshing only when the cached one expires."""
    token, expires_at = _load_token()
    if token and time.time() < expires_at - 120:   # 2-min safety buffer
        return token

    print("[Chartmetric] Exchanging refresh token ...")
    resp = requests.post(
        f"{BASE_URL}/api/token",
        json={"refreshtoken": CHARTMETRIC_REFRESH_TOKEN},
        timeout=15,
    )
    resp.raise_for_status()
    data  = resp.json()
    token = (
        data.get("token")
        or data.get("access_token")
        or data.get("accessToken")
        or ""
    )
    if not token:
        raise ValueError(f"Chartmetric: token exchange returned no token — {data}")

    _save_token(token, time.time() + 3600)
    print("[Chartmetric] Token obtained and cached to disk.")
    return token


def _headers() -> dict:
    return {"Authorization": f"Bearer {get_access_token()}"}


# ── Response unwrapper ────────────────────────────────────────────────────────

def _unwrap(body: dict) -> list:
    """
    Chartmetric wraps chart data in various shapes.  Try common paths and
    return the first list we find.
    """
    # Direct list
    if isinstance(body, list):
        return body
    # body.obj  (most chart endpoints)
    obj = body.get("obj", body)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "chart", "items", "rows", "sounds", "tracks"):
            if isinstance(obj.get(key), list):
                return obj[key]
        # any value that is a list
        for v in obj.values():
            if isinstance(v, list) and v:
                return v
    return []


def _get_field(item: dict, *keys: str, default="") -> str | int:
    """Try a list of keys in order; return first non-empty value."""
    for k in keys:
        v = item.get(k)
        if v is not None and v != "":
            return v
    return default


def _recent_monday(weeks_back: int = 0) -> str:
    """Return the most recent Monday (or N weeks earlier) as YYYY-MM-DD.
    Chartmetric chart endpoints require a date; Mondays are safe weekly boundaries."""
    today = datetime.utcnow().date()
    days_since_monday = today.weekday()          # 0=Mon … 6=Sun
    last_monday = today - timedelta(days=days_since_monday + 7 * weeks_back)
    return last_monday.strftime("%Y-%m-%d")


def _get_with_retry(url: str, headers: dict, params: dict, max_retries: int = 3) -> requests.Response:
    """GET with automatic retry on 429 (rate limit) and date fallback on 400."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code == 429:
            wait = 2 ** attempt * 3   # 3s, 6s, 12s
            print(f"  [429] Rate limited — waiting {wait}s …")
            time.sleep(wait)
            continue
        if resp.status_code == 400 and "date" in params:
            # Try one week earlier
            from datetime import datetime as _dt, timedelta as _td
            current = _dt.strptime(params["date"], "%Y-%m-%d").date()
            earlier = (current - _td(days=7)).strftime("%Y-%m-%d")
            if earlier != params["date"]:
                print(f"  [400] Bad date {params['date']} — trying {earlier} …")
                params = {**params, "date": earlier}
                time.sleep(1)
                continue
        return resp
    return resp   # return last response regardless


# ── TikTok sounds chart ───────────────────────────────────────────────────────

def fetch_tiktok_sounds(limit: int = 100, date: str | None = None) -> pd.DataFrame:
    """
    GET /api/charts/tiktok/sounds
    Returns columns: song, artist, cm_track_id, rank, video_count, source
    """
    params: dict = {"limit": limit, "date": date or _recent_monday()}

    try:
        resp = _get_with_retry(
            f"{BASE_URL}/api/charts/tiktok/sounds",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        items = _unwrap(resp.json())
    except Exception as e:
        print(f"[Chartmetric] TikTok sounds error: {e}")
        return pd.DataFrame()

    rows = []
    for it in items:
        cm_track_id = it.get("cm_track", "")
        if isinstance(cm_track_id, dict):
            cm_track_id = cm_track_id.get("id", "")

        rows.append({
            "song":        it.get("song_name") or it.get("tiktok_name_of_sound") or "",
            "artist":      it.get("artists") or it.get("tiktok_sound_creator_name") or "",
            "cm_track_id": cm_track_id,
            "rank":        it.get("rank", 0),
            "video_count": it.get("tiktok_last_7_days_video_count")
                           or it.get("tiktok_last_24_hours_video_count", 0),
            "source":      "cm_tiktok_sounds",
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    # Drop AI-generated sounds with no real song
    if "song" in df.columns:
        df = df[df["song"].str.strip() != ""].reset_index(drop=True)
    print(f"[Chartmetric] TikTok sounds: {len(df)} rows")
    return df


# ── TikTok tracks chart ───────────────────────────────────────────────────────

def fetch_tiktok_tracks(limit: int = 100, date: str | None = None) -> pd.DataFrame:
    """
    GET /api/charts/tiktok/tracks
    Returns columns: song, artist, cm_track_id, rank, video_count, source
    """
    time.sleep(2)   # respect rate limits between consecutive chart calls
    params: dict = {"limit": limit, "date": date or _recent_monday()}

    try:
        resp = _get_with_retry(
            f"{BASE_URL}/api/charts/tiktok/tracks",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        items = _unwrap(resp.json())
    except Exception as e:
        print(f"[Chartmetric] TikTok tracks error: {e}")
        return pd.DataFrame()

    rows = []
    for it in items:
        # artist_names / tiktok_artist_names are lists (may contain None entries)
        artist_list = it.get("tiktok_artist_names") or it.get("artist_names") or []
        artist = ", ".join(str(a) for a in artist_list if a) if isinstance(artist_list, list) else str(artist_list)

        cm_track_id = it.get("cm_track", "") or it.get("id", "")
        if isinstance(cm_track_id, dict):
            cm_track_id = cm_track_id.get("id", "")

        rows.append({
            "song":        it.get("name") or "",
            "artist":      artist,
            "cm_track_id": cm_track_id,
            "rank":        it.get("rank", 0),
            "video_count": it.get("posts", 0),   # 'posts' = total TikTok videos using track
            "source":      "cm_tiktok_tracks",
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if "song" in df.columns:
        df = df[df["song"].str.strip() != ""].reset_index(drop=True)
    print(f"[Chartmetric] TikTok tracks: {len(df)} rows")
    return df


# ── Instagram sounds (per artist) ────────────────────────────────────────────

def fetch_ig_sounds_for_artist(cm_artist_id: int | str, limit: int = 50) -> pd.DataFrame:
    """
    GET /api/SNS/deepSocial/cm_artist/{id}/instagram
    Returns top Reels/posts for a specific Chartmetric artist ID.

    To look up a cm_artist_id:
        GET /api/search/artist?q=<artist name>  → use the 'id' field.
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/api/SNS/deepSocial/cm_artist/{cm_artist_id}/instagram",
            headers=_headers(),
            params={"limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
        items = _unwrap(resp.json())
    except Exception as e:
        print(f"[Chartmetric] IG sounds ({cm_artist_id}) error: {e}")
        return pd.DataFrame()

    rows = []
    for it in items:
        rows.append({
            "post_id":  _get_field(it, "id"),
            "type":     _get_field(it, "type", "media_type"),
            "url":      _get_field(it, "url", "permalink"),
            "plays":    _get_field(it, "play_count", "video_view_count", default=0),
            "likes":    _get_field(it, "like_count", default=0),
            "comments": _get_field(it, "comments_count", "comment_count", default=0),
            "sound":    _get_field(it, "audio_title", "music_title"),
            "source":   "cm_ig_sounds",
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"[Chartmetric] IG sounds for artist {cm_artist_id}: {len(df)} rows")
    return df


def lookup_artist_id(artist_name: str) -> int | None:
    """
    Search Chartmetric for an artist and return their cm_artist_id.
    Used before calling fetch_ig_sounds_for_artist().
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/api/search/artist",
            headers=_headers(),
            params={"q": artist_name, "limit": 5},
            timeout=15,
        )
        resp.raise_for_status()
        items = _unwrap(resp.json())
        if items:
            return items[0].get("id") or items[0].get("cm_artist", {}).get("id")
    except Exception as e:
        print(f"[Chartmetric] Artist lookup error: {e}")
    return None


# ── Combined trending fetch ───────────────────────────────────────────────────

def get_chartmetric_trending(limit: int = 100) -> dict[str, pd.DataFrame]:
    """
    Fetch TikTok sounds + tracks charts and save each to data/cm_*.csv.
    Returns {"tiktok_sounds": df, "tiktok_tracks": df}.
    """
    _DATA_DIR.mkdir(exist_ok=True)
    print("[Chartmetric] Fetching trending charts ...")

    results = {
        "tiktok_sounds": fetch_tiktok_sounds(limit=limit),
        "tiktok_tracks": fetch_tiktok_tracks(limit=limit),
    }

    for name, df in results.items():
        if not df.empty:
            path = _DATA_DIR / f"cm_{name}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved {len(df)} rows → {path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = get_chartmetric_trending()
    for name, df in data.items():
        print(f"\n=== {name} ({len(df)} rows) ===")
        if not df.empty:
            print(df[["rank", "song", "artist", "video_count"]].head(15).to_string(index=False))

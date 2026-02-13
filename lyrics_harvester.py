#!/usr/bin/env python3
"""
🎵  Lyrics Harvester
====================
Discovers and saves lyrics for:
  • Top 50 trending Hindi songs
  • Top 50 trending Haryanvi songs
  • Top 50 trending Punjabi songs

Each song's lyrics are saved as a separate .txt file under lyrics/<language>/.

Usage:
    python lyrics_harvester.py
    python lyrics_harvester.py --count 25
    python lyrics_harvester.py --languages hindi punjabi
    python lyrics_harvester.py --output my_lyrics --count 10

Dependencies (auto-installed on first run):
    requests, beautifulsoup4, ytmusicapi
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import html as html_mod
import random
import logging
import unicodedata
import urllib.parse
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime
from typing import Optional

# ──────────────────────────────────────────────────────────────
# Auto-install missing packages
# ──────────────────────────────────────────────────────────────

def _ensure_packages():
    needed = [("requests", "requests"), ("bs4", "beautifulsoup4"), ("ytmusicapi", "ytmusicapi")]
    missing = [pkg for mod, pkg in needed if not _can_import(mod)]
    if missing:
        import subprocess
        print(f"📦  Installing: {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("   ✓ installed.\n")


def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


_ensure_packages()

import requests
from bs4 import BeautifulSoup

_YT_AVAILABLE = False
_yt_instance = None
try:
    from ytmusicapi import YTMusic
    _yt_instance = YTMusic()
    _YT_AVAILABLE = True
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = Path("lyrics")
DEFAULT_COUNT = 50
TIMEOUT = 20
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger("harvester")

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8,pa;q=0.7",
})

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _get(url: str, params: dict | None = None, **kw) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT, **kw)
            r.raise_for_status()
            return r
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


def _safe_json(resp: requests.Response | None):
    if resp is None:
        return None
    try:
        text = resp.text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        if text.startswith("<!"):
            return None
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _delay(lo: float = 0.8, hi: float = 2.0):
    time.sleep(random.uniform(lo, hi))


def _sanitize(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"\s+", " ", text).strip().replace(" ", "_")
    return text[:max_len] or "unnamed"


def _clean_lyrics(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    out: list[str] = []
    prev_blank = False
    for line in lines:
        line = line.rstrip()
        blank = not line
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    return "\n".join(out).strip()


def _normalize(text: str) -> str:
    """Lower-case, strip accents, remove non-alnum for fuzzy comparison."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text)


def _strip_meta(title: str) -> str:
    """Remove parenthetical metadata like (From 'X'), (feat. Y), (Lofi Mix)."""
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip()
    cleaned = re.sub(r"\s*feat\.?\s.*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or title


def _title_match(query_title: str, result_title: str, threshold: float = 0.50) -> bool:
    """Check if a result title is a reasonable match for the query title."""
    q = _normalize(_strip_meta(query_title))
    r = _normalize(_strip_meta(result_title))
    if not q or not r:
        return False
    # Exact match
    if q == r:
        return True
    # Substring match only when lengths are close (within 1.5x)
    if q in r and len(r) <= len(q) * 1.5:
        return True
    if r in q and len(q) <= len(r) * 1.5:
        return True
    # Fuzzy ratio
    return SequenceMatcher(None, q, r).ratio() >= threshold


def _artist_match(query_artist: str, result_artist: str) -> bool:
    """Check if any word in the query artist appears in the result artist (or vice versa)."""
    q_parts = set(_normalize(query_artist).split()) if query_artist else set()
    r_parts = set(_normalize(result_artist).split()) if result_artist else set()
    # Remove very short/common words
    q_parts = {p for p in q_parts if len(p) > 2}
    r_parts = {p for p in r_parts if len(p) > 2}
    if not q_parts or not r_parts:
        return True  # Can't verify, allow
    # Normalised full-string comparison
    q_full = _normalize(query_artist)
    r_full = _normalize(result_artist)
    if q_full in r_full or r_full in q_full:
        return True
    # Any significant word overlap
    return bool(q_parts & r_parts)


# ──────────────────────────────────────────────────────────────
# Song Discovery  —  YouTube Music
# ──────────────────────────────────────────────────────────────

_YT_QUERIES: dict[str, list[str]] = {
    "hindi": [
        "Top Hindi Songs 2025 playlist",
        "Top Hindi Songs 2026 playlist",
        "Bollywood trending songs playlist",
        "Best Hindi songs latest hits",
        "New Hindi Bollywood songs",
    ],
    "haryanvi": [
        "Top Haryanvi Songs 2025 playlist",
        "Top Haryanvi Songs 2026 playlist",
        "Haryanvi trending hits playlist",
        "Haryanvi DJ songs latest",
        "New Haryanvi songs hits",
        "Haryanvi songs original playlist",
        "Best Haryanvi songs Amanraj Gill",
        "Haryanvi songs Sapna Choudhary",
        "Haryanvi songs Pranjal Dahiya",
        "Haryanvi songs Renuka Panwar",
    ],
    "punjabi": [
        "Top Punjabi Songs 2025 playlist",
        "Top Punjabi Songs 2026 playlist",
        "Punjabi trending songs playlist",
        "Best Punjabi songs latest hits",
        "New Punjabi songs",
    ],
}


def _extract_artist(track: dict) -> str:
    artists = track.get("artists") or []
    return (
        ", ".join(a.get("name", "") for a in artists[:2] if a.get("name"))
        or "Unknown"
    )


def _discover_yt(language: str, needed: int) -> list[dict]:
    if not _YT_AVAILABLE or _yt_instance is None:
        return []
    yt = _yt_instance
    songs: list[dict] = []
    queries = _YT_QUERIES.get(language, [f"top {language} songs 2026"])

    # Phase 1: playlist search
    for q in queries:
        if len(songs) >= needed:
            break
        try:
            playlists = yt.search(q, filter="playlists", limit=5)
            for pl in playlists[:3]:
                if len(songs) >= needed:
                    break
                bid = pl.get("browseId")
                if not bid:
                    continue
                try:
                    data = yt.get_playlist(bid, limit=needed)
                    for t in data.get("tracks", []):
                        if t.get("title"):
                            songs.append({
                                "title": t["title"],
                                "artist": _extract_artist(t),
                                "video_id": t.get("videoId"),
                            })
                except Exception:
                    continue
            _delay(0.4, 1.0)
        except Exception:
            continue

    # Phase 2: direct song search
    if len(songs) < needed:
        for q in queries:
            if len(songs) >= needed:
                break
            try:
                results = yt.search(q, filter="songs", limit=min(needed, 30))
                for t in results:
                    if t.get("title"):
                        songs.append({
                            "title": t["title"],
                            "artist": _extract_artist(t),
                            "video_id": t.get("videoId"),
                        })
                _delay(0.4, 0.8)
            except Exception:
                continue
    return songs


# ──────────────────────────────────────────────────────────────
# Song Discovery  —  JioSaavn
# ──────────────────────────────────────────────────────────────

_SAAVN = "https://www.jiosaavn.com/api.php"


def _parse_saavn_song(item: dict) -> dict | None:
    title = html_mod.unescape(item.get("title", item.get("song", ""))).strip()
    artist = html_mod.unescape(
        item.get("primary_artists", item.get("music", "Unknown"))
    ).strip()
    if not title:
        return None
    return {"title": title, "artist": artist, "jiosaavn_id": item.get("id")}


def _discover_saavn(language: str, needed: int) -> list[dict]:
    songs: list[dict] = []
    lang = language.lower()

    # trending
    try:
        resp = _get(_SAAVN, params={
            "__call": "content.getTrending", "api_version": "4",
            "_format": "json", "_marker": "0", "ctx": "wap6dot0",
            "entity_type": "song", "entity_language": lang,
        })
        data = _safe_json(resp)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and (s := _parse_saavn_song(item)):
                    songs.append(s)
        elif isinstance(data, dict):
            for item in data.get("results", data.get("data", [])):
                if isinstance(item, dict) and (s := _parse_saavn_song(item)):
                    songs.append(s)
        _delay(0.4, 1.0)
    except Exception:
        pass

    # search
    if len(songs) < needed:
        base_terms = [
            f"trending {language} songs",
            f"latest {language} hits 2025",
            f"top {language} songs 2026",
            f"best {language} songs",
        ]
        # Extra queries for Haryanvi (niche genre)
        if language == "haryanvi":
            base_terms.extend([
                "haryanvi songs Amanraj Gill",
                "haryanvi songs Sapna Choudhary",
                "haryanvi songs Pranjal Dahiya",
                "haryanvi songs Renuka Panwar",
                "haryanvi songs Ruchika Jangid",
                "haryanvi songs Masoom Sharma",
            ])
        for term in base_terms:
            if len(songs) >= needed:
                break
            try:
                resp = _get(_SAAVN, params={
                    "__call": "search.getResults", "p": "1", "q": term,
                    "_format": "json", "_marker": "0", "ctx": "wap6dot0",
                    "n": str(min(needed, 50)),
                })
                data = _safe_json(resp)
                if isinstance(data, dict):
                    for item in data.get("results", []):
                        if (s := _parse_saavn_song(item)):
                            songs.append(s)
                _delay(0.4, 1.0)
            except Exception:
                continue
    return songs


# ──────────────────────────────────────────────────────────────
# Song Discovery  —  Genius (extra fallback)
# ──────────────────────────────────────────────────────────────


def _discover_genius(language: str, needed: int) -> list[dict]:
    songs: list[dict] = []
    for q in [f"trending {language} song", f"latest {language} hit", f"popular {language} song 2025"]:
        if len(songs) >= needed:
            break
        try:
            resp = _get("https://genius.com/api/search/multi", params={"q": q})
            data = _safe_json(resp)
            if not data:
                continue
            for section in data.get("response", {}).get("sections", []):
                for hit in section.get("hits", []):
                    if hit.get("type") == "song":
                        r = hit.get("result", {})
                        title = r.get("title", "").strip()
                        artist = r.get("primary_artist", {}).get("name", "Unknown")
                        if title:
                            songs.append({"title": title, "artist": artist, "genius_url": r.get("url")})
            _delay(0.5, 1.2)
        except Exception:
            continue
    return songs


# ──────────────────────────────────────────────────────────────
# Master Song Discovery
# ──────────────────────────────────────────────────────────────



_JUNK_TITLE_PATTERNS = re.compile(
    r"(?i)(jukebox|video jukebox|compilation|mashup|nonstop|non-stop|medley|"
    r"music video|official video|lyric video|lyrical|full album|back to back|"
    r"top hits|best of \d{4}|superhit|dj mix|video song|audio jukebox)"
)


def _clean_song_title(title: str) -> str:
    """Strip YouTube metadata noise from a song title."""
    # Remove everything after first pipe or bullet
    title = re.split(r"\s*[|•]\s*", title)[0].strip()
    # Remove trailing year references like "2022", "2025"
    title = re.sub(r"\s+\d{4}$", "", title).strip()
    # Remove "(Official Video)" etc
    title = re.sub(r"\s*\((?:Official|Full|HD)\s+(?:Video|Audio|Song)\)", "", title, flags=re.I).strip()
    return title or title


def _is_valid_song(song: dict) -> bool:
    """Reject jukebox compilations, overly long titles, and non-song entries."""
    title = song.get("title", "")
    # Too long = probably a YouTube description, not a song
    if len(title) > 120:
        return False
    # Contains jukebox / compilation markers
    if _JUNK_TITLE_PATTERNS.search(title):
        return False
    # Too many pipe chars = YouTube metadata
    if title.count("|") >= 2:
        return False
    return True

def _dedup(songs: list[dict]) -> list[dict]:
    """Deduplicate songs, preferring original versions over Lofi/remix variants."""
    seen: dict[str, dict] = {}
    for s in songs:
        title = s.get("title", "")
        # Dedup key uses stripped title so Lofi versions merge with originals
        key = _normalize(f"{_strip_meta(title)}{s.get('artist', '')}")
        if not key:
            continue
        if key not in seen:
            seen[key] = s
        else:
            # Prefer non-remix/non-Lofi versions
            existing = seen[key].get("title", "")
            is_remix = bool(re.search(r"(?i)(lofi|slowed|reverb|remix|8d)", title))
            existing_remix = bool(re.search(r"(?i)(lofi|slowed|reverb|remix|8d)", existing))
            if existing_remix and not is_remix:
                seen[key] = s  # Replace with original version
    return list(seen.values())


def discover_songs(language: str, count: int = 50) -> list[dict]:
    pool: list[dict] = []

    if _YT_AVAILABLE:
        try:
            yt_songs = _discover_yt(language, count + 20)
            log.info(f"  YouTube Music → {len(yt_songs)} candidates")
            pool.extend(yt_songs)
        except Exception as exc:
            log.warning(f"  YouTube Music error: {exc}")

    try:
        js_songs = _discover_saavn(language, count + 20)
        log.info(f"  JioSaavn      → {len(js_songs)} candidates")
        pool.extend(js_songs)
    except Exception as exc:
        log.warning(f"  JioSaavn error: {exc}")

    remaining = count - len(_dedup(pool))
    if remaining > 0:
        try:
            g_songs = _discover_genius(language, remaining + 10)
            log.info(f"  Genius        → {len(g_songs)} candidates")
            pool.extend(g_songs)
        except Exception as exc:
            log.warning(f"  Genius error: {exc}")

    # Filter out junk entries (jukeboxes, compilations, YouTube metadata)
    pool = [s for s in pool if _is_valid_song(s)]
    # Clean up YouTube metadata from song titles
    for s in pool:
        s["title"] = _clean_song_title(s["title"])
    result = _dedup(pool)
    if not result:
        log.warning(f"  ⚠  No songs found for '{language}' from any source!")
    else:
        log.info(f"  Unique songs  → {len(result)} (need {count})")
    return result[:count]


# ══════════════════════════════════════════════════════════════
# Lyrics Fetching
# ══════════════════════════════════════════════════════════════


def _lyrics_jiosaavn(title: str, artist: str, song_id: str | None = None) -> Optional[str]:
    """Fetch lyrics from JioSaavn.  Tries: exact ID → search by title+artist → title only."""
    ids_to_try: list[str] = []
    if song_id:
        ids_to_try.append(song_id)

    # Search queries in priority order
    search_queries = [
        f"{_strip_meta(title)} {artist}",
        _strip_meta(title),
    ]

    for q in search_queries:
        if ids_to_try:
            break
        try:
            resp = _get(_SAAVN, params={
                "__call": "search.getResults", "p": "1", "q": q,
                "_format": "json", "_marker": "0", "ctx": "wap6dot0", "n": "5",
            })
            data = _safe_json(resp)
            if isinstance(data, dict):
                for item in data.get("results", []):
                    if isinstance(item, dict) and _title_match(title, item.get("title", item.get("song", ""))):
                        ids_to_try.append(item["id"])
                        break
            _delay(0.2, 0.5)
        except Exception:
            continue

    # Fetch lyrics for each candidate ID
    for sid in ids_to_try[:2]:
        try:
            resp = _get(_SAAVN, params={
                "__call": "lyrics.getLyrics", "lyrics_id": sid,
                "api_version": "4", "_format": "json", "_marker": "0", "ctx": "wap6dot0",
            })
            data = _safe_json(resp)
            if isinstance(data, dict) and data.get("lyrics"):
                soup = BeautifulSoup(data["lyrics"], "html.parser")
                text = soup.get_text(separator="\n")
                if text and len(text.strip()) > 40:
                    return text
        except Exception:
            continue
    return None


def _lyrics_ytmusic(title: str, artist: str, video_id: str | None = None) -> Optional[str]:
    """Fetch lyrics from YouTube Music."""
    if not _YT_AVAILABLE or _yt_instance is None:
        return None
    yt = _yt_instance

    # Find video ID if not provided
    video_ids: list[str] = []
    if video_id:
        video_ids.append(video_id)

    if not video_ids:
        for q in [f"{_strip_meta(title)} {artist}", _strip_meta(title)]:
            try:
                results = yt.search(q, filter="songs", limit=3)
                for r in results:
                    if r.get("videoId") and _title_match(title, r.get("title", "")):
                        video_ids.append(r["videoId"])
                        break
                if video_ids:
                    break
            except Exception:
                continue

    # Try getting lyrics for each video ID
    for vid in video_ids[:2]:
        try:
            watch = yt.get_watch_playlist(vid)
            browse_id = watch.get("lyrics")
            if browse_id:
                data = yt.get_lyrics(browse_id)
                if data and data.get("lyrics") and len(data["lyrics"].strip()) > 40:
                    return data["lyrics"]
        except Exception:
            continue
    return None


def _strip_genius_metadata(text: str) -> str:
    """Remove Genius-specific metadata lines from scraped lyrics."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip contributor counts, read more, translation headers
        if re.match(r'^\d+\s+Contributor', stripped):
            continue
        if re.match(r'^Translations?', stripped, re.IGNORECASE):
            continue
        if 'Read More' in stripped and len(stripped) > 200:
            continue
        if re.match(r'^.*Lyrics$', stripped) and len(stripped) < 100:
            # e.g. "Sprinter Lyrics" — skip song title header
            continue
        if stripped.startswith('You might also like'):
            continue
        if re.match(r'^Embed$', stripped):
            continue
        if re.match(r'^\d+$', stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _lyrics_genius(title: str, artist: str, url: str | None = None) -> Optional[str]:
    """Scrape lyrics from Genius.com with title + artist validation."""
    try:
        if not url:
            for q in [f"{_strip_meta(title)} {artist}", _strip_meta(title)]:
                resp = _get("https://genius.com/api/search/multi", params={"q": q})
                data = _safe_json(resp)
                if not data:
                    continue
                for section in data.get("response", {}).get("sections", []):
                    for hit in section.get("hits", []):
                        if hit.get("type") != "song":
                            continue
                        res = hit.get("result", {})
                        res_title = res.get("title", "")
                        res_artist = res.get("primary_artist", {}).get("name", "")
                        # VALIDATE: must match BOTH title AND artist
                        if _title_match(title, res_title) and _artist_match(artist, res_artist):
                            url = res.get("url")
                            break
                    if url:
                        break
                if url:
                    break
                _delay(0.3, 0.6)

        if not url:
            return None

        _delay(0.3, 0.8)
        resp = _get(url)
        if not resp:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if not divs:
            return None

        parts: list[str] = []
        for div in divs:
            for br in div.find_all("br"):
                br.replace_with("\n")
            parts.append(div.get_text())

        text = _strip_genius_metadata("\n".join(parts))

        # Sanity: reject prose (avg line > 200 chars)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None
        avg_line = sum(len(l) for l in lines) / len(lines)
        if avg_line > 200:
            return None

        return text
    except Exception:
        pass
    return None


def _lyrics_ovh(title: str, artist: str) -> Optional[str]:
    """lyrics.ovh free API."""
    try:
        clean_title = _strip_meta(title)
        url = (
            f"https://api.lyrics.ovh/v1/"
            f"{urllib.parse.quote(artist)}/{urllib.parse.quote(clean_title)}"
        )
        resp = _get(url)
        if resp and resp.status_code == 200:
            data = resp.json()
            text = data.get("lyrics", "")
            if text and len(text.strip()) > 40:
                return text
    except Exception:
        pass
    return None


def _lyrics_lrclib(title: str, artist: str) -> Optional[str]:
    """Fetch lyrics from lrclib.net (free, no auth needed)."""
    clean = _strip_meta(title)
    for q in [f"{clean} {artist}", clean]:
        try:
            resp = _get(
                "https://lrclib.net/api/search", params={"q": q},
                headers={"Lrclib-Client": "LyricsHarvester/1.0"},
            )
            if not resp:
                continue
            data = resp.json()
            if not isinstance(data, list):
                continue
            for item in data:
                if (_title_match(title, item.get("trackName", "")) and
                        _artist_match(artist, item.get("artistName", ""))):
                    plain = item.get("plainLyrics", "")
                    if plain and len(plain.strip()) > 40:
                        return plain
                    synced = item.get("syncedLyrics", "")
                    if synced and len(synced.strip()) > 40:
                        lines = [re.sub(r'^\[\d+:\d+\.\d+\]\s*', '', l) for l in synced.split("\n")]
                        return "\n".join(lines)
        except Exception:
            continue
    return None


def _scrape_lyrics_page(url: str, artist: str = "") -> Optional[str]:
    """Generic lyrics page scraper with artist + junk validation."""
    try:
        resp = _get(url)
        if not resp:
            return None

        # --- Artist validation: check title/h1/h2 tags (not full sidebar) ---
        name_words: set[str] = set()
        if artist:
            soup_pre = BeautifulSoup(resp.text[:8000], "html.parser")
            check_parts: list[str] = []
            for htag in soup_pre.find_all(["title", "h1", "h2"]):
                check_parts.append(htag.get_text().lower())
            check_text = " ".join(check_parts)
            for part in re.split(r'[,&/]+', artist.lower()):
                for word in part.split():
                    if len(word) > 2:
                        name_words.add(word)
            if name_words and check_text.strip() and not any(w in check_text for w in name_words):
                return None  # headings don't mention the artist at all

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "ins"]):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        selectors = [
            {"attrs": {"data-lyrics-container": "true"}},
            {"class_": re.compile(r"lyrics?[-_]?(body|content|text)", re.I)},
            {"class_": "entry-content"},
            {"id": re.compile(r"^lyrics?$", re.I)},
            {"class_": "song-lyrics"},
            {"class_": "lyrics"},
        ]
        for sel in selectors:
            divs = soup.find_all(["div", "span", "pre", "p"], **sel)
            if divs:
                text = "\n".join(d.get_text(separator="\n") for d in divs)
                text = _clean_lyrics(text)
                # --- Junk / non-lyrics rejection ---
                first_200 = text[:200].lower()
                junk = ["disclaimer", "privacy policy", "cookie",
                        "terms of service", "contact us", "about us",
                        "copyright notice", "all rights reserved"]
                if any(m in first_200 for m in junk):
                    return None
                # --- Wrong-artist rejection via metadata in scraped text ---
                if name_words:
                    header = text[:500].lower()
                    if any(m in header for m in ["sung by", "singer", "artist:", "vocals"]):
                        if not any(w in header for w in name_words):
                            continue  # metadata shows different artist
                lines = [l for l in text.split("\n") if l.strip()]
                if len(text.strip()) > 60 and len(lines) >= 3:
                    avg = sum(len(l) for l in lines) / len(lines)
                    if avg < 200:
                        return text
        return None
    except Exception:
        return None


_INDIAN_LYRIC_SITES = [
    ("lyricsmint.com", "https://www.lyricsmint.com/", "s"),
    ("lyricsted.com",  "https://www.lyricsted.com/",  "s"),
    ("lyricsbell.com", "https://www.lyricsbell.com/", "s"),
]


def _lyrics_indian_sites(title: str, artist: str) -> Optional[str]:
    """Search popular Indian lyrics sites and scrape."""
    clean = _strip_meta(title)
    for site_name, base_url, param in _INDIAN_LYRIC_SITES:
        for q in [f"{clean} {artist}", clean]:
            try:
                resp = _get(base_url, params={param: q})
                if not resp:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                target_url = None
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "")
                    text = a.get_text().strip()
                    if (site_name in href and len(text) > 3 and
                            _title_match(title, text) and href != base_url):
                        target_url = href
                        break
                if not target_url:
                    continue
                _delay(0.3, 0.6)
                text = _scrape_lyrics_page(target_url, artist=artist)
                if text:
                    return text
            except Exception:
                continue
    return None


def _lyrics_web_search(title: str, artist: str) -> Optional[str]:
    """Search DuckDuckGo Lite for lyrics pages, then scrape them."""
    clean = _strip_meta(title)
    queries = [f"{clean} {artist} lyrics", f"{clean} {artist} song lyrics"]
    lyrics_domains = {
        "lyricsmint.com", "lyricsted.com", "lyricsbell.com",
        "genius.com", "azlyrics.com", "musixmatch.com",
        "lyrics.com", "songlyrics.com",
    }
    for query in queries:
        try:
            resp = _get("https://lite.duckduckgo.com/lite/", params={"q": query, "kl": "wt-wt"})
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            candidates: list[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "duckduckgo" not in href:
                    candidates.append(href)
            candidates.sort(key=lambda u: 0 if any(d in u for d in lyrics_domains) else 1)
            for url in candidates[:4]:
                _delay(0.3, 0.5)
                text = _scrape_lyrics_page(url, artist=artist)
                if text:
                    return text
        except Exception:
            continue
        _delay(0.5, 1.0)
    # Fallback: Google search snippet
    try:
        resp = _get("https://www.google.com/search", params={"q": f"{clean} {artist} lyrics", "hl": "en"})
        if resp:
            gsoup = BeautifulSoup(resp.text, "html.parser")
            for sel in [{"attrs": {"data-lyricid": True}}, {"attrs": {"class": "hwc"}}]:
                divs = gsoup.find_all("div", **sel)
                if divs:
                    text = "\n".join(d.get_text(separator="\n") for d in divs)
                    if len(text.strip()) > 60:
                        return text
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────


def fetch_lyrics(
    title: str, artist: str,
    video_id: str | None = None,
    jiosaavn_id: str | None = None,
    genius_url: str | None = None,
) -> Optional[str]:
    """Try every lyrics source in order until one returns valid lyrics."""
    MIN_LEN = 40

    sources = [
        ("JioSaavn",      lambda: _lyrics_jiosaavn(title, artist, jiosaavn_id)),
        ("YTMusic",       lambda: _lyrics_ytmusic(title, artist, video_id)),
        ("Genius",        lambda: _lyrics_genius(title, artist, genius_url)),
        ("lrclib",        lambda: _lyrics_lrclib(title, artist)),
        ("IndianSites",   lambda: _lyrics_indian_sites(title, artist)),
        ("LyricsOVH",     lambda: _lyrics_ovh(title, artist)),
        ("WebSearch",     lambda: _lyrics_web_search(title, artist)),
    ]

    for name, fn in sources:
        try:
            text = fn()
            if text and len(text.strip()) >= MIN_LEN:
                cleaned = _clean_lyrics(text)
                # Final sanity: reject if the "lyrics" look like prose (avg line > 200 chars)
                lines = [l for l in cleaned.split("\n") if l.strip()]
                if lines:
                    avg = sum(len(l) for l in lines) / len(lines)
                    if avg > 200:
                        continue
                return cleaned
        except Exception:
            pass
        _delay(0.2, 0.6)

    return None


# ══════════════════════════════════════════════════════════════
# File I/O
# ══════════════════════════════════════════════════════════════


def _save_lyrics(output_dir: Path, language: str, index: int,
                 title: str, artist: str, lyrics: str) -> Path:
    lang_dir = output_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    safe_t = _sanitize(title, 60)
    safe_a = _sanitize(artist, 30)
    fname = f"{index:02d}_{safe_t}-{safe_a}.txt"
    path = lang_dir / fname
    header = f"Title:  {title}\nArtist: {artist}\n{'─' * 50}\n\n"
    path.write_text(header + lyrics + "\n", encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════


def run(output_dir: Path = DEFAULT_OUTPUT, count: int = DEFAULT_COUNT,
        languages: list[str] | None = None):
    languages = languages or ["hindi", "haryanvi", "punjabi"]

    print(
        f"\n╔{'═' * 54}╗\n"
        f"║{'🎵  LYRICS HARVESTER':^54}║\n"
        f"║{f'Top {count} songs  ×  {len(languages)} languages':^54}║\n"
        f"║{datetime.now().strftime('%Y-%m-%d %H:%M'):^54}║\n"
        f"╚{'═' * 54}╝\n"
    )

    grand_total = 0
    grand_found = 0
    report: dict[str, dict] = {}

    for lang in languages:
        print(f"\n{'━' * 58}")
        print(f"  🔍  Discovering top {count} {lang.upper()} songs …")
        print(f"{'━' * 58}\n")

        songs = discover_songs(lang, count)
        actual = len(songs)
        print(f"\n  📋  {actual} songs found.  Fetching lyrics …\n")

        found = 0
        failed: list[str] = []

        for i, song in enumerate(songs, 1):
            title = song.get("title", "Unknown")
            artist = song.get("artist", "Unknown")
            tag = f"  [{i:>2}/{actual}]"

            try:
                lyrics = fetch_lyrics(
                    title, artist,
                    video_id=song.get("video_id"),
                    jiosaavn_id=song.get("jiosaavn_id"),
                    genius_url=song.get("genius_url"),
                )
                if lyrics:
                    _save_lyrics(output_dir, lang, i, title, artist, lyrics)
                    found += 1
                    print(f"{tag} ✅  {title}  —  {artist}")
                else:
                    failed.append(f"{title} — {artist}")
                    print(f"{tag} ❌  {title}  —  {artist}  (not found)")
            except Exception as exc:
                failed.append(f"{title} — {artist}")
                print(f"{tag} ⚠️   {title}  —  {artist}  ({exc})")

            _delay(0.3, 1.0)

        grand_total += actual
        grand_found += found
        report[lang] = {"total": actual, "found": found, "failed": failed}
        pct = (found / max(actual, 1)) * 100
        print(f"\n  {lang.upper()} summary: {found}/{actual} lyrics saved ({pct:.0f}%)")

    # Final report
    print(f"\n{'═' * 58}")
    print(f"  📊  FINAL REPORT")
    print(f"{'═' * 58}")
    for lang, stats in report.items():
        pct = (stats["found"] / max(stats["total"], 1)) * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {lang.capitalize():>12}:  {stats['found']:>3}/{stats['total']}  {bar}  {pct:.0f}%")

    total_pct = (grand_found / max(grand_total, 1)) * 100
    print(f"  {'TOTAL':>12}:  {grand_found:>3}/{grand_total}  ({total_pct:.0f}%)")
    print(f"\n  📁  Lyrics saved to:  {output_dir.resolve()}")

    report_path = output_dir / "harvest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "summary": {k: {"total": v["total"], "found": v["found"]} for k, v in report.items()},
            "failed": {k: v["failed"] for k, v in report.items()},
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  📄  Report:           {report_path.resolve()}\n")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🎵 Lyrics Harvester")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=DEFAULT_COUNT, help="Songs per language (default: 50)")
    parser.add_argument("--languages", "-l", nargs="+", default=["hindi", "haryanvi", "punjabi"], help="Languages")
    args = parser.parse_args()
    run(output_dir=args.output, count=args.count, languages=args.languages)

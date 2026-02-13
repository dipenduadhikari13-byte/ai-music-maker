#!/usr/bin/env python3
"""
🎵  Lyrics Harvester v2  —  Artist-Based
==========================================
Fetches lyrics for the **top 20 songs** of each of the **top 10 artists**
in Hindi, Haryanvi, and Punjabi music.

Organization:
    lyrics_v2/<language>/<ArtistName>/01_SongTitle.txt

Usage:
    python lyrics_harvester_v2.py
    python lyrics_harvester_v2.py --songs-per-artist 10
    python lyrics_harvester_v2.py --languages hindi haryanvi
    python lyrics_harvester_v2.py --output my_lyrics

Dependencies (auto-installed): requests, beautifulsoup4, ytmusicapi
"""

from __future__ import annotations

import os, sys, re, time, json, html as html_mod, random, logging
import unicodedata, urllib.parse
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Auto-install ──────────────────────────────────────────────

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

# ── Configuration ─────────────────────────────────────────────

DEFAULT_OUTPUT   = Path("lyrics_v2")
DEFAULT_SONGS    = 20          # songs per artist
TIMEOUT          = 20
MAX_RETRIES      = 3

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger("harvester_v2")

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8,pa;q=0.7",
})

# ══════════════════════════════════════════════════════════════
#  TOP 10 ARTISTS PER LANGUAGE
# ══════════════════════════════════════════════════════════════

TOP_ARTISTS: dict[str, list[str]] = {
    "hindi": [
        "Arijit Singh",
        "Shreya Ghoshal",
        "Jubin Nautiyal",
        "Atif Aslam",
        "Neha Kakkar",
        "Darshan Raval",
        "Vishal Mishra",
        "Pritam",
        "Armaan Malik",
        "Sachet Tandon",
    ],
    "haryanvi": [
        "Sapna Choudhary",
        "Masoom Sharma",
        "Renuka Panwar",
        "Pranjal Dahiya",
        "Amanraj Gill",
        "Ruchika Jangid",
        "Raj Mawar",
        "Kanchan Nagar",
        "Ajay Bhagta",
        "Khasa Aala Chahar",
    ],
    "punjabi": [
        "Sidhu Moose Wala",
        "AP Dhillon",
        "Diljit Dosanjh",
        "Karan Aujla",
        "Ammy Virk",
        "Harrdy Sandhu",
        "Guru Randhawa",
        "Jass Manak",
        "Shubh",
        "Jasmine Sandlas",
    ],
}

# ══════════════════════════════════════════════════════════════
#  HELPERS  (shared with v1)
# ══════════════════════════════════════════════════════════════

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

def _delay(lo: float = 0.6, hi: float = 1.5):
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
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text)

def _strip_meta(title: str) -> str:
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip()
    cleaned = re.sub(r"\s*feat\.?\s.*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or title

def _title_match(query_title: str, result_title: str, threshold: float = 0.50) -> bool:
    q = _normalize(_strip_meta(query_title))
    r = _normalize(_strip_meta(result_title))
    if not q or not r:
        return False
    if q == r:
        return True
    if q in r and len(r) <= len(q) * 1.5:
        return True
    if r in q and len(q) <= len(r) * 1.5:
        return True
    return SequenceMatcher(None, q, r).ratio() >= threshold

def _artist_match(query_artist: str, result_artist: str) -> bool:
    q_parts = {p for p in _normalize(query_artist).split() if len(p) > 2} if query_artist else set()
    r_parts = {p for p in _normalize(result_artist).split() if len(p) > 2} if result_artist else set()
    if not q_parts or not r_parts:
        return True
    q_full = _normalize(query_artist)
    r_full = _normalize(result_artist)
    if q_full in r_full or r_full in q_full:
        return True
    return bool(q_parts & r_parts)

# ══════════════════════════════════════════════════════════════
#  SONG DISCOVERY PER ARTIST
# ══════════════════════════════════════════════════════════════

_JUNK = re.compile(
    r"(?i)(jukebox|video jukebox|compilation|mashup|nonstop|non-stop|medley|"
    r"music video|official video|lyric video|lyrical|full album|back to back|"
    r"top hits|best of \d{4}|superhit|dj mix|video song|audio jukebox)"
)

def _is_valid_song(title: str) -> bool:
    if len(title) > 120:
        return False
    if _JUNK.search(title):
        return False
    if title.count("|") >= 2:
        return False
    return True

def _clean_title(title: str) -> str:
    title = re.split(r"\s*[|•]\s*", title)[0].strip()
    title = re.sub(r"\s+\d{4}$", "", title).strip()
    title = re.sub(r"\s*\((?:Official|Full|HD)\s+(?:Video|Audio|Song)\)", "", title, flags=re.I).strip()
    return title

_SAAVN = "https://www.jiosaavn.com/api.php"

def _discover_artist_songs_yt(artist: str, language: str, needed: int) -> list[dict]:
    """Find songs by a specific artist via YouTube Music."""
    if not _YT_AVAILABLE or _yt_instance is None:
        return []
    yt = _yt_instance
    songs: list[dict] = []
    queries = [
        f"{artist} top songs",
        f"{artist} best songs",
        f"{artist} {language} songs",
        f"{artist} latest hits",
    ]
    # Try artist page first
    try:
        results = yt.search(artist, filter="artists", limit=3)
        for r in results[:2]:
            bid = r.get("browseId")
            if not bid:
                continue
            try:
                artist_data = yt.get_artist(bid)
                # songs section
                for section_key in ["songs", "singles"]:
                    section = artist_data.get(section_key, {})
                    if isinstance(section, dict):
                        for t in section.get("results", []):
                            if t.get("title") and _is_valid_song(t["title"]):
                                songs.append({
                                    "title": _clean_title(t["title"]),
                                    "artist": artist,
                                    "video_id": t.get("videoId"),
                                })
                # If there's a browse param, get more
                browse = (artist_data.get("songs", {}) or {}).get("browseId")
                if browse and len(songs) < needed:
                    try:
                        playlist = yt.get_playlist(browse, limit=needed)
                        for t in playlist.get("tracks", []):
                            if t.get("title") and _is_valid_song(t["title"]):
                                songs.append({
                                    "title": _clean_title(t["title"]),
                                    "artist": artist,
                                    "video_id": t.get("videoId"),
                                })
                    except Exception:
                        pass
            except Exception:
                continue
        _delay(0.3, 0.8)
    except Exception:
        pass

    # Fallback: search queries
    if len(songs) < needed:
        for q in queries:
            if len(songs) >= needed * 2:
                break
            try:
                results = yt.search(q, filter="songs", limit=min(needed, 30))
                for t in results:
                    if t.get("title") and _is_valid_song(t["title"]):
                        # Verify artist
                        t_artist = ", ".join(
                            a.get("name", "") for a in (t.get("artists") or [])[:2] if a.get("name")
                        ) or "Unknown"
                        if _artist_match(artist, t_artist):
                            songs.append({
                                "title": _clean_title(t["title"]),
                                "artist": artist,
                                "video_id": t.get("videoId"),
                            })
                _delay(0.3, 0.6)
            except Exception:
                continue
    return songs


def _discover_artist_songs_saavn(artist: str, language: str, needed: int) -> list[dict]:
    """Find songs by a specific artist via JioSaavn."""
    songs: list[dict] = []
    queries = [
        f"{artist}",
        f"{artist} {language}",
        f"{artist} top songs",
        f"{artist} latest",
    ]
    for q in queries:
        if len(songs) >= needed * 2:
            break
        try:
            resp = _get(_SAAVN, params={
                "__call": "search.getResults", "p": "1", "q": q,
                "_format": "json", "_marker": "0", "ctx": "wap6dot0",
                "n": str(min(needed * 2, 50)),
            })
            data = _safe_json(resp)
            if isinstance(data, dict):
                for item in data.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    title = html_mod.unescape(item.get("title", item.get("song", ""))).strip()
                    item_artist = html_mod.unescape(
                        item.get("primary_artists", item.get("music", ""))
                    ).strip()
                    if title and _is_valid_song(title) and _artist_match(artist, item_artist):
                        songs.append({
                            "title": _clean_title(title),
                            "artist": artist,
                            "jiosaavn_id": item.get("id"),
                        })
            _delay(0.3, 0.7)
        except Exception:
            continue
    return songs


def _dedup_songs(songs: list[dict]) -> list[dict]:
    """Deduplicate by normalized title, prefer non-remix."""
    seen: dict[str, dict] = {}
    for s in songs:
        title = s.get("title", "")
        key = _normalize(_strip_meta(title))
        if not key:
            continue
        if key not in seen:
            seen[key] = s
        else:
            is_remix = bool(re.search(r"(?i)(lofi|slowed|reverb|remix|8d)", title))
            existing_remix = bool(re.search(r"(?i)(lofi|slowed|reverb|remix|8d)", seen[key].get("title", "")))
            if existing_remix and not is_remix:
                seen[key] = s
    return list(seen.values())


def discover_artist_songs(artist: str, language: str, needed: int = 20) -> list[dict]:
    """Discover top songs for a specific artist from multiple sources."""
    pool: list[dict] = []

    # YouTube Music
    if _YT_AVAILABLE:
        try:
            yt_songs = _discover_artist_songs_yt(artist, language, needed)
            log.info(f"    YouTube Music → {len(yt_songs)} candidates")
            pool.extend(yt_songs)
        except Exception as exc:
            log.warning(f"    YouTube Music error: {exc}")

    # JioSaavn
    try:
        js_songs = _discover_artist_songs_saavn(artist, language, needed)
        log.info(f"    JioSaavn      → {len(js_songs)} candidates")
        pool.extend(js_songs)
    except Exception as exc:
        log.warning(f"    JioSaavn error: {exc}")

    result = _dedup_songs(pool)
    log.info(f"    Unique songs  → {len(result)} (need {needed})")
    return result[:needed]


# ══════════════════════════════════════════════════════════════
#  LYRICS FETCHING  (all backends)
# ══════════════════════════════════════════════════════════════

def _lyrics_jiosaavn(title: str, artist: str, song_id: str | None = None) -> Optional[str]:
    ids_to_try: list[str] = []
    if song_id:
        ids_to_try.append(song_id)
    for q in [f"{_strip_meta(title)} {artist}", _strip_meta(title)]:
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
            _delay(0.2, 0.4)
        except Exception:
            continue
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
    if not _YT_AVAILABLE or _yt_instance is None:
        return None
    yt = _yt_instance
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
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\d+\s+Contributor', stripped):
            continue
        if re.match(r'^Translations?', stripped, re.IGNORECASE):
            continue
        if 'Read More' in stripped and len(stripped) > 200:
            continue
        if re.match(r'^.*Lyrics$', stripped) and len(stripped) < 100:
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
                        if _title_match(title, res_title) and _artist_match(artist, res_artist):
                            url = res.get("url")
                            break
                    if url:
                        break
                if url:
                    break
                _delay(0.3, 0.5)
        if not url:
            return None
        _delay(0.3, 0.6)
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
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None
        if sum(len(l) for l in lines) / len(lines) > 200:
            return None
        return text
    except Exception:
        pass
    return None

def _lyrics_lrclib(title: str, artist: str) -> Optional[str]:
    clean = _strip_meta(title)
    for q in [f"{clean} {artist}", clean]:
        try:
            resp = _get(
                "https://lrclib.net/api/search", params={"q": q},
                headers={"Lrclib-Client": "LyricsHarvester/2.0"},
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

def _lyrics_ovh(title: str, artist: str) -> Optional[str]:
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

_INDIAN_LYRIC_SITES = [
    ("lyricsmint.com", "https://www.lyricsmint.com/", "s"),
    ("lyricsted.com",  "https://www.lyricsted.com/",  "s"),
    ("lyricsbell.com", "https://www.lyricsbell.com/", "s"),
]

def _scrape_lyrics_page(url: str, artist: str = "") -> Optional[str]:
    try:
        resp = _get(url)
        if not resp:
            return None
        name_words: set[str] = set()
        if artist:
            soup_pre = BeautifulSoup(resp.text[:8000], "html.parser")
            check_parts = [htag.get_text().lower() for htag in soup_pre.find_all(["title", "h1", "h2"])]
            check_text = " ".join(check_parts)
            for part in re.split(r'[,&/]+', artist.lower()):
                for word in part.split():
                    if len(word) > 2:
                        name_words.add(word)
            if name_words and check_text.strip() and not any(w in check_text for w in name_words):
                return None
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
                first_200 = text[:200].lower()
                junk = ["disclaimer", "privacy policy", "cookie", "terms of service",
                        "contact us", "about us", "copyright notice", "all rights reserved"]
                if any(m in first_200 for m in junk):
                    return None
                if name_words:
                    header = text[:500].lower()
                    if any(m in header for m in ["sung by", "singer", "artist:", "vocals"]):
                        if not any(w in header for w in name_words):
                            continue
                lines = [l for l in text.split("\n") if l.strip()]
                if len(text.strip()) > 60 and len(lines) >= 3:
                    avg = sum(len(l) for l in lines) / len(lines)
                    if avg < 200:
                        return text
        return None
    except Exception:
        return None

def _lyrics_indian_sites(title: str, artist: str) -> Optional[str]:
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
                _delay(0.3, 0.5)
                text = _scrape_lyrics_page(target_url, artist=artist)
                if text:
                    return text
            except Exception:
                continue
    return None

def _lyrics_web_search(title: str, artist: str) -> Optional[str]:
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
            candidates = [a["href"] for a in soup.find_all("a", href=True)
                          if a["href"].startswith("http") and "duckduckgo" not in a["href"]]
            candidates.sort(key=lambda u: 0 if any(d in u for d in lyrics_domains) else 1)
            for url in candidates[:4]:
                _delay(0.3, 0.5)
                text = _scrape_lyrics_page(url, artist=artist)
                if text:
                    return text
        except Exception:
            continue
        _delay(0.5, 0.8)
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


def fetch_lyrics(
    title: str, artist: str,
    video_id: str | None = None,
    jiosaavn_id: str | None = None,
    genius_url: str | None = None,
) -> Optional[str]:
    """Try every lyrics source in order until one returns valid lyrics."""
    MIN_LEN = 40
    sources = [
        ("JioSaavn",    lambda: _lyrics_jiosaavn(title, artist, jiosaavn_id)),
        ("YTMusic",     lambda: _lyrics_ytmusic(title, artist, video_id)),
        ("Genius",      lambda: _lyrics_genius(title, artist, genius_url)),
        ("lrclib",      lambda: _lyrics_lrclib(title, artist)),
        ("IndianSites", lambda: _lyrics_indian_sites(title, artist)),
        ("LyricsOVH",   lambda: _lyrics_ovh(title, artist)),
        ("WebSearch",   lambda: _lyrics_web_search(title, artist)),
    ]
    for name, fn in sources:
        try:
            text = fn()
            if text and len(text.strip()) >= MIN_LEN:
                cleaned = _clean_lyrics(text)
                lines = [l for l in cleaned.split("\n") if l.strip()]
                if lines:
                    avg = sum(len(l) for l in lines) / len(lines)
                    if avg > 200:
                        continue
                return cleaned
        except Exception:
            pass
        _delay(0.15, 0.4)
    return None


# ══════════════════════════════════════════════════════════════
#  FILE I/O
# ══════════════════════════════════════════════════════════════

def _save_lyrics(output_dir: Path, language: str, artist: str,
                 index: int, title: str, lyrics: str) -> Path:
    artist_dir = output_dir / language / _sanitize(artist, 40)
    artist_dir.mkdir(parents=True, exist_ok=True)
    safe_t = _sanitize(title, 60)
    fname = f"{index:02d}_{safe_t}.txt"
    path = artist_dir / fname
    header = f"Title:  {title}\nArtist: {artist}\n{'─' * 50}\n\n"
    path.write_text(header + lyrics + "\n", encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run(output_dir: Path = DEFAULT_OUTPUT, songs_per_artist: int = DEFAULT_SONGS,
        languages: list[str] | None = None):
    languages = languages or ["hindi", "haryanvi", "punjabi"]

    total_artists = sum(len(TOP_ARTISTS.get(l, [])) for l in languages)
    total_songs_target = total_artists * songs_per_artist

    print(
        f"\n╔{'═' * 58}╗\n"
        f"║{'🎵  LYRICS HARVESTER v2  —  Artist-Based':^58}║\n"
        f"║{f'{total_artists} artists × {songs_per_artist} songs = {total_songs_target} target':^58}║\n"
        f"║{datetime.now().strftime('%Y-%m-%d %H:%M'):^58}║\n"
        f"╚{'═' * 58}╝\n"
    )

    grand_total = 0
    grand_found = 0
    report: dict[str, dict] = {}

    for lang in languages:
        artists = TOP_ARTISTS.get(lang, [])
        if not artists:
            log.warning(f"  No artists configured for '{lang}', skipping.")
            continue

        print(f"\n{'━' * 62}")
        print(f"  🔍  {lang.upper()}  —  {len(artists)} artists × {songs_per_artist} songs")
        print(f"{'━' * 62}")

        lang_total = 0
        lang_found = 0
        lang_failed: list[str] = []
        artist_stats: dict[str, dict] = {}

        for ai, artist in enumerate(artists, 1):
            print(f"\n  🎤  [{ai}/{len(artists)}]  {artist}")
            print(f"  {'─' * 50}")

            songs = discover_artist_songs(artist, lang, songs_per_artist)
            actual = len(songs)
            print(f"    📋  {actual} songs discovered. Fetching lyrics …\n")

            a_found = 0
            a_failed: list[str] = []

            for si, song in enumerate(songs, 1):
                title = song.get("title", "Unknown")
                tag = f"    [{si:>2}/{actual}]"

                try:
                    lyrics = fetch_lyrics(
                        title, artist,
                        video_id=song.get("video_id"),
                        jiosaavn_id=song.get("jiosaavn_id"),
                        genius_url=song.get("genius_url"),
                    )
                    if lyrics:
                        _save_lyrics(output_dir, lang, artist, si, title, lyrics)
                        a_found += 1
                        print(f"{tag} ✅  {title}")
                    else:
                        a_failed.append(title)
                        print(f"{tag} ❌  {title}  (not found)")
                except Exception as exc:
                    a_failed.append(title)
                    print(f"{tag} ⚠️   {title}  ({exc})")

                _delay(0.2, 0.6)

            pct = (a_found / max(actual, 1)) * 100
            print(f"\n    {artist}: {a_found}/{actual} lyrics ({pct:.0f}%)")
            artist_stats[artist] = {"total": actual, "found": a_found, "failed": a_failed}
            lang_total += actual
            lang_found += a_found
            lang_failed.extend([f"{title} — {artist}" for title in a_failed])

        grand_total += lang_total
        grand_found += lang_found
        report[lang] = {
            "total": lang_total, "found": lang_found,
            "failed": lang_failed, "artists": artist_stats,
        }
        pct = (lang_found / max(lang_total, 1)) * 100
        print(f"\n  {lang.upper()} TOTAL: {lang_found}/{lang_total} lyrics saved ({pct:.0f}%)")

    # ── Final report ──
    print(f"\n{'═' * 62}")
    print(f"  📊  FINAL REPORT")
    print(f"{'═' * 62}")
    for lang, stats in report.items():
        pct = (stats["found"] / max(stats["total"], 1)) * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {lang.capitalize():>12}:  {stats['found']:>3}/{stats['total']}  {bar}  {pct:.0f}%")
        for artist, a_stats in stats.get("artists", {}).items():
            a_pct = (a_stats["found"] / max(a_stats["total"], 1)) * 100
            print(f"    {'→':>10} {artist:<25} {a_stats['found']:>2}/{a_stats['total']}  ({a_pct:.0f}%)")

    total_pct = (grand_found / max(grand_total, 1)) * 100
    print(f"\n  {'TOTAL':>12}:  {grand_found:>3}/{grand_total}  ({total_pct:.0f}%)")
    print(f"\n  📁  Lyrics saved to:  {output_dir.resolve()}")

    # Save JSON report
    report_path = output_dir / "harvest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "songs_per_artist": songs_per_artist,
            "languages": languages,
            "artists": {k: v for k, v in TOP_ARTISTS.items() if k in languages},
        },
        "summary": {k: {"total": v["total"], "found": v["found"]} for k, v in report.items()},
        "artist_stats": {
            lang: {
                artist: {"total": a["total"], "found": a["found"], "failed": a["failed"]}
                for artist, a in stats.get("artists", {}).items()
            }
            for lang, stats in report.items()
        },
        "failed": {k: v["failed"] for k, v in report.items()},
    }
    report_path.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  📄  Report:           {report_path.resolve()}\n")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🎵 Lyrics Harvester v2 — Artist-Based")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--songs-per-artist", "-n", type=int, default=DEFAULT_SONGS, help="Songs per artist (default: 20)")
    parser.add_argument("--languages", "-l", nargs="+", default=["hindi", "haryanvi", "punjabi"], help="Languages")
    args = parser.parse_args()
    run(output_dir=args.output, songs_per_artist=args.songs_per_artist, languages=args.languages)

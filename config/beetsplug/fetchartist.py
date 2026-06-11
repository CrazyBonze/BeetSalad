"""
fetchartist – Artist image fetcher for beets.

Downloads artist thumbnail images from multiple configurable sources and saves
them alongside the artist's albums so that Navidrome (and other media servers)
can display them.

Navidrome looks for ``artist.*`` in:
  1. The artist folder  (e.g.  ``Music/Pink Floyd/artist.jpg``)
  2. Any album folder   (e.g.  ``Music/Pink Floyd/DSOTM/artist.jpg``)
  3. An external service (Spotify by default)

This plugin places ``artist.jpg`` (configurable) in the artist-level directory
so that Navidrome finds it at priority 1.

Architecture mirrors ``beetsplug.fetchart``:
  • Pluggable *sources* with a configurable priority order
  • Each source yields ``Candidate`` objects (url + optional metadata)
  • The first acceptable candidate wins (downloaded, validated, resized, saved)
  • Image processing re-uses beets' ``ArtResizer`` when available
  • CLI subcommand ``beet fetchartist [-f]`` for manual runs
  • Auto-mode on import (hooks ``import_task_apply``)

Sources (all optional, order set by ``sources`` config key):

  filesystem     – local ``artist.*`` already in the artist directory
  fanarttv       – fanart.tv ``artistthumb`` via MusicBrainz artist ID
  theaudiodb     – TheAudioDB ``strArtistThumb`` via MB ID or name search
  spotify        – Spotify Web API artist images via client-credentials
  wikidata       – Wikidata P18 (image) claims via entity search
  discogs        – Discogs artist images via token auth

Requires: ``requests``, ``Pillow`` (optional, for format conversion / crop).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unicodedata
from contextlib import closing
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional

import requests

from beets import config as beets_config
from beets import util
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, decargs, print_
from beets.util.artresizer import ArtResizer

if TYPE_CHECKING:
    from beets.library import Library


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------


class Candidate(NamedTuple):
    """A potential artist image."""

    url: Optional[str] = None
    path: Optional[str] = None
    source_name: str = ""
    size: Optional[tuple[int, int]] = None  # (width, height) if known


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _norm(value: str | None) -> str:
    """Normalise a string for fuzzy comparison."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    for old, new in {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "&": " and ",
    }.items():
        value = value.replace(old, new)
    value = re.sub(r"[^\w\s]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _artist_match(left: str | None, right: str | None) -> bool:
    ln, rn = _norm(left), _norm(right)
    if not ln or not rn:
        return False
    if ln == rn:
        return True
    va = {"various artists", "various", "va"}
    return ln in va and rn in va


def _mb_artistid(album) -> Optional[str]:
    """Extract a MusicBrainz artist ID from an album or its items."""
    for field in ("mb_albumartistid", "mb_artistid"):
        val = getattr(album, field, None)
        if val:
            return str(val).split(";")[0].strip()
    try:
        for item in album.items():
            for field in ("mb_albumartistid", "mb_artistid"):
                val = getattr(item, field, None)
                if val:
                    return str(val).split(";")[0].strip()
    except Exception:
        pass
    return None


def _spotify_id(album) -> Optional[str]:
    """Extract a Spotify artist ID if present."""
    for field in ("spotify_artist_id", "spotify_albumartist_id"):
        val = getattr(album, field, None)
        if val:
            return str(val).split(";")[0].strip()
    try:
        for item in album.items():
            for field in ("spotify_artist_id",):
                val = getattr(item, field, None)
                if val:
                    return str(val).split(";")[0].strip()
    except Exception:
        pass
    return None


def _discogs_artistid(album) -> Optional[int]:
    for field in ("discogs_artistid", "discogs_artist_id"):
        val = getattr(album, field, None)
        if val not in (None, "", 0, "0"):
            try:
                return int(str(val))
            except Exception:
                pass
    return None


def _artist_dir(album) -> Optional[str]:
    """Derive the artist-level directory from the album's item paths.

    Assumes ``<library>/<artist>/<album>/tracks`` layout.

    This function must only be called when item paths point to their final
    library locations.  During import, that means hooking into the
    ``import_task_files`` event (which fires *after* files have been
    copied/moved), NOT ``import_stages`` (which fires before).  For the
    CLI subcommand the library is already in its final state.

    We read the actual on-disk ``item.path``, walk up to find the artist
    directory, and verify it lives inside the configured library root.
    """
    lib_dir = util.displayable_path(beets_config["directory"].as_filename())
    lib_norm = os.path.normpath(lib_dir)

    # -- Try album.path first (beets sets this to the album dir in the
    #    library once the album is committed to the DB). -----------------
    album_dir = None
    album_path = getattr(album, "path", None)
    if album_path:
        candidate = util.displayable_path(album_path)
        if _is_under(candidate, lib_norm):
            album_dir = candidate

    # -- Fall back to the first item's actual on-disk path. --------------
    if not album_dir:
        try:
            items = list(album.items())
        except Exception:
            return None
        if not items:
            return None

        item_path = util.displayable_path(items[0].path)
        candidate = os.path.dirname(item_path)
        if _is_under(candidate, lib_norm):
            album_dir = candidate

    if not album_dir:
        return None

    # Go up one level from the album directory to get the artist directory.
    artist_dir = os.path.dirname(album_dir)

    # If artist_dir *is* the library root, the layout is flat (no artist
    # subfolder).  Fall back to the album directory itself.
    if os.path.normpath(artist_dir) == lib_norm:
        return album_dir

    # Final sanity check: the artist dir must still be inside the library.
    if not _is_under(artist_dir, lib_norm):
        return album_dir

    return artist_dir


def _is_under(path: str, root: str) -> bool:
    """Check whether *path* is inside *root* (normalised comparison)."""
    try:
        return os.path.normpath(path).startswith(os.path.normpath(root))
    except Exception:
        return False


def _cfg():
    """Shortcut to the plugin's config subtree."""
    return beets_config["fetchartist"]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
#  Sources
# ---------------------------------------------------------------------------


class ArtistArtSource:
    """Base class for all artist-image sources."""

    NAME: str = ""

    def __init__(self, log):
        self._log = log

    def get(
        self, artist_name: str, mb_artistid: Optional[str], album, **kw
    ) -> Iterator[Candidate]:
        """Yield ``Candidate`` objects for the given artist."""
        raise NotImplementedError

    # Convenience HTTP helper -----------------------------------------------
    def _request(self, url: str, **kwargs) -> requests.Response:
        timeout = _cfg()["request_timeout"].get(int)
        kwargs.setdefault("timeout", timeout)
        headers = kwargs.pop("headers", {})
        headers.setdefault("User-Agent", "beets-fetchartist/1.0 (+https://beets.io)")
        return requests.get(url, headers=headers, **kwargs)


class FileSystemSource(ArtistArtSource):
    """Look for an existing ``artist.*`` file in the artist directory."""

    NAME = "filesystem"

    def get(self, artist_name, mb_artistid, album, **kw):
        adir = _artist_dir(album)
        if not adir or not os.path.isdir(adir):
            return

        cover_names = _cfg()["cover_names"].as_str_seq()
        for entry in sorted(os.listdir(adir)):
            stem, ext = os.path.splitext(entry)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            if stem.lower() in [n.lower() for n in cover_names]:
                full = os.path.join(adir, entry)
                if os.path.isfile(full):
                    self._log.debug("filesystem: found {}", full)
                    yield Candidate(path=full, source_name=self.NAME)


class FanartTVSource(ArtistArtSource):
    """fanart.tv ``artistthumb`` images (requires MusicBrainz artist ID)."""

    NAME = "fanarttv"
    API_BASE = "https://webservice.fanart.tv/v3/music"

    def _available(self) -> bool:
        key = _cfg()["fanarttv_key"].get()
        if not key:
            self._log.debug("fanarttv: disabled – no API key")
            return False
        return True

    def get(self, artist_name, mb_artistid, album, **kw):
        if not self._available():
            return
        if not mb_artistid:
            self._log.debug("fanarttv: no MB artist ID available")
            return

        key = _cfg()["fanarttv_key"].get(str)
        url = f"{self.API_BASE}/{mb_artistid}"
        params = {"api_key": key}
        client_key = _cfg()["fanarttv_client_key"].get()
        if client_key:
            params["client_key"] = client_key

        try:
            resp = self._request(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            self._log.debug("fanarttv: request failed: {}", exc)
            return

        thumbs = data.get("artistthumb") or []
        self._log.debug("fanarttv: {} thumbnails for {}", len(thumbs), mb_artistid)

        # Sort by likes descending.
        thumbs.sort(key=lambda t: int(t.get("likes", 0)), reverse=True)

        for thumb in thumbs:
            img_url = thumb.get("url")
            if img_url:
                yield Candidate(url=img_url, source_name=self.NAME)


class TheAudioDBSource(ArtistArtSource):
    """TheAudioDB artist thumbnails (free tier, MB ID or name search)."""

    NAME = "theaudiodb"
    API_BASE = "https://www.theaudiodb.com/api/v1/json"

    def _api_key(self) -> str:
        return _cfg()["theaudiodb_key"].get(str) or "2"

    def _lookup_by_mbid(self, mbid: str) -> Optional[dict]:
        url = f"{self.API_BASE}/{self._api_key()}/artist-mb.php"
        try:
            resp = self._request(url, params={"i": mbid})
            resp.raise_for_status()
            data = resp.json()
            artists = data.get("artists")
            return artists[0] if artists else None
        except Exception as exc:
            self._log.debug("theaudiodb: MB lookup failed: {}", exc)
            return None

    def _search_by_name(self, name: str) -> Optional[dict]:
        url = f"{self.API_BASE}/{self._api_key()}/search.php"
        try:
            resp = self._request(url, params={"s": name})
            resp.raise_for_status()
            data = resp.json()
            artists = data.get("artists")
            return artists[0] if artists else None
        except Exception as exc:
            self._log.debug("theaudiodb: name search failed: {}", exc)
            return None

    def get(self, artist_name, mb_artistid, album, **kw):
        artist_data = None
        if mb_artistid:
            artist_data = self._lookup_by_mbid(mb_artistid)

        if not artist_data and artist_name:
            artist_data = self._search_by_name(artist_name)
            if artist_data:
                # Verify the name roughly matches.
                tadb_name = artist_data.get("strArtist")
                if not _artist_match(tadb_name, artist_name):
                    self._log.debug(
                        "theaudiodb: name mismatch: {!r} vs {!r}",
                        tadb_name,
                        artist_name,
                    )
                    artist_data = None

        if not artist_data:
            self._log.debug("theaudiodb: no artist found")
            return

        # Prefer thumb, then wider images.
        for key in (
            "strArtistThumb",
            "strArtistFanart",
            "strArtistFanart2",
            "strArtistFanart3",
            "strArtistFanart4",
        ):
            img_url = artist_data.get(key)
            if img_url and img_url.lower() != "null":
                self._log.debug("theaudiodb: found {} = {}", key, img_url)
                yield Candidate(url=img_url, source_name=self.NAME)
                # For thumbnails, only the first (thumb) matters most.
                if key == "strArtistThumb":
                    return


class SpotifySource(ArtistArtSource):
    """Spotify artist images via client-credentials OAuth."""

    NAME = "spotify"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"

    _access_token: Optional[str] = None

    def _available(self) -> bool:
        cid = _cfg()["spotify_client_id"].get()
        secret = _cfg()["spotify_client_secret"].get()
        if not cid or not secret:
            self._log.debug("spotify: disabled – no client_id / client_secret")
            return False
        return True

    def _authenticate(self) -> Optional[str]:
        if self._access_token:
            return self._access_token
        cid = _cfg()["spotify_client_id"].get(str)
        secret = _cfg()["spotify_client_secret"].get(str)
        try:
            resp = requests.post(
                self.TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(cid, secret),
                timeout=_cfg()["request_timeout"].get(int),
            )
            resp.raise_for_status()
            self._access_token = resp.json()["access_token"]
            return self._access_token
        except Exception as exc:
            self._log.debug("spotify: auth failed: {}", exc)
            return None

    def _search_artist(self, name: str) -> Optional[dict]:
        token = self._authenticate()
        if not token:
            return None
        try:
            resp = requests.get(
                f"{self.API_BASE}/search",
                params={"q": f'artist:"{name}"', "type": "artist", "limit": 5},
                headers={"Authorization": f"Bearer {token}"},
                timeout=_cfg()["request_timeout"].get(int),
            )
            resp.raise_for_status()
            items = resp.json().get("artists", {}).get("items", [])
            for item in items:
                if _artist_match(item.get("name"), name):
                    return item
            return items[0] if items else None
        except Exception as exc:
            self._log.debug("spotify: search failed: {}", exc)
            return None

    def _get_artist_by_id(self, spotify_id: str) -> Optional[dict]:
        token = self._authenticate()
        if not token:
            return None
        try:
            resp = requests.get(
                f"{self.API_BASE}/artists/{spotify_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=_cfg()["request_timeout"].get(int),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self._log.debug("spotify: artist lookup failed: {}", exc)
            return None

    def get(self, artist_name, mb_artistid, album, **kw):
        if not self._available():
            return

        artist_data = None
        sid = _spotify_id(album)
        if sid:
            artist_data = self._get_artist_by_id(sid)

        if not artist_data and artist_name:
            artist_data = self._search_artist(artist_name)

        if not artist_data:
            self._log.debug("spotify: no artist found")
            return

        images = artist_data.get("images") or []
        # Spotify returns images sorted widest-first.
        # We want the largest available, but square-ish preferred.
        for img in images:
            url = img.get("url")
            w, h = img.get("width"), img.get("height")
            if url:
                size = (w, h) if w and h else None
                yield Candidate(url=url, source_name=self.NAME, size=size)


class WikidataSource(ArtistArtSource):
    """Wikidata P18 (image) claims via entity search."""

    NAME = "wikidata"
    SEARCH_URL = "https://www.wikidata.org/w/api.php"
    COMMONS_URL = "https://commons.wikimedia.org/w/api.php"

    def _search_entities(self, query: str) -> list[str]:
        try:
            resp = self._request(
                self.SEARCH_URL,
                params={
                    "action": "wbsearchentities",
                    "format": "json",
                    "language": "en",
                    "type": "item",
                    "limit": _cfg()["wikidata_search_limit"].get(int),
                    "search": query,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._log.debug("wikidata: search failed: {}", exc)
            return []
        return [r["id"] for r in data.get("search", []) if r.get("id")]

    def _entity_image_names(self, entity_ids: list[str]) -> Iterator[str]:
        if not entity_ids:
            return
        try:
            resp = self._request(
                self.SEARCH_URL,
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "props": "claims",
                    "ids": "|".join(entity_ids),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._log.debug("wikidata: entity fetch failed: {}", exc)
            return

        for entity in (data.get("entities") or {}).values():
            for claim in (entity.get("claims") or {}).get("P18") or []:
                try:
                    yield claim["mainsnak"]["datavalue"]["value"]
                except Exception:
                    continue

    def _commons_image_url(self, filename: str) -> Optional[str]:
        title = filename if filename.startswith("File:") else f"File:{filename}"
        try:
            resp = self._request(
                self.COMMONS_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "titles": title,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._log.debug("wikidata: commons lookup failed: {}", exc)
            return None

        for page in (data.get("query") or {}).get("pages", {}).values():
            try:
                return page["imageinfo"][0]["url"]
            except Exception:
                continue
        return None

    def get(self, artist_name, mb_artistid, album, **kw):
        queries = []
        if artist_name:
            queries.append(f"{artist_name} musician")
            queries.append(artist_name)

        seen: set[str] = set()
        for query in queries:
            entity_ids = self._search_entities(query)
            for image_name in self._entity_image_names(entity_ids):
                image_url = self._commons_image_url(image_name)
                if image_url and image_url not in seen:
                    seen.add(image_url)
                    yield Candidate(url=image_url, source_name=self.NAME)


class DiscogsSource(ArtistArtSource):
    """Discogs artist images (requires personal access token)."""

    NAME = "discogs"
    API_BASE = "https://api.discogs.com"

    def _available(self) -> bool:
        token = _cfg()["discogs_token"].get()
        if not token:
            self._log.debug("discogs: disabled – no token")
            return False
        return True

    def _headers(self) -> dict[str, str]:
        token = _cfg()["discogs_token"].get(str)
        return {
            "User-Agent": "beets-fetchartist/1.0 (+https://beets.io)",
            "Accept": "application/json",
            "Authorization": f"Discogs token={token}",
        }

    def _artist_by_id(self, artist_id: int) -> Optional[dict]:
        try:
            resp = self._request(
                f"{self.API_BASE}/artists/{artist_id}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self._log.debug("discogs: artist lookup failed: {}", exc)
            return None

    def _search_artist(self, name: str) -> Optional[dict]:
        try:
            resp = self._request(
                f"{self.API_BASE}/database/search",
                params={"type": "artist", "q": name, "per_page": 5},
                headers=self._headers(),
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except Exception as exc:
            self._log.debug("discogs: search failed: {}", exc)
            return None

        for r in results:
            if _artist_match(r.get("title"), name):
                return self._artist_by_id(r["id"])
        if results:
            return self._artist_by_id(results[0]["id"])
        return None

    def get(self, artist_name, mb_artistid, album, **kw):
        if not self._available():
            return

        artist_data = None
        did = _discogs_artistid(album)
        if did:
            artist_data = self._artist_by_id(did)

        if not artist_data and artist_name:
            artist_data = self._search_artist(artist_name)

        if not artist_data:
            self._log.debug("discogs: no artist found")
            return

        images = artist_data.get("images") or []

        # Sort: primary first, then by area descending.
        def _sort_key(img):
            primary = 0 if img.get("type") == "primary" else 1
            try:
                area = int(img.get("width") or 0) * int(img.get("height") or 0)
            except Exception:
                area = 0
            return (primary, -area)

        for img in sorted(images, key=_sort_key):
            url = img.get("uri") or img.get("resource_url")
            if url:
                size = None
                try:
                    w, h = img.get("width"), img.get("height")
                    if w and h:
                        size = (int(w), int(h))
                except Exception:
                    pass
                yield Candidate(url=url, source_name=self.NAME, size=size)


# All available sources, keyed by config name.
ALL_SOURCES: dict[str, type[ArtistArtSource]] = {
    "filesystem": FileSystemSource,
    "fanarttv": FanartTVSource,
    "theaudiodb": TheAudioDBSource,
    "spotify": SpotifySource,
    "wikidata": WikidataSource,
    "discogs": DiscogsSource,
}


# ---------------------------------------------------------------------------
#  Image processing helpers
# ---------------------------------------------------------------------------


def _download(url: str, log) -> Optional[str]:
    """Download an image URL to a temporary file.  Return the path."""
    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=_cfg()["request_timeout"].get(int),
            headers={"User-Agent": "beets-fetchartist/1.0"},
        )
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if ct and not ct.startswith("image/"):
            log.debug("fetchartist: not an image content-type: {}", ct)
            return None

        suffix = ".jpg"
        if "png" in ct:
            suffix = ".png"
        elif "webp" in ct:
            suffix = ".webp"
        elif "gif" in ct:
            suffix = ".gif"

        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix="fetchartist_",
            delete=False,
        )
        with closing(resp):
            for chunk in resp.iter_content(8192):
                tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception as exc:
        log.debug("fetchartist: download failed for {}: {}", url, exc)
        return None


def _image_size(path: str) -> Optional[tuple[int, int]]:
    """Return (width, height) using ArtResizer or Pillow."""
    try:
        size = ArtResizer.shared.get_size(path)
        if size and size != (0, 0):
            return size
    except Exception:
        pass
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    return None


def _check_constraints(path: str, log) -> bool:
    """Validate image against configured size / ratio constraints."""
    size = _image_size(path)
    if not size:
        log.debug("fetchartist: could not determine image size for {}", path)
        return True  # be lenient

    w, h = size
    minwidth = _cfg()["minwidth"].get(int)
    if minwidth and w < minwidth:
        log.debug("fetchartist: image too small ({}px < {}px min)", w, minwidth)
        return False

    max_filesize = _cfg()["max_filesize"].get(int)
    if max_filesize:
        try:
            fsize = os.path.getsize(path)
            if fsize > max_filesize:
                log.debug("fetchartist: file too large ({} > {})", fsize, max_filesize)
                return False
        except Exception:
            pass

    if _cfg()["enforce_ratio"].get(bool):
        if w and h:
            ratio = max(w, h) / max(min(w, h), 1)
            tolerance = _cfg()["ratio_tolerance"].get(float)
            if ratio > 1.0 + tolerance:
                log.debug("fetchartist: aspect ratio {:.2f} exceeds tolerance", ratio)
                return False

    return True


def _resize_and_convert(path: str, log) -> str:
    """Resize to maxwidth and/or convert to the configured format."""
    maxwidth = _cfg()["maxwidth"].get(int)
    cover_format = _cfg()["cover_format"].get() or None
    quality = _cfg()["quality"].get(int)

    # Resize via ArtResizer if needed.
    if maxwidth:
        size = _image_size(path)
        if size and size[0] > maxwidth:
            try:
                resized = ArtResizer.shared.resize(
                    maxwidth,
                    path,
                    quality=quality if quality else 0,
                )
                if resized and os.path.isfile(resized):
                    log.debug("fetchartist: resized {} -> {}px wide", path, maxwidth)
                    path = resized
            except Exception as exc:
                log.debug("fetchartist: resize failed: {}", exc)

    # Format conversion via Pillow.
    if cover_format:
        try:
            from PIL import Image

            target_fmt = cover_format.upper()
            if target_fmt == "JPG":
                target_fmt = "JPEG"
            with Image.open(path) as im:
                if im.mode in ("RGBA", "P") and target_fmt == "JPEG":
                    im = im.convert("RGB")
                ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
                    target_fmt, ".jpg"
                )
                out = tempfile.NamedTemporaryFile(
                    suffix=ext, prefix="fetchartist_conv_", delete=False
                )
                save_kwargs = {}
                if target_fmt == "JPEG" and quality:
                    save_kwargs["quality"] = quality
                im.save(out.name, target_fmt, **save_kwargs)
                log.debug("fetchartist: converted to {}", target_fmt)
                path = out.name
        except ImportError:
            log.debug("fetchartist: Pillow not available for format conversion")
        except Exception as exc:
            log.debug("fetchartist: format conversion failed: {}", exc)

    return path


# ---------------------------------------------------------------------------
#  Plugin
# ---------------------------------------------------------------------------


class FetchArtistPlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.config.add(
            {
                # Behaviour.
                "auto": True,
                "overwrite": False,
                "cautious": False,
                # Source priority (space-separated list, like fetchart).
                "sources": "filesystem fanarttv theaudiodb spotify wikidata discogs",
                # Output.
                "filename": "artist",  # saved as  <filename>.jpg
                "cover_names": ["artist"],  # filesystem source looks for these
                # Image constraints.
                "minwidth": 300,
                "maxwidth": 0,  # 0 = no limit
                "enforce_ratio": True,  # require ~square
                "ratio_tolerance": 0.5,  # how far from 1:1 is acceptable
                "cover_format": None,  # JPEG, PNG, WEBP  (None = keep original)
                "quality": 0,  # JPEG quality (0 = default)
                "max_filesize": 0,  # bytes (0 = no limit)
                # Source credentials.
                "fanarttv_key": None,
                "fanarttv_client_key": None,
                "theaudiodb_key": "2",  # free API key
                "spotify_client_id": None,
                "spotify_client_secret": None,
                "discogs_token": None,
                # Misc.
                "request_timeout": 15,
                "wikidata_search_limit": 5,
            }
        )

        # Redact secrets.
        for key in (
            "fanarttv_key",
            "fanarttv_client_key",
            "spotify_client_id",
            "spotify_client_secret",
            "discogs_token",
        ):
            self.config[key].redact = True

        # Session-level cache: tracks artist directories we have already
        # processed (fetched, skipped-because-exists, or tried-and-failed)
        # during the current import or CLI run.  Keyed by normalised artist
        # directory path.  Prevents redundant API lookups when importing
        # multiple albums by the same artist in one session.
        self._session_handled: set[str] = set()

        if self.config["auto"].get(bool):
            # Use the import_task_files event instead of import_stages.
            # import_stages run BEFORE file manipulation (copy/move), so
            # item.path still points at the source directory.
            # import_task_files fires AFTER the filesystem work is done –
            # files have been copied/moved and tags written, so item.path
            # is the final library location.
            self.register_listener("import_task_files", self._on_import_task_files)
            # Clear the session cache at the start of each import run.
            self.register_listener("import_begin", self._on_import_begin)

    # -- CLI ----------------------------------------------------------------

    def commands(self) -> list[Subcommand]:
        cmd = Subcommand(
            "fetchartist",
            help="download artist images from configured sources",
        )
        cmd.parser.add_option(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help="overwrite existing artist images",
        )
        cmd.func = self._command
        return [cmd]

    def _command(self, lib: Library, opts, args: list[str]) -> None:
        force = opts.force or self.config["overwrite"].get(bool)
        # Clear session cache for this CLI run so we track what we process.
        self._session_handled.clear()

        query = decargs(args)
        albums = list(lib.albums(query))
        if not albums:
            print_("No matching albums.")
            return

        for album in albums:
            self._fetch_for_album(album, force=force)

    # -- Import hook --------------------------------------------------------

    def _on_import_begin(self, session) -> None:
        """Clear the session cache at the start of each import run."""
        self._session_handled.clear()

    def _on_import_task_files(self, task, session) -> None:
        """Called after files have been moved/copied to the library.

        At this point item.path is the final destination, so we can
        safely derive the artist directory from the actual file locations.
        """
        try:
            items = task.imported_items()
        except Exception:
            return
        if not items:
            return

        for item in items:
            album = item.get_album()
            if not album:
                continue
            # _fetch_for_album handles deduplication via _session_handled.
            self._fetch_for_album(album, force=False)

    # -- Core logic ---------------------------------------------------------

    def _ordered_sources(self) -> list[ArtistArtSource]:
        """Instantiate sources in the configured order."""
        raw = self.config["sources"].get(str)
        names = raw.split()
        sources = []
        for name in names:
            cls = ALL_SOURCES.get(name.lower())
            if cls:
                sources.append(cls(self._log))
            else:
                self._log.warning("fetchartist: unknown source: {}", name)
        return sources

    def _fetch_for_album(self, album, *, force: bool = False) -> bool:
        """Try to fetch artist art for the given album's albumartist.

        Returns True if art was saved.
        """
        artist_name = getattr(album, "albumartist", None) or ""
        if not artist_name:
            return False

        adir = _artist_dir(album)
        if not adir:
            self._log.debug(
                "fetchartist: cannot determine artist dir for {}", artist_name
            )
            return False

        # --- Session-level deduplication -----------------------------------
        # If we have already processed this artist directory during the
        # current import/CLI session, bail out immediately.  This covers:
        #   • Artist already had art on disk (skipped)
        #   • We fetched art on a previous album by the same artist
        #   • We tried and failed (no art found) — don't re-search
        #   • force/overwrite: we already overwrote once, don't redo
        cache_key = os.path.normpath(adir)
        if cache_key in self._session_handled:
            self._log.debug(
                "fetchartist: already handled {} this session, skipping",
                artist_name,
            )
            return False
        # Mark as handled *now* — before any network calls — so that even
        # if we're called concurrently for another album by the same artist
        # we won't duplicate work.
        self._session_handled.add(cache_key)

        filename = self.config["filename"].get(str)
        # Check if art already exists on disk.
        if not force:
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                existing = os.path.join(adir, f"{filename}{ext}")
                if os.path.isfile(existing):
                    self._log.info(
                        "fetchartist: {} already has art: {}",
                        artist_name,
                        existing,
                    )
                    return False

        mb_id = _mb_artistid(album)
        self._log.info(
            "fetchartist: fetching art for {} (mb_artistid={})",
            artist_name,
            mb_id or "none",
        )

        sources = self._ordered_sources()

        for source in sources:
            self._log.debug(
                "fetchartist: trying source {} for {}",
                source.NAME,
                artist_name,
            )
            try:
                candidates = source.get(
                    artist_name=artist_name,
                    mb_artistid=mb_id,
                    album=album,
                )
            except Exception as exc:
                self._log.debug("fetchartist: source {} error: {}", source.NAME, exc)
                continue

            for candidate in candidates:
                path = candidate.path

                # Download remote candidates.
                if not path and candidate.url:
                    path = _download(candidate.url, self._log)
                if not path or not os.path.isfile(path):
                    continue

                # Validate constraints.
                if not _check_constraints(path, self._log):
                    if not candidate.path:
                        _safe_remove(path)
                    continue

                # Resize / convert.
                processed = _resize_and_convert(path, self._log)

                # Determine final extension from the processed file.
                _, ext = os.path.splitext(processed)
                if not ext:
                    ext = ".jpg"
                dest = os.path.join(adir, f"{filename}{ext}")

                try:
                    os.makedirs(adir, exist_ok=True)
                    shutil.copy2(processed, dest)
                    self._log.info(
                        "fetchartist: saved {} art -> {} (via {})",
                        artist_name,
                        dest,
                        candidate.source_name,
                    )
                except Exception as exc:
                    self._log.error("fetchartist: failed to save {}: {}", dest, exc)
                    continue
                finally:
                    # Clean up temp files (but not if it was the filesystem source).
                    if processed != candidate.path:
                        _safe_remove(processed)
                    if path != candidate.path and path != processed:
                        _safe_remove(path)

                return True

        self._log.info("fetchartist: no art found for {}", artist_name)
        return False


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass

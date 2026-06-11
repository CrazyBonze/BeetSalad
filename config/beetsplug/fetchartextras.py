from __future__ import annotations

import re
import unicodedata
from typing import Iterator, Optional

import requests

from beets import config as beets_config
from beets.plugins import BeetsPlugin


class FetchArtExtrasPlugin(BeetsPlugin):
    """Monkey-patch beetsplug.fetchart with safer extra sources.

    Put this plugin BEFORE fetchart in your beets config:

        plugins: fetchartextras fetchart

    Added sources:
      - discogs[id, search]
      - deezer[id, search]
      - wikidata[default]
      - itunesplus[default]

    Also patches the stock `itunes` source to disable the unsafe first-result
    fallback and use normalized exact matching only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config.add(
            {
                "discogs_token": None,
                "discogs_user_agent": "beets/fetchartextras (+https://beets.io)",
                "discogs_search_limit": 10,
                "discogs_strict_search": True,
                "deezer_search_limit": 10,
                "wikidata_limit": 8,
                "itunes_strict_year": True,
                "itunes_search_limit": 50,
                "debug_album_fields": False,
            }
        )
        self.config["discogs_token"].redact = True
        self._patch_fetchart()

    def _patch_fetchart(self) -> None:
        from beetsplug import fetchart as fa

        def _extras():
            return beets_config["fetchartextras"]

        def _norm(value: str | None) -> str:
            if not value:
                return ""
            value = unicodedata.normalize("NFKC", str(value)).casefold()
            replacements = {
                "’": "'",
                "‘": "'",
                "“": '"',
                "”": '"',
                "&": " and ",
            }
            for old, new in replacements.items():
                value = value.replace(old, new)
            value = re.sub(r"[^\w\s]+", " ", value)
            value = re.sub(r"\s+", " ", value).strip()
            return value

        def _album_item(album):
            try:
                return album.items().get()
            except Exception:
                return None

        def _field_from_obj(obj, name: str):
            try:
                value = obj.get(name)
            except Exception:
                value = getattr(obj, name, None)
            return value

        def _field(album, *names: str):
            item = _album_item(album)
            for name in names:
                value = _field_from_obj(album, name)
                if value not in (None, "", 0, "0"):
                    return value
                if item is not None:
                    value = _field_from_obj(item, name)
                    if value not in (None, "", 0, "0"):
                        return value
            return None

        def _int_field(album, *names: str) -> Optional[int]:
            value = _field(album, *names)
            if value in (None, "", 0, "0"):
                return None
            try:
                return int(str(value))
            except Exception:
                return None

        def _artist_match(left: str | None, right: str | None) -> bool:
            ln = _norm(left)
            rn = _norm(right)
            if not ln or not rn:
                return False
            if ln == rn:
                return True
            va_aliases = {"various artists", "various", "va"}
            return ln in va_aliases and rn in va_aliases

        def _title_exact(left: str | None, right: str | None) -> bool:
            ln = _norm(left)
            rn = _norm(right)
            return bool(ln and rn and ln == rn)

        def _title_soft(left: str | None, right: str | None) -> bool:
            ln = _norm(left)
            rn = _norm(right)
            if not ln or not rn:
                return False
            return ln == rn or ln in rn or rn in ln

        def _year_match(candidate_year, album_year) -> bool:
            if not _extras()["itunes_strict_year"].get(bool):
                return True
            if not candidate_year or not album_year:
                return True
            try:
                return abs(int(candidate_year) - int(album_year)) <= 1
            except Exception:
                return True

        def _debug_album(source, album) -> None:
            if not _extras()["debug_album_fields"].get(bool):
                return

            album_fields = {
                "album": getattr(album, "album", None),
                "albumartist": getattr(album, "albumartist", None),
                "year": getattr(album, "year", None),
                "mb_albumid": _field_from_obj(album, "mb_albumid"),
                "mb_releasegroupid": _field_from_obj(album, "mb_releasegroupid"),
                "discogs_album_id": _field_from_obj(album, "discogs_album_id"),
                "deezer_album_id": _field_from_obj(album, "deezer_album_id"),
                "spotify_album_id": _field_from_obj(album, "spotify_album_id"),
                "asin": _field_from_obj(album, "asin"),
            }
            source._log.debug("album debug: {}", album_fields)

        def _discogs_release_id(source, album) -> Optional[int]:
            # 1) Prefer album-level fields from musicbrainz external_ids /
            #    discogs-related plugins.
            rid = _int_field(
                album,
                "discogs_album_id",
                "discogs_albumid",
                "discogs_release_id",
                "discogs_releaseid",
            )
            if rid:
                source._log.debug("discogs: resolved album-level release id {}", rid)
                return rid

            # 2) Fall back to item-level consensus. This protects against grabbing
            #    the first item's ID when the query accidentally spans multiple
            #    releases/albums.
            try:
                items = list(album.items())
            except Exception:
                items = []

            wanted_album = _norm(getattr(album, "album", None))
            counts: dict[int, int] = {}

            for item in items:
                item_album = _norm(getattr(item, "album", None))
                if wanted_album and item_album and item_album != wanted_album:
                    continue

                rid = None
                for field_name in (
                    "discogs_album_id",
                    "discogs_albumid",
                    "discogs_release_id",
                    "discogs_releaseid",
                ):
                    value = _field_from_obj(item, field_name)
                    if value in (None, "", 0, "0"):
                        continue
                    try:
                        rid = int(str(value))
                        break
                    except Exception:
                        continue

                if rid:
                    counts[rid] = counts.get(rid, 0) + 1

            if not counts:
                source._log.debug(
                    "discogs: no album-level or item-level release id found"
                )
                return None

            source._log.debug("discogs: item-level release id counts {}", counts)

            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                source._log.debug(
                    "discogs: resolved item-level release id {}", ranked[0][0]
                )
                return ranked[0][0]

            source._log.debug(
                "discogs: ambiguous item-level release ids; skipping exact id mode"
            )
            return None

        def _itunes_results(source, album) -> list[dict]:
            if not album.albumartist or not album.album:
                return []
            payload = {
                "term": f"{album.albumartist} {album.album}",
                "entity": "album",
                "media": "music",
                "limit": _extras()["itunes_search_limit"].get(int),
            }
            try:
                response = source.request(ITunesPlus.API_URL, params=payload)
                response.raise_for_status()
                results = response.json().get("results", [])
                source._log.debug("itunesplus: {} search results", len(results))
                return results
            except (requests.RequestException, ValueError, AttributeError) as exc:
                source._log.debug("itunesplus: search failed: {}", exc)
                return []

        def _itunes_candidates(source, album) -> Iterator[fa.Candidate]:
            image_suffix = (
                "100000x100000-999"
                if source._config["high_resolution"].get(bool)
                else "1200x1200bb"
            )
            album_year = _int_field(album, "year")

            for result in _itunes_results(source, album):
                cand_artist = result.get("artistName")
                cand_title = result.get("collectionName")
                release_date = result.get("releaseDate") or ""
                cand_year = None
                if release_date:
                    try:
                        cand_year = int(str(release_date)[:4])
                    except Exception:
                        cand_year = None

                artist_ok = _artist_match(cand_artist, album.albumartist)
                title_ok = _title_exact(cand_title, album.album)
                year_ok = _year_match(cand_year, album_year)

                source._log.debug(
                    "itunesplus candidate: artist={!r} title={!r} year={} -> artist_ok={} title_ok={} year_ok={}",
                    cand_artist,
                    cand_title,
                    cand_year,
                    artist_ok,
                    title_ok,
                    year_ok,
                )

                if not (artist_ok and title_ok and year_ok):
                    continue

                art_url = result.get("artworkUrl100")
                if not art_url:
                    continue

                yield source._candidate(
                    url=str(art_url).replace("100x100bb", image_suffix),
                    match=fa.MetadataMatch.EXACT,
                )

        class DiscogsArt(fa.RemoteArtSource):
            NAME = "Discogs"
            ID = "discogs"
            VALID_MATCHING_CRITERIA = ["id", "search"]
            API_BASE = "https://api.discogs.com"

            @staticmethod
            def add_default_config(config):
                config.add({})

            @classmethod
            def available(cls, log, _config) -> bool:
                token = _extras()["discogs_token"].get()
                if not token:
                    log.debug("discogs: Disabling art source due to missing token")
                    return False
                return True

            def _headers(self) -> dict[str, str]:
                headers = {
                    "User-Agent": _extras()["discogs_user_agent"].get(str),
                    "Accept": "application/json",
                }
                token = _extras()["discogs_token"].get(str)
                if token:
                    headers["Authorization"] = f"Discogs token={token}"
                return headers

            def _release(self, release_id: int) -> Optional[dict]:
                try:
                    response = self.request(
                        f"{self.API_BASE}/releases/{release_id}",
                        headers=self._headers(),
                    )
                    response.raise_for_status()
                    return response.json()
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug(
                        "discogs: release lookup failed for {}: {}",
                        release_id,
                        exc,
                    )
                    return None

            def _yield_images(self, payload: dict, match) -> Iterator[fa.Candidate]:
                images = payload.get("images") or []
                if not isinstance(images, list):
                    self._log.debug("discogs: release payload had no images list")
                    return

                def sort_key(img: dict) -> tuple[int, int]:
                    primary = 0 if img.get("type") == "primary" else 1
                    try:
                        area = int(img.get("width") or 0) * int(img.get("height") or 0)
                    except Exception:
                        area = 0
                    return (primary, -area)

                yielded = 0
                for img in sorted(images, key=sort_key):
                    if (img.get("type") or "").lower() == "secondary":
                        self._log.debug(
                            "discogs: skipping secondary image (type=secondary)"
                        )
                        continue
                    url = img.get("uri") or img.get("resource_url")
                    if not url:
                        continue
                    size = None
                    try:
                        if img.get("width") and img.get("height"):
                            size = (int(img["width"]), int(img["height"]))
                    except Exception:
                        size = None
                    yielded += 1
                    yield self._candidate(url=url, match=match, size=size)

                self._log.debug("discogs: yielded {} image candidates", yielded)

            def _search_release_ids(self, album) -> Iterator[tuple[int, object]]:
                if not album.albumartist or not album.album:
                    return

                params = {
                    "type": "release",
                    "release_title": album.album,
                    "artist": album.albumartist,
                    "per_page": _extras()["discogs_search_limit"].get(int),
                    "page": 1,
                }
                year = _int_field(album, "year")
                if year:
                    params["year"] = year

                try:
                    response = self.request(
                        f"{self.API_BASE}/database/search",
                        params=params,
                        headers=self._headers(),
                    )
                    response.raise_for_status()
                    data = response.json()
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug("discogs: search failed: {}", exc)
                    return

                strict = _extras()["discogs_strict_search"].get(bool)
                results = data.get("results") or []
                self._log.debug("discogs: {} search results", len(results))

                for result in results:
                    rid = result.get("id")
                    if not rid:
                        continue

                    title = result.get("title") or ""
                    parts = [p.strip() for p in title.split(" - ", 1)]
                    cand_artist = parts[0] if len(parts) == 2 else result.get("artist")
                    cand_title = parts[1] if len(parts) == 2 else result.get("title")

                    artist_ok = _artist_match(cand_artist, album.albumartist)
                    exact_title_ok = _title_exact(cand_title, album.album)
                    soft_title_ok = _title_soft(cand_title, album.album)

                    result_year = result.get("year")
                    year_ok = True
                    if year and result_year:
                        try:
                            year_ok = abs(int(result_year) - int(year)) <= 1
                        except Exception:
                            year_ok = True

                    self._log.debug(
                        "discogs candidate: id={} title={!r} year={} -> artist_ok={} exact_title_ok={} soft_title_ok={} year_ok={}",
                        rid,
                        title,
                        result_year,
                        artist_ok,
                        exact_title_ok,
                        soft_title_ok,
                        year_ok,
                    )

                    if artist_ok and exact_title_ok and year_ok:
                        yield int(rid), fa.MetadataMatch.EXACT
                    elif not strict and artist_ok and soft_title_ok:
                        yield int(rid), fa.MetadataMatch.FALLBACK

            def get(self, album, plugin, paths) -> Iterator[fa.Candidate]:
                _debug_album(self, album)
                self._log.debug(
                    "discogs[{}]: album={} - {}",
                    ",".join(self.match_by),
                    getattr(album, "albumartist", ""),
                    getattr(album, "album", ""),
                )

                if "id" in self.match_by:
                    release_id = _discogs_release_id(self, album)
                    if release_id:
                        payload = self._release(release_id)
                        if payload:
                            yield from self._yield_images(
                                payload, fa.MetadataMatch.EXACT
                            )
                            return

                if "search" in self.match_by:
                    seen: set[int] = set()
                    for rid, match in self._search_release_ids(album):
                        if rid in seen:
                            continue
                        seen.add(rid)
                        payload = self._release(rid)
                        if not payload:
                            continue
                        yielded_any = False
                        for candidate in self._yield_images(payload, match):
                            yielded_any = True
                            yield candidate
                        if yielded_any and match == fa.MetadataMatch.EXACT:
                            return

        class DeezerArt(fa.RemoteArtSource):
            NAME = "Deezer"
            ID = "deezer"
            VALID_MATCHING_CRITERIA = ["id", "search"]
            API_BASE = "https://api.deezer.com"

            @staticmethod
            def add_default_config(config):
                config.add({})

            def _yield_cover(self, payload: dict, match) -> Iterator[fa.Candidate]:
                for key in ("cover_xl", "cover_big", "cover_medium", "cover"):
                    url = payload.get(key)
                    if url:
                        yield self._candidate(url=url, match=match)
                        return

            def _lookup_album(self, album_id: int) -> Optional[dict]:
                try:
                    response = self.request(f"{self.API_BASE}/album/{album_id}")
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict) and data.get("error"):
                        return None
                    return data
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug(
                        "deezer: album lookup failed for {}: {}", album_id, exc
                    )
                    return None

            def _matches_payload(
                self, payload: dict, album, soft: bool = False
            ) -> bool:
                payload_artist = (payload.get("artist") or {}).get("name")
                payload_title = payload.get("title") or payload.get("title_short")
                artist_ok = _artist_match(payload_artist, album.albumartist)
                title_ok = (
                    _title_soft(payload_title, album.album)
                    if soft
                    else _title_exact(payload_title, album.album)
                )
                return artist_ok and title_ok

            def _search_payloads(self, album) -> Iterator[tuple[dict, object]]:
                if not album.albumartist or not album.album:
                    return
                query = f'artist:"{album.albumartist}" album:"{album.album}"'
                try:
                    response = self.request(
                        f"{self.API_BASE}/search/album",
                        params={
                            "q": query,
                            "limit": _extras()["deezer_search_limit"].get(int),
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug("deezer: search failed: {}", exc)
                    return

                results = data.get("data") or []
                self._log.debug("deezer: {} search results", len(results))

                for payload in results:
                    payload_artist = (payload.get("artist") or {}).get("name")
                    payload_title = payload.get("title") or payload.get("title_short")
                    exact_ok = self._matches_payload(payload, album, soft=False)
                    soft_ok = self._matches_payload(payload, album, soft=True)

                    self._log.debug(
                        "deezer candidate: id={} artist={!r} title={!r} -> exact_ok={} soft_ok={}",
                        payload.get("id"),
                        payload_artist,
                        payload_title,
                        exact_ok,
                        soft_ok,
                    )

                    if exact_ok:
                        yield payload, fa.MetadataMatch.EXACT
                    elif soft_ok:
                        yield payload, fa.MetadataMatch.FALLBACK

            def get(self, album, plugin, paths) -> Iterator[fa.Candidate]:
                _debug_album(self, album)
                self._log.debug(
                    "deezer[{}]: album={} - {}",
                    ",".join(self.match_by),
                    getattr(album, "albumartist", ""),
                    getattr(album, "album", ""),
                )

                if "id" in self.match_by:
                    album_id = _int_field(album, "deezer_album_id")
                    if album_id:
                        self._log.debug("deezer: using exact album id {}", album_id)
                        payload = self._lookup_album(album_id)
                        if payload:
                            yield from self._yield_cover(
                                payload, fa.MetadataMatch.EXACT
                            )
                            return

                if "search" in self.match_by:
                    for payload, match in self._search_payloads(album):
                        for candidate in self._yield_cover(payload, match):
                            yield candidate
                        if match == fa.MetadataMatch.EXACT:
                            return

        class WikidataArt(fa.RemoteArtSource):
            NAME = "Wikidata / Wikimedia Commons"
            ID = "wikidata"
            SEARCH_URL = "https://www.wikidata.org/w/api.php"
            COMMONS_URL = "https://commons.wikimedia.org/w/api.php"

            @staticmethod
            def add_default_config(config):
                config.add({})

            def _search_entities(self, query: str) -> list[str]:
                try:
                    response = self.request(
                        self.SEARCH_URL,
                        params={
                            "action": "wbsearchentities",
                            "format": "json",
                            "language": "en",
                            "type": "item",
                            "limit": _extras()["wikidata_limit"].get(int),
                            "search": query,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug("wikidata: entity search failed: {}", exc)
                    return []
                results = [r.get("id") for r in data.get("search") or [] if r.get("id")]
                self._log.debug(
                    "wikidata: query={!r} -> {} entities", query, len(results)
                )
                return results

            def _entity_image_names(self, entity_ids: list[str]) -> Iterator[str]:
                if not entity_ids:
                    return
                try:
                    response = self.request(
                        self.SEARCH_URL,
                        params={
                            "action": "wbgetentities",
                            "format": "json",
                            "props": "claims",
                            "ids": "|".join(entity_ids),
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug("wikidata: entity fetch failed: {}", exc)
                    return

                for entity in (data.get("entities") or {}).values():
                    for image_claim in (entity.get("claims") or {}).get("P18") or []:
                        try:
                            yield image_claim["mainsnak"]["datavalue"]["value"]
                        except Exception:
                            continue

            def _commons_image_url(self, filename: str) -> Optional[str]:
                title = (
                    filename
                    if str(filename).startswith("File:")
                    else f"File:{filename}"
                )
                try:
                    response = self.request(
                        self.COMMONS_URL,
                        params={
                            "action": "query",
                            "format": "json",
                            "prop": "imageinfo",
                            "iiprop": "url",
                            "titles": title,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                except (requests.RequestException, ValueError) as exc:
                    self._log.debug(
                        "wikidata: commons image lookup failed for {}: {}", title, exc
                    )
                    return None

                for page in (data.get("query") or {}).get("pages", {}).values():
                    try:
                        return page["imageinfo"][0]["url"]
                    except Exception:
                        continue
                return None

            def get(self, album, plugin, paths) -> Iterator[fa.Candidate]:
                _debug_album(self, album)

                queries = []
                if album.albumartist and album.album:
                    queries.append(f"{album.album} {album.albumartist}")
                if album.album:
                    queries.append(album.album)

                seen: set[str] = set()
                for query in queries:
                    entity_ids = self._search_entities(query)
                    for image_name in self._entity_image_names(entity_ids):
                        image_url = self._commons_image_url(image_name)
                        if not image_url or image_url in seen:
                            continue
                        seen.add(image_url)
                        yield self._candidate(
                            url=image_url, match=fa.MetadataMatch.FALLBACK
                        )

        class ITunesPlus(fa.RemoteArtSource):
            NAME = "iTunes Store (safe)"
            ID = "itunesplus"
            API_URL = "https://itunes.apple.com/search"

            @staticmethod
            def add_default_config(config):
                config.add({})

            def get(self, album, plugin, paths) -> Iterator[fa.Candidate]:
                _debug_album(self, album)
                yield from _itunes_candidates(self, album)

        def _patched_itunes_get(self, album, plugin, paths):
            _debug_album(self, album)
            yield from _itunes_candidates(self, album)

        # Patch stock iTunes to use the safer implementation too.
        fa.ITunesStore.get = _patched_itunes_get

        # Register/replace sources in fetchart's class registry.
        sources = {
            cls
            for cls in fa.ART_SOURCES
            if cls.ID not in {"discogs", "deezer", "wikidata", "itunesplus"}
        }
        sources.update({DiscogsArt, DeezerArt, WikidataArt, ITunesPlus})
        fa.ART_SOURCES = sources

        # Patch FetchArtPlugin.__init__ to inject custom sources into the
        # resolved self.sources list after fetchart builds it from config.
        # This is needed because beets 2.7+ validates sources against
        # ART_SOURCES at __init__ time, before our new classes are visible
        # to the sanitize_pairs validator (which rejects unknown names).
        #
        # Insertion order mirrors the config.yaml sources list:
        #   ..., lastfm, discogs:id, deezer:id, fanarttv, ...
        #   ..., coverart:releasegroup, discogs:search, deezer:search,
        #        wikidata, itunesplus, itunes
        _orig_fa_init = fa.FetchArtPlugin.__init__

        def _patched_fa_init(fa_self, *args, **kwargs):
            _orig_fa_init(fa_self, *args, **kwargs)
            log = fa_self._log

            def _make(cls, criteria):
                return cls(log, fa_self.config, match_by=[criteria])

            # Build the injected source instances.
            injected = [
                _make(DiscogsArt, "id"),
                _make(DeezerArt, "id"),
                _make(DiscogsArt, "search"),
                _make(DeezerArt, "search"),
                _make(WikidataArt, "default"),
                _make(ITunesPlus, "default"),
            ]

            # Insert discogs:id and deezer:id right after lastfm (index after
            # the last lastfm entry), and the rest before itunes at the tail.
            # Simpler: splice them in at the documented positions.
            new_sources = []
            tail = []
            for src in fa_self.sources:
                sid = src.__class__.ID
                if sid == "itunes":
                    # Insert tail extras just before itunes.
                    tail.append(src)
                else:
                    new_sources.append(src)

            # Find position after lastfm to insert id-based sources.
            insert_after_lastfm = len(new_sources)
            for i, src in enumerate(new_sources):
                if src.__class__.ID == "lastfm":
                    insert_after_lastfm = i + 1
                    break

            id_sources = injected[:2]  # discogs:id, deezer:id
            rest_sources = injected[
                2:
            ]  # discogs:search, deezer:search, wikidata, itunesplus

            new_sources[insert_after_lastfm:insert_after_lastfm] = id_sources
            fa_self.sources = new_sources + rest_sources + tail

            log.debug(
                "fetchartextras: injected sources: {}",
                [f"{s.__class__.ID}:{s.match_by}" for s in injected],
            )

        fa.FetchArtPlugin.__init__ = _patched_fa_init

        self._log.debug(
            "registered fetchart sources: discogs[id/search], deezer[id/search], wikidata, itunesplus"
        )

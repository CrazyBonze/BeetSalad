"""Assigns genres based on MusicBrainz release type (albumtype/albumtypes).

For each type found in `albumtypes`, if the type is configured in the `map`,
the corresponding genre string is appended to the album's genre field.

Configuration example:

    typetogenre:
        auto: yes
        map:
            soundtrack: yes         # uses default display name "Soundtrack"
            live: yes               # uses default display name "Live"
            spokenword: Speech      # overrides default "Spoken Word" -> "Speech"
            dj-mix: yes             # uses default display name "DJ-Mix"
            album: no               # explicitly disabled (same as omitting)

Plugin ordering note:
    This plugin hooks into `album_imported` rather than `import_stages`, so it
    always runs after lastgenre (which uses import_stages). This ensures
    type-derived genres are appended after lastgenre has done its work and
    won't be overwritten by it. For the CLI command, run `beet typetogenre`
    after `beet lastgenre` if using both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beets import ui
from beets.plugins import BeetsPlugin

if TYPE_CHECKING:
    import optparse

    from beets.library import Album, Library


# Types whose display names cannot be derived from simple title-casing.
# Everything else falls through to str.title().
TYPE_DISPLAY_EXCEPTIONS: dict[str, str] = {
    "ep": "EP",
    "dj-mix": "DJ-Mix",
    "spokenword": "Spoken Word",
    "mixtape": "Mixtape/Street",
    "audio drama": "Audio Drama",
    "field recording": "Field Recording",
}


def _default_display_name(type_key: str) -> str:
    """Return the canonical display name for a release type key.

    Falls back to title-casing the key for any unknown/future types,
    with a small exceptions dict for types whose display names can't be
    derived that way.
    """
    return TYPE_DISPLAY_EXCEPTIONS.get(type_key.lower(), type_key.title())


def _resolve_genre_string(type_key: str, config_value) -> str | None:
    """Given a type key and its config value, return the genre string to use.

    - False / "no" / None  -> skip this type entirely, return None
    - True  / "yes"        -> use the canonical display name
    - any other string     -> use that string as-is
    """
    if config_value is None:
        return None

    # confuse parses bare `yes`/`no` as booleans
    if isinstance(config_value, bool):
        return _default_display_name(type_key) if config_value else None

    # String values: "yes" means use default, anything else is an override
    val = str(config_value).strip()
    if val.lower() in ("yes", "true", "1"):
        return _default_display_name(type_key)
    if val.lower() in ("no", "false", "0", ""):
        return None

    return val  # explicit override string


class TypeToGenrePlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()

        self.config.add(
            {
                "auto": True,
                "map": {},
            }
        )

        if self.config["auto"].get(bool):
            self.register_listener("album_imported", self.album_imported)

    # --- core logic ---

    def _genres_for_album(self, album: Album) -> list[str]:
        """Return the list of genre strings that should be appended to this
        album based on its albumtypes and the configured map.
        """
        # albumtypes is a list field; fall back to albumtype if unavailable
        albumtypes: list[str] = album.albumtypes or []
        if not albumtypes and album.albumtype:
            albumtypes = [album.albumtype]

        if not albumtypes:
            self._log.debug("No albumtypes found for album: {}", album)
            return []

        map_config = self.config["map"].get(dict) or {}
        genres_to_add: list[str] = []

        for type_key in albumtypes:
            type_key = type_key.lower().strip()
            if type_key not in map_config:
                self._log.debug("Type '{}' not in typetogenre map, skipping.", type_key)
                continue

            genre_str = _resolve_genre_string(type_key, map_config[type_key])
            if genre_str:
                genres_to_add.append(genre_str)
                self._log.debug("Type '{}' -> genre '{}'", type_key, genre_str)

        return genres_to_add

    def _apply_genres(self, album: Album, write: bool) -> None:
        """Append type-derived genres to the album, skipping any already
        present to avoid duplicates on re-runs.
        """
        genres_to_add = self._genres_for_album(album)
        if not genres_to_add:
            return

        # beets >= 2.7 stores genres as a list in the ``genres`` field.
        # Earlier versions used a comma-separated string in ``genre``.
        raw = album.get("genres")
        if isinstance(raw, list):
            existing: list[str] = list(raw)
        elif isinstance(raw, str) and raw:
            existing = [
                g.strip() for g in raw.replace(";", ",").split(",") if g.strip()
            ]
        else:
            # Fall back to the legacy singular field if present.
            legacy = album.get("genre") or ""
            existing = [
                g.strip() for g in legacy.replace(";", ",").split(",") if g.strip()
            ]

        existing_lower = {g.lower() for g in existing}

        added: list[str] = []
        for genre in genres_to_add:
            if genre.lower() not in existing_lower:
                existing.append(genre)
                existing_lower.add(genre.lower())
                added.append(genre)

        if not added:
            self._log.debug(
                "All type genres already present for '{}', nothing to do.",
                album,
            )
            return

        # Write back to whichever field is active.
        if album.get("genres") is not None or not hasattr(album, "genre"):
            album.genres = existing  # beets >= 2.7
        else:
            album.genre = ", ".join(existing)  # beets < 2.7 fallback

        self._log.info(
            "Appended type genre(s) {} to '{}' -> '{}'",
            added,
            album,
            existing,
        )
        album.try_sync(write=write, move=False)

    # --- hooks ---

    def album_imported(self, lib: Library, album: Album) -> None:
        """Hook: fires after all import stages (including lastgenre) complete.

        Note: album_imported receives (lib, album) directly, unlike the
        `imported` hook which receives (session, task).
        """
        self._apply_genres(album, write=ui.should_write())

    # --- CLI command ---

    def commands(self) -> list[ui.Subcommand]:
        cmd = ui.Subcommand(
            "typetogenre",
            help="append release-type derived genres to albums",
        )
        cmd.parser.add_option(
            "-p",
            "--pretend",
            action="store_true",
            default=False,
            help="show changes without writing them",
        )

        def func(lib: Library, opts: optparse.Values, args: list[str]) -> None:
            write = ui.should_write() and not opts.pretend
            for album in lib.albums(args):
                self._apply_genres(album, write=write)

        cmd.func = func
        return [cmd]

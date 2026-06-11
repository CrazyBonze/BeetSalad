"""Primary Artist plugin for beets.

Extracts the first/primary artist from multi-artist album credits and
stores it as a flexible attribute. This ensures that collaboration albums
(e.g. "Apocalyptica feat. Corey Taylor") get filed under the primary
artist's folder (e.g. "Apocalyptica/"), producing a folder structure
compatible with Lidarr and other media managers.

Discovery uses beets' built-in multi-valued fields which come directly
from structured MusicBrainz data:

  - ``album.albumartists``     : list of canonical artist names
  - ``album.mb_albumartistids``: list of artist MBIDs

These fields are populated by the MusicBrainz plugin from the raw
``artist-credit`` block — no string parsing is needed.  A single-entity
band like "Florence and the Machine" has one entry in these lists and
is never misidentified as a collaboration.

The plugin provides:
  - A template field ``$primary_albumartist`` for use in path formats.
  - Automatic extraction during import.
  - A CLI command ``beet primaryartist`` to retroactively fix libraries.

Usage in beets config::

    plugins: primaryartist

    paths:
        default: $primary_albumartist/$album%aunique{}/$track $title

    primaryartist:
        auto: yes
        preferred_artists: []     # future: override list
"""


# ---------------------------------------------------------------------------
# Plugin class (requires beets)
# ---------------------------------------------------------------------------

try:
    from beets import ui
    from beets.plugins import BeetsPlugin
except ImportError:
    BeetsPlugin = None

if BeetsPlugin is not None:

    def _get_primary(model):
        """Extract the primary artist name and MBID from a beets
        model (Album or Item) using the multi-valued fields that
        come directly from MusicBrainz structured data.

        Always returns ``(name, mbid)`` — for solo artists this will
        be the full artist name (same as albumartist). Returns
        ``(None, None)`` only if no usable data is found.
        """
        for names_field, ids_field in [
            ("albumartists", "mb_albumartistids"),
            ("artists", "mb_artistids"),
        ]:
            val = getattr(model, names_field, None)
            if val and isinstance(val, list) and len(val) >= 1:
                name = val[0]
                ids = getattr(model, ids_field, None)
                mbid = ids[0] if ids and isinstance(ids, list) and ids else None
                return name, mbid

        # No multi-value fields available — fall back to scalar.
        name = getattr(model, "albumartist", None) or getattr(model, "artist", None)
        mbid = getattr(model, "mb_albumartistid", None) or getattr(
            model, "mb_artistid", None
        )
        return name, mbid

    def _tmpl_primary_albumartist(album):
        """Album-level template field ``$primary_albumartist``."""
        flex = getattr(album, "_values_flex", {})
        val = flex.get("primary_albumartist", "")
        if val:
            return val
        name, _ = _get_primary(album)
        return name or album.albumartist

    def _tmpl_primary_albumartist_item(item):
        """Item-level template field ``$primary_albumartist``."""
        flex = getattr(item, "_values_flex", {})
        val = flex.get("primary_albumartist", "")
        if val:
            return val
        name, _ = _get_primary(item)
        return name or item.albumartist or item.artist

    class PrimaryArtistPlugin(BeetsPlugin):
        def __init__(self):
            super().__init__()

            self.config.add(
                {
                    "auto": True,
                    "preferred_artists": [],
                }
            )

            self.album_template_fields["primary_albumartist"] = (
                _tmpl_primary_albumartist
            )
            self.template_fields["primary_albumartist"] = _tmpl_primary_albumartist_item

            if self.config["auto"].get(bool):
                self.register_listener(
                    "import_task_files",
                    self._on_import_task_files,
                )
                self.register_listener("item_imported", self._on_item_imported)

        # ---- import events ------------------------------------------------

        def _on_import_task_files(self, task, session):
            """Called after all filesystem ops (copy/move/write) and
            reimport preservation are complete.

            This is the only safe place to set flex attrs during
            import because reimport preservation has already restored
            old flex attrs by this point — we overwrite with the
            correct value. Files are in their final location and
            artpath points to the actual cover file.
            """
            import os

            album = getattr(task, "album", None)
            if album is None:
                return

            # Reload to get the latest state from the database.
            album.load()

            name, mbid = self._resolve(album)
            if not name:
                return

            album["primary_albumartist"] = name
            if mbid:
                album["primary_albumartistid"] = mbid
            album.store()

            for item in album.items():
                item["primary_albumartist"] = name
                if mbid:
                    item["primary_albumartistid"] = mbid
                item.store()

            self._log.info(
                "primaryartist: {0} -> {1}",
                album.albumartist,
                name,
            )

            # Re-fire art_set for the thumbnails plugin.
            artpath = album.artpath
            if artpath and os.path.exists(artpath):
                from beets import plugins as beets_plugins

                beets_plugins.send(
                    "art_set",
                    album=album,
                    artpath=artpath,
                )

        def _on_item_imported(self, lib, item):
            """Handle singleton imports."""
            name, mbid = self._resolve(item)
            if not name:
                return

            item["primary_albumartist"] = name
            if mbid:
                item["primary_albumartistid"] = mbid
            item.store()

        # ---- CLI command --------------------------------------------------

        def commands(self):
            cmd = ui.Subcommand(
                "primaryartist",
                help="set primary_albumartist on existing albums",
            )
            cmd.parser.add_option(
                "-p",
                "--pretend",
                action="store_true",
                default=False,
                help="show changes without applying them",
            )
            cmd.parser.add_option(
                "-d",
                "--debug",
                action="store_true",
                default=False,
                help="dump all relevant fields for matching albums",
            )
            cmd.func = self._cmd_primaryartist
            return [cmd]

        def _cmd_primaryartist(self, lib, opts, args):
            query = ui.decargs(args)
            albums = lib.albums(query)
            if not albums:
                self._log.info("No matching albums found.")
                return

            if opts.debug:
                self._cmd_debug(albums)
                return

            for album in albums:
                name, mbid = self._resolve(album)

                if not name:
                    continue

                flex = getattr(album, "_values_flex", {})
                if flex.get("primary_albumartist") == name:
                    continue

                if opts.pretend:
                    ui.print_(f"  {album.albumartist} -> {name}  [{album.album}]")
                    continue

                album["primary_albumartist"] = name
                if mbid:
                    album["primary_albumartistid"] = mbid
                album.store()

                for item in album.items():
                    item["primary_albumartist"] = name
                    if mbid:
                        item["primary_albumartistid"] = mbid
                    item.store()

                self._log.info(
                    "primaryartist: {0} -> {1}  [{2}]",
                    album.albumartist,
                    name,
                    album.album,
                )

                artpath = album.artpath
                if artpath:
                    from beets import plugins as beets_plugins

                    beets_plugins.send(
                        "art_set",
                        album=album,
                        artpath=artpath,
                    )

        # ---- internal -----------------------------------------------------

        def _cmd_debug(self, albums):
            """Dump all relevant fields for each album."""
            for album in albums:
                ui.print_(f"\n{'=' * 60}")
                ui.print_(f"Album: {album.album}")
                ui.print_(f"{'=' * 60}")

                # Core artist fields
                fields = [
                    "albumartist",
                    "albumartist_credit",
                    "albumartist_sort",
                    "albumartists",
                    "albumartists_sort",
                    "albumartists_credit",
                    "mb_albumartistid",
                    "mb_albumartistids",
                    "mb_albumid",
                    "mb_releasegroupid",
                    "albumtype",
                    "albumtypes",
                    "comp",
                ]

                for field in fields:
                    val = getattr(album, field, "«not set»")
                    typ = type(val).__name__
                    ui.print_(f"  {field:30s} ({typ:8s}) = {val!r}")

                # Flex attrs
                flex = getattr(album, "_values_flex", {})
                if flex:
                    ui.print_("\n  --- Flex attributes ---")
                    for k, v in sorted(flex.items()):
                        ui.print_(f"  {k:30s}          = {v!r}")
                else:
                    ui.print_("\n  --- No flex attributes ---")

                # What _get_primary returns
                name, mbid = _get_primary(album)
                ui.print_("\n  --- _get_primary result ---")
                ui.print_(f"  primary_name:  {name!r}")
                ui.print_(f"  primary_mbid:  {mbid!r}")

                # What _resolve returns
                rname, rmbid = self._resolve(album)
                ui.print_("\n  --- _resolve result ---")
                ui.print_(f"  resolved_name: {rname!r}")
                ui.print_(f"  resolved_mbid: {rmbid!r}")

                # First item for comparison
                items = list(album.items())
                if items:
                    item = items[0]
                    ui.print_("\n  --- First item fields ---")
                    for field in [
                        "artist",
                        "artists",
                        "artists_ids",
                        "albumartist",
                        "albumartists",
                        "mb_artistid",
                        "mb_artistids",
                        "mb_albumartistid",
                        "mb_albumartistids",
                    ]:
                        val = getattr(item, field, "«not set»")
                        typ = type(val).__name__
                        ui.print_(f"  {field:30s} ({typ:8s}) = {val!r}")

            ui.print_("")

        def _resolve(self, model):
            """Resolve the primary artist for a model.

            Returns ``(name, mbid)`` or ``(None, None)`` if no change
            is needed (solo artist or single-entity band).
            """
            preferred = self.config["preferred_artists"].as_str_seq()

            if preferred:
                full = (
                    getattr(model, "albumartist", None)
                    or getattr(model, "artist", None)
                    or ""
                )
                lower = full.lower()
                for pref in preferred:
                    if pref.lower() in lower:
                        # Use preferred artist — still need an MBID.
                        _, mbid = _get_primary(model)
                        return pref, mbid

            return _get_primary(model)

"""
rsgain plugin for beets.

Replaces the built-in replaygain plugin with rsgain for faster, higher-quality
ReplayGain analysis.  rsgain uses libebur128 for EBU R128 loudness measurement
and writes tags directly to audio files.

On import, files are grouped by their album directory and each directory is
scanned with ``rsgain easy``, which applies recommended per-format presets and
calculates both track and album gain in a single pass.  Multithreading is
handled entirely by rsgain (``--multithread``).

Opus files are handled correctly by rsgain's easy mode preset: it writes
R128_TRACK_GAIN / R128_ALBUM_GAIN tags at -23 LUFS automatically, matching the
behaviour of the built-in plugin's ``r128: [Opus]`` setting.

After rsgain writes tags to the files, the plugin reads the values back and
stores them in the beets database so they are queryable.

CLI:
    beet rsgain [-f] [-t N] [QUERY]

    -f / --force        Re-scan files that already have ReplayGain tags.
    -t / --threads N    Number of parallel threads (default: CPU core count).

Configuration:
    rsgain:
        auto: yes               # Scan automatically during import.
        threads: 0              # 0 = use all CPU cores (rsgain default).
        overwrite: no           # Re-scan files that already have ReplayGain tags.
"""

from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

from beets import ui
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand
from beets.util import displayable_path

if TYPE_CHECKING:
    from beets.importer import ImportSession, ImportTask
    from beets.library import Item, Library


# ---------------------------------------------------------------------------
# ReplayGain tag names by format
# ---------------------------------------------------------------------------

# Formats that use R128_*_GAIN integer tags (in 256ths of a dB).
_R128_FORMATS = {"opus"}

# Map of file extension → (track_gain_tag, album_gain_tag, track_peak_tag, album_peak_tag)
# For R128 formats, peak tags are not used.
_RG_TAGS: Dict[str, tuple] = {
    # Standard ReplayGain 2.0 (stored as strings with " dB" suffix in most formats)
    "default": (
        "REPLAYGAIN_TRACK_GAIN",
        "REPLAYGAIN_ALBUM_GAIN",
        "REPLAYGAIN_TRACK_PEAK",
        "REPLAYGAIN_ALBUM_PEAK",
    ),
    # R128 (Opus): integer values in 256ths of a dB, no peak tags
    "opus": (
        "R128_TRACK_GAIN",
        "R128_ALBUM_GAIN",
        None,
        None,
    ),
}


def _tag_names(ext: str) -> tuple:
    """Return (track_gain, album_gain, track_peak, album_peak) tag names."""
    return _RG_TAGS.get(ext.lower(), _RG_TAGS["default"])


# ---------------------------------------------------------------------------
# rsgain runner
# ---------------------------------------------------------------------------

def _run_rsgain_easy(directory: str, threads: int = 0, overwrite: bool = True) -> bool:
    """Run ``rsgain easy`` on *directory*.

    Args:
        directory:  Path to the album directory to scan.
        threads:    Number of parallel threads; 0 means rsgain default (all cores).
        overwrite:  When False, pass ``--skip-existing`` so rsgain skips files
                    that already have ReplayGain tags.

    Returns:
        (True, None) on success, (False, stderr) on failure.
    """
    cmd = ["rsgain", "easy"]

    if threads > 0:
        cmd += ["--multithread", str(threads)]

    if not overwrite:
        cmd.append("--skip-existing")

    cmd.append(directory)

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
    except FileNotFoundError:
        raise ui.UserError("rsgain not found on PATH — install rsgain first")
    except subprocess.TimeoutExpired:
        raise ui.UserError(f"rsgain timed out scanning {directory!r}")

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        return False, stderr

    return True, None


# ---------------------------------------------------------------------------
# Tag reading helpers
# ---------------------------------------------------------------------------

def _read_rg_tags(item: "Item") -> dict:
    """Read ReplayGain tags from *item*'s file and return a dict of values.

    Returns a dict with keys matching beets' internal field names:
        rg_track_gain, rg_album_gain, rg_track_peak, rg_album_peak

    Values are floats (dB) or None if the tag is absent/unparseable.
    For R128 integer tags, the Q7.8 fixed-point value is converted to dB.
    """
    path = item.path
    if not isinstance(path, str):
        path = os.fsdecode(path)

    ext = os.path.splitext(path)[1].lstrip(".").lower()
    track_tag, album_tag, track_peak_tag, album_peak_tag = _tag_names(ext)
    is_r128 = ext in _R128_FORMATS

    # Use mutagen directly for the most reliable tag reading.
    try:
        import mutagen
        audio = mutagen.File(path, easy=False)
    except Exception:
        return {}

    if audio is None:
        return {}

    def _get(tag_name: Optional[str]) -> Optional[float]:
        if tag_name is None:
            return None
        try:
            val = audio.get(tag_name)
            if val is None:
                return None
            # mutagen returns lists for most tag types
            if isinstance(val, (list, tuple)):
                val = val[0]
            raw = str(val).strip()
            # Strip " dB" suffix if present
            raw = raw.replace(" dB", "").replace("dB", "").strip()
            f = float(raw)
            # R128 Q7.8 fixed-point → dB
            if is_r128:
                f = f / 256.0
            return f
        except (ValueError, TypeError, AttributeError):
            return None

    result = {}
    tg = _get(track_tag)
    ag = _get(album_tag)
    tp = _get(track_peak_tag)
    ap = _get(album_peak_tag)

    if tg is not None:
        result["rg_track_gain"] = tg
    if ag is not None:
        result["rg_album_gain"] = ag
    if tp is not None:
        result["rg_track_peak"] = tp
    if ap is not None:
        result["rg_album_peak"] = ap

    return result


def _store_rg_tags(item: "Item") -> bool:
    """Read RG tags from file and write them into the beets DB.

    Returns True if any values were stored.
    """
    tags = _read_rg_tags(item)
    if not tags:
        return False
    for field, value in tags.items():
        item[field] = value
    item.store()
    return True


# ---------------------------------------------------------------------------
# Directory grouping helper
# ---------------------------------------------------------------------------

def _group_by_directory(items: List["Item"]) -> Dict[str, List["Item"]]:
    """Group *items* by their containing directory."""
    groups: Dict[str, List["Item"]] = defaultdict(list)
    for item in items:
        path = item.path
        if not isinstance(path, str):
            path = os.fsdecode(path)
        groups[os.path.dirname(path)].append(item)
    return dict(groups)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class RsgainPlugin(BeetsPlugin):

    def __init__(self) -> None:
        super().__init__()

        self.config.add({
            "auto": True,
            "threads": 0,
            "overwrite": False,
        })

        if self.config["auto"].get(bool):
            self.register_listener("album_imported", self.on_album_imported)
            self.register_listener("item_imported", self.on_item_imported)

    # -- Import listeners ----------------------------------------------------

    def on_album_imported(self, lib: "Library", album) -> None:
        """Scan the album directory after an album is imported."""
        items = list(album.items())
        if not items:
            return

        # All items in an album share the same directory (after move/copy).
        # Group defensively in case of edge cases (e.g. multi-disc split paths).
        groups = _group_by_directory(items)
        threads = self.config["threads"].get(int)
        overwrite = self.config["overwrite"].get(bool)

        for directory, dir_items in groups.items():
            self._log.info("rsgain: scanning {}", displayable_path(directory))
            ok, err = _run_rsgain_easy(directory, threads=threads, overwrite=overwrite)
            if ok:
                for item in dir_items:
                    _store_rg_tags(item)
            else:
                self._log.warning(
                    "rsgain failed for {}: {}",
                    displayable_path(directory),
                    err or "(no output)",
                )

    def on_item_imported(self, lib: "Library", item: "Item") -> None:
        """Scan a singleton item after it is imported."""
        path = item.path
        if not isinstance(path, str):
            path = os.fsdecode(path)

        directory = os.path.dirname(path)
        threads = self.config["threads"].get(int)
        overwrite = self.config["overwrite"].get(bool)

        self._log.info("rsgain: scanning {}", displayable_path(directory))
        ok, err = _run_rsgain_easy(directory, threads=threads, overwrite=overwrite)
        if ok:
            _store_rg_tags(item)
        else:
            self._log.warning(
                "rsgain failed for {}: {}",
                displayable_path(directory),
                err or "(no output)",
            )

    # -- CLI command ---------------------------------------------------------

    def commands(self) -> list:
        cmd = Subcommand(
            "rsgain",
            help="scan items with rsgain and write ReplayGain tags",
        )
        cmd.parser.add_option(
            "-f", "--force",
            action="store_true", default=False, dest="force",
            help="re-scan files that already have ReplayGain tags",
        )
        cmd.parser.add_option(
            "-t", "--threads",
            type="int", default=None, dest="threads",
            metavar="N",
            help="number of parallel threads (default: all CPU cores)",
        )
        cmd.func = self._command
        return [cmd]

    def _command(self, lib: "Library", opts, args: list) -> None:
        items = list(lib.items(ui.decargs(args)))
        if not items:
            ui.print_("No items matched the query.")
            return

        threads = opts.threads if opts.threads is not None else self.config["threads"].get(int)
        overwrite = opts.force or self.config["overwrite"].get(bool)

        groups = _group_by_directory(items)
        total_dirs = len(groups)
        scanned = 0
        failed = 0

        for directory, dir_items in groups.items():
            shown = displayable_path(directory)
            self._log.info("rsgain: scanning {} ...", shown)
            ok, err = _run_rsgain_easy(directory, threads=threads, overwrite=overwrite)
            if ok:
                stored = sum(1 for item in dir_items if _store_rg_tags(item))
                self._log.debug("rsgain: stored tags for {}/{} items in {}", stored, len(dir_items), shown)
                scanned += 1
            else:
                self._log.warning("rsgain failed for {}: {}", shown, err or "(no output)")
                failed += 1

        ok_str = ui.colorize("text_success", f"{scanned} ok")
        fail_str = ui.colorize("text_error", f"{failed} failed")
        ui.print_(f"\nrsgain: {total_dirs} directories — {ok_str}, {fail_str}")

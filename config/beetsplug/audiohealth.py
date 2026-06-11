"""
AudioHealth plugin for beets.

Smarter replacement for the built-in badfiles plugin.  Runs ffmpeg against
audio files, classifies stderr output into four severity tiers, applies
count-based escalation, persists results as flexible attributes on each item,
and provides a `beet audiohealth` command for on-demand scanning.

Severity tiers (highest to lowest):
    critical  — Navidrome / streaming servers can't serve the file at all.
                Container broken, codec unrecognizable, no decodable stream.
    error     — Audible corruption.  Decoder recovers but listener hears
                pops, gaps, or distortion where frames are damaged.
    warn      — Technically imperfect but sounds fine.  Single sporadic
                frame errors, minor container metadata issues.
    ignore    — Noise.  Timestamp jitter, cover art encoding details, etc.

Count-based escalation:
    When a warn-level pattern fires many times in one file, the cumulative
    damage becomes audible → escalate to error.  Likewise many error hits
    may indicate the file is unservable → escalate to critical.  Thresholds
    are configurable.

Stored flex attrs on each item:
    audiohealth      = ok | warn | error | critical
    audiohealth_log  = stderr lines (non-ignored), truncated to 2 KB

Query examples:
    beet ls audiohealth:critical       # unplayable files
    beet ls audiohealth:error          # audibly corrupt
    beet ls audiohealth:warn           # minor issues
    beet ls audiohealth_log::sync      # regex search the log text
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from beets import ui
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, should_write
from beets.util import displayable_path

if TYPE_CHECKING:
    from beets.importer import ImportTask
    from beets.library import Item, Library


# ---------------------------------------------------------------------------
# Severity constants and helpers
# ---------------------------------------------------------------------------

SEV_OK = "ok"
SEV_IGNORE = "ignore"
SEV_WARN = "warn"
SEV_ERROR = "error"
SEV_CRITICAL = "critical"

_SEV_RANK = {
    SEV_OK: 0,
    SEV_IGNORE: 1,
    SEV_WARN: 2,
    SEV_ERROR: 3,
    SEV_CRITICAL: 4,
}

# Map severities to beets color keys.
_SEV_COLOR = {
    SEV_OK: "text_success",             # bold green
    SEV_WARN: "text_warning",           # bold yellow
    SEV_ERROR: "text_error",            # bold red
    SEV_CRITICAL: "text_error",         # bold red (same as error)
}


def _colorize(color_key: str, text: str) -> str:
    """Wrap *text* in beets terminal color codes, falling back to plain.

    The ``colorize`` function has moved around across beets versions:
      - beets >= 2.0:  beets.ui.colorize (part of the UI overhaul)
      - older beets:   beets.util.color.colorize
    We try both and cache whichever works.
    """
    fn = _colorize._resolved  # type: ignore[attr-defined]
    if fn is None:
        return text
    if fn is _colorize:
        for mod_path in ("beets.ui", "beets.util.color"):
            try:
                mod = __import__(mod_path, fromlist=["colorize"])
                fn = getattr(mod, "colorize", None)
                if fn is not None:
                    _colorize._resolved = fn  # type: ignore[attr-defined]
                    return fn(color_key, text)
            except Exception:
                continue
        _colorize._resolved = None  # type: ignore[attr-defined]
        return text
    return fn(color_key, text)

_colorize._resolved = _colorize  # sentinel  # type: ignore[attr-defined]


def _sev_color(severity: str, text: str) -> str:
    """Colorize *text* according to *severity*."""
    key = _SEV_COLOR.get(severity)
    return _colorize(key, text) if key else text


def _worst(a: str, b: str) -> str:
    """Return whichever severity is more serious."""
    return a if _SEV_RANK.get(a, 0) >= _SEV_RANK.get(b, 0) else b


def _should_stop(stop_level: str, severity: str) -> bool:
    """Decide whether *severity* warrants stopping given *stop_level*.

    stop_level is one of: none, critical, error, warn, all.
    """
    if stop_level == "none":
        return False
    if stop_level == "all":
        return severity in (SEV_WARN, SEV_ERROR, SEV_CRITICAL)
    # Compare ranks: stop if actual severity >= configured threshold.
    threshold = _SEV_RANK.get(stop_level, 0)
    actual = _SEV_RANK.get(severity, 0)
    return actual >= threshold


# ---------------------------------------------------------------------------
# Check result container
# ---------------------------------------------------------------------------

class CheckResult:
    """Holds the outcome of checking one audio file."""

    __slots__ = ("severity", "lines")

    def __init__(self) -> None:
        self.severity: str = SEV_OK
        self.lines: List[Tuple[str, str]] = []  # [(severity, raw_line), ...]

    def add(self, severity: str, line: str) -> None:
        self.lines.append((severity, line))
        self.severity = _worst(self.severity, severity)

    @property
    def log_text(self) -> str:
        """Non-ignored lines formatted for storage / display."""
        parts = []
        for sev, line in self.lines:
            if sev != SEV_IGNORE:
                parts.append(f"[{sev}] {line}")
        return "\n".join(parts)

    @property
    def has_visible_issues(self) -> bool:
        return self.severity in (SEV_WARN, SEV_ERROR, SEV_CRITICAL)

    def severity_counts(self) -> Dict[str, int]:
        """Count lines by severity (excluding ignore)."""
        counts: Dict[str, int] = Counter()
        for sev, _line in self.lines:
            if sev != SEV_IGNORE:
                counts[sev] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# Severity rules  (regex classifier + count-based escalation)
# ---------------------------------------------------------------------------

class SeverityRules:
    """Compiled regex patterns for classifying ffmpeg stderr lines.

    Four pattern tiers: critical > error > warn > ignore.
    Lines matching nothing default to ``error`` (unknown stderr = suspicious).

    After per-line classification, count-based escalation promotes severity:
    if a ``warn`` pattern fires >= ``warn_to_error`` times, those hits are
    re-tagged as ``error``.  Similarly ``error`` × ``error_to_critical``.
    """

    def __init__(
        self,
        critical_patterns: List[str],
        error_patterns: List[str],
        warn_patterns: List[str],
        ignore_patterns: List[str],
        warn_to_error: int = 5,
        error_to_critical: int = 10,
    ) -> None:
        self._critical = [re.compile(p, re.IGNORECASE) for p in critical_patterns]
        self._error = [re.compile(p, re.IGNORECASE) for p in error_patterns]
        self._warn = [re.compile(p, re.IGNORECASE) for p in warn_patterns]
        self._ignore = [re.compile(p, re.IGNORECASE) for p in ignore_patterns]
        self.warn_to_error = warn_to_error
        self.error_to_critical = error_to_critical

    def classify_line(self, line: str) -> str:
        """Classify a single stderr line (no escalation yet)."""
        line = line.strip()
        if not line:
            return SEV_IGNORE

        for pat in self._ignore:
            if pat.search(line):
                return SEV_IGNORE

        for pat in self._critical:
            if pat.search(line):
                return SEV_CRITICAL

        for pat in self._error:
            if pat.search(line):
                return SEV_ERROR

        for pat in self._warn:
            if pat.search(line):
                return SEV_WARN

        # Unknown stderr content defaults to error (be cautious).
        return SEV_ERROR

    def escalate(self, result: CheckResult) -> None:
        """Apply count-based escalation to a finished CheckResult.

        Mutates result.lines severities and recalculates result.severity.
        """
        counts = result.severity_counts()
        warn_count = counts.get(SEV_WARN, 0)
        error_count = counts.get(SEV_ERROR, 0)

        promote_warn = (self.warn_to_error > 0 and warn_count >= self.warn_to_error)
        promote_error = (self.error_to_critical > 0 and error_count >= self.error_to_critical)

        if not promote_warn and not promote_error:
            return

        # Rebuild lines with escalated severities.
        new_lines = []
        for sev, line in result.lines:
            if sev == SEV_WARN and promote_warn:
                sev = SEV_ERROR
            # Note: check the *original* severity for error→critical,
            # not the just-promoted warns. Promoted warns are newly error
            # but weren't part of the error_count that triggered this.
            new_lines.append((sev, line))

        # Second pass: if error→critical promotion is needed, do it on
        # lines that were originally error (now still error after pass 1).
        if promote_error:
            final_lines = []
            for sev, line in new_lines:
                if sev == SEV_ERROR:
                    sev = SEV_CRITICAL
                final_lines.append((sev, line))
            new_lines = final_lines

        result.lines = new_lines

        # Recalculate overall severity.
        result.severity = SEV_OK
        for sev, _line in result.lines:
            result.severity = _worst(result.severity, sev)


# ---------------------------------------------------------------------------
# Checker base class
# ---------------------------------------------------------------------------

class BaseChecker:
    """Abstract checker.  Subclass for format-specific tools."""

    name: str = "base"

    def can_check(self, ext: str) -> bool:
        raise NotImplementedError

    def check(self, path: str, rules: SeverityRules) -> CheckResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FFmpeg checker
# ---------------------------------------------------------------------------

class FFmpegChecker(BaseChecker):
    """Decode the entire file with ``ffmpeg -f null`` and classify stderr."""

    name = "ffmpeg"

    def can_check(self, ext: str) -> bool:
        return True

    def check(self, path: str, rules: SeverityRules) -> CheckResult:
        result = CheckResult()

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", path,
            "-f", "null",
            "-",
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=600,
            )
        except FileNotFoundError:
            result.add(SEV_CRITICAL, "ffmpeg not found on PATH")
            return result
        except subprocess.TimeoutExpired:
            result.add(SEV_ERROR, "ffmpeg timed out (>600 s)")
            return result
        except OSError as exc:
            result.add(SEV_CRITICAL, f"failed to run ffmpeg: {exc}")
            return result

        stderr = proc.stderr.decode("utf-8", "replace").strip()

        # Non-zero exit with no stderr → file couldn't be opened at all.
        if proc.returncode != 0 and not stderr:
            result.add(SEV_CRITICAL, f"ffmpeg exited with code {proc.returncode} (no output)")
            return result

        if not stderr:
            return result  # clean file

        # Classify each line.
        for raw_line in stderr.splitlines():
            raw_line = raw_line.strip()
            if raw_line:
                sev = rules.classify_line(raw_line)
                result.add(sev, raw_line)

        # Apply count-based escalation.
        rules.escalate(result)

        return result


# ---------------------------------------------------------------------------
# Default severity patterns
# ---------------------------------------------------------------------------
# Patterns are Python regexes tested with re.search (case-insensitive).
# Checked in order: ignore → critical → error → warn → default(error).

# Critical: Navidrome / streaming servers will fail to serve the file.
DEFAULT_CRITICAL = [
    r"moov atom not found",             # MP4/M4A container completely broken
    r"Invalid NAL unit size",           # video codec garbage in audio file
    r"no audio.*found",                 # no decodable audio stream at all
    r"misdetection possible",           # ffmpeg can't identify the format
    r"Format .+ detected only with low score", # container barely recognisable
]

# Error: Audible corruption — pops, gaps, distortion.
DEFAULT_ERROR = [
    r"invalid sync code",               # FLAC frame sync bytes mangled
    r"invalid frame header",            # frame structure broken
    r"decode_frame.*failed",            # decoder gave up on a frame
    r"FLAC__STREAM_DECODER_ERROR",      # native FLAC decoder error
]

# Warn: Technically imperfect but sounds fine on playback.
DEFAULT_WARN = [
    r"Header missing",                  # MP3: sporadic missing frame header
    r"Error while decoding",            # occasional single-frame hiccup
    r"Error submitting.*packet",        # downstream of a single bad frame
    r"Could not find codec parameters", # stream probe difficulty
    r"max_analyze_duration",            # probe timeout
    r"overread",                        # reading past buffer edge
    r"CRC mismatch",                    # single-frame CRC failure
    r"Estimating duration",             # container missing duration
    r"invalid data found",              # ambiguous, could be minor
    r"missing picture",                 # cover art metadata absent
    r"invalid new backstep",            # MP3 bit reservoir glitch
    r"big_values too big",              # MP3 Huffman table overrun
    r"not enough frames to estimate",   # short stream probe issue
    r"Discarding corrupted packet",     # single packet dropped
]

# Ignore: No impact on anything.
DEFAULT_IGNORE = [
    r"non.monoton",                     # DTS timestamp jitter
    r"Discarding ID3 tags",             # ID3 in FLAC containers
    r"Last message repeated",           # ffmpeg dedup line
    r"deprecated pixel format",         # cover art encoding detail
    r"Application provided invalid",    # minor API-level note
    r"Queue input is backward",         # timestamp reorder
    r"sample/frame number mismatch",    # minor sync blip in Vorbis
    r"Discarding padding",              # padding trimmed on decode
    r"Could not update timestamps",     # benign muxer note
]


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class AudioHealthPlugin(BeetsPlugin):

    def __init__(self) -> None:
        super().__init__()

        self.config.add({
            "auto": True,
            "stop_on_error": "none",
            "overwrite": False,
            "severity_rules": {
                "critical": list(DEFAULT_CRITICAL),
                "error": list(DEFAULT_ERROR),
                "warn": list(DEFAULT_WARN),
                "ignore": list(DEFAULT_IGNORE),
            },
            "escalation": {
                "warn_to_error": 5,
                "error_to_critical": 10,
            },
        })

        self._rules: Optional[SeverityRules] = None
        self._checkers: Optional[List[BaseChecker]] = None

        if self.config["auto"].get(bool):
            self.register_listener(
                "import_task_start", self.on_import_task_start
            )
            self.register_listener(
                "import_task_before_choice", self.on_import_task_before_choice
            )
            self.import_stages = [self.on_imported]

    # -- Lazy initialisation -------------------------------------------------

    def _load_patterns(self, key: str, default: List[str]) -> List[str]:
        """Load a pattern list from config, falling back to default."""
        try:
            return list(self.config["severity_rules"][key].as_str_seq())
        except Exception:
            return list(default)

    def _get_rules(self) -> SeverityRules:
        if self._rules is None:
            critical = self._load_patterns("critical", DEFAULT_CRITICAL)
            error = self._load_patterns("error", DEFAULT_ERROR)
            warn = self._load_patterns("warn", DEFAULT_WARN)
            ignore = self._load_patterns("ignore", DEFAULT_IGNORE)

            esc_cfg = self.config["escalation"]
            try:
                w2e = esc_cfg["warn_to_error"].get(int)
            except Exception:
                w2e = 5
            try:
                e2c = esc_cfg["error_to_critical"].get(int)
            except Exception:
                e2c = 10

            self._rules = SeverityRules(
                critical_patterns=critical,
                error_patterns=error,
                warn_patterns=warn,
                ignore_patterns=ignore,
                warn_to_error=w2e,
                error_to_critical=e2c,
            )
        return self._rules

    def _get_checkers(self) -> List[BaseChecker]:
        if self._checkers is None:
            self._checkers = [FFmpegChecker()]
        return self._checkers

    def _find_checker(self, ext: str) -> Optional[BaseChecker]:
        for checker in self._get_checkers():
            if checker.can_check(ext.lower()):
                return checker
        return None

    # -- Core checking -------------------------------------------------------

    def check_path(self, path: str) -> CheckResult:
        """Run the appropriate checker against a filesystem path."""
        ext = os.path.splitext(path)[1].lstrip(".")
        checker = self._find_checker(ext)
        if checker is None:
            r = CheckResult()
            r.add(SEV_WARN, f"no checker available for .{ext}")
            return r
        return checker.check(path, self._get_rules())

    def check_item(self, item: "Item") -> CheckResult:
        """Check a beets Item (resolves its path first)."""
        path = item.path
        if not isinstance(path, str):
            path = os.fsdecode(path)

        if not os.path.exists(path):
            r = CheckResult()
            r.add(SEV_CRITICAL, "file does not exist")
            return r

        return self.check_path(path)

    @staticmethod
    def store_result(
        item: "Item",
        result: CheckResult,
        write: bool = False,
    ) -> None:
        """Persist a CheckResult as flexible attributes on *item*."""
        item["audiohealth"] = result.severity
        log = result.log_text
        item["audiohealth_log"] = log[:2048] if log else ""

        if write:
            item.try_write()
        item.store()

    # -- Display helpers ------------------------------------------------------

    @staticmethod
    def _print_result(shown: str, result: CheckResult) -> None:
        """Print a check result to the terminal with color-coded severity."""
        if result.has_visible_issues:
            tag = _sev_color(result.severity, f"[{result.severity.upper()}]")
            path = _colorize("text_highlight_minor", shown)
            ui.print_(f"  {tag} {path}")

            for sev, line in result.lines:
                if sev != SEV_IGNORE:
                    line_tag = _sev_color(sev, f"[{sev}]")
                    detail = _colorize("text_highlight_minor", line)
                    ui.print_(f"    {line_tag} {detail}")

    @staticmethod
    def _print_ok(shown: str) -> None:
        """Print an 'ok' result (used in verbose mode)."""
        tag = _sev_color(SEV_OK, "[OK]")
        path = _colorize("text_highlight_minor", shown)
        ui.print_(f"  {tag} {path}")

    # -- Import hooks --------------------------------------------------------

    def on_import_task_start(self, task, session) -> None:
        """Check every candidate file and cache results on the task.

        We do NOT print here — import_task_start fires BEFORE beets prints
        the album header, so our output would appear under the previous
        album's context.
        """
        items = getattr(task, "items", None) or []
        task._audiohealth_cache = []

        for item in items:
            path = item.path
            if not isinstance(path, str):
                path = os.fsdecode(path)

            result = self.check_path(path)
            task._audiohealth_cache.append(result)

    def on_import_task_before_choice(self, task, session):
        """Display cached check results and optionally ask user what to do."""
        cache = getattr(task, "_audiohealth_cache", [])
        if not cache:
            return None

        items = getattr(task, "items", None) or []

        task_worst = SEV_OK
        for idx, result in enumerate(cache):
            task_worst = _worst(task_worst, result.severity)
            if result.has_visible_issues and idx < len(items):
                shown = displayable_path(items[idx].path)
                self._print_result(shown, result)

        stop_level = self.config["stop_on_error"].as_str()
        if not _should_stop(stop_level, task_worst):
            return None

        from beets import importer

        header = _sev_color(task_worst, "AUDIOHEALTH")
        level_text = _sev_color(task_worst, f"{task_worst}-level")

        ui.print_(
            f"\n{header} found {level_text} issues "
            f"(stop_on_error: {stop_level})."
        )
        ui.print_("What would you like to do?")
        sel = ui.input_options(["Continue", "Skip", "aBort"])
        if sel == "s":
            return importer.Action.SKIP
        elif sel == "b":
            raise importer.ImportAbortError()
        return None

    def on_imported(self, session, task: "ImportTask") -> None:
        """After items are written to the library, persist check results."""
        cache = getattr(task, "_audiohealth_cache", [])
        imported = task.imported_items()

        for idx, item in enumerate(imported):
            if idx < len(cache):
                result = cache[idx]
            else:
                result = self.check_item(item)
                shown = displayable_path(item.path)
                self._print_result(shown, result)

            self.store_result(item, result)
            self._log.debug(
                "audiohealth for {}: {}",
                displayable_path(item.path),
                result.severity,
            )

    # -- CLI command ---------------------------------------------------------

    def commands(self) -> list:
        cmd = Subcommand(
            "audiohealth",
            help="check audio files for corruption and log results",
        )
        cmd.parser.add_option(
            "-f", "--force",
            action="store_true", default=False, dest="force",
            help="recheck items that already have audiohealth results",
        )
        cmd.parser.add_option(
            "-v", "--verbose",
            action="store_true", default=False, dest="verbose",
            help="show results for clean files too",
        )
        cmd.parser.add_option(
            "-w", "--write",
            action="store_true", default=False, dest="write",
            help="write tags to files after updating DB fields",
        )
        cmd.func = self._command
        return [cmd]

    def _command(self, lib: "Library", opts, args: list) -> None:
        items = list(lib.items(args))
        overwrite = opts.force or self.config["overwrite"].get(bool)
        write = opts.write or should_write()
        verbose = opts.verbose

        counts = {SEV_OK: 0, SEV_WARN: 0, SEV_ERROR: 0, SEV_CRITICAL: 0}
        total = len(items)

        for i, item in enumerate(items, 1):
            shown = displayable_path(item.path)

            existing = item.get("audiohealth")
            if existing and not overwrite:
                self._log.info(
                    "audiohealth for {} already set: {} (use -f to recheck)",
                    shown, existing,
                )
                if existing in counts:
                    counts[existing] += 1
                if verbose and existing == SEV_OK:
                    self._print_ok(shown)
                elif verbose and existing in (SEV_WARN, SEV_ERROR, SEV_CRITICAL):
                    tag = _sev_color(existing, f"[{existing.upper()}]")
                    path = _colorize("text_highlight_minor", shown)
                    ui.print_(f"  {tag} {path} (cached)")
                continue

            result = self.check_item(item)
            self.store_result(item, result, write=write)

            if result.severity in counts:
                counts[result.severity] += 1

            if result.has_visible_issues:
                self._print_result(shown, result)
            elif verbose:
                self._print_ok(shown)

            if total > 50 and i % 100 == 0:
                self._log.info("checked {}/{} items ...", i, total)

        # Summary.
        ok_str = _sev_color(SEV_OK, f"{counts[SEV_OK]} ok")
        warn_str = _sev_color(SEV_WARN, f"{counts[SEV_WARN]} warn")
        err_str = _sev_color(SEV_ERROR, f"{counts[SEV_ERROR]} error")
        crit_str = _sev_color(SEV_CRITICAL, f"{counts[SEV_CRITICAL]} critical")
        ui.print_(
            f"\naudiohealth: {total} checked — "
            f"{ok_str}, {warn_str}, {err_str}, {crit_str}"
        )
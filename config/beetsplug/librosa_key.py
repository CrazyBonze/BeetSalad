"""
Librosa Key plugin for beets.

Computes global musical key using a Krumhansl–Schmuckler-style approach:
- Compute chroma (C..B) over a segment
- Average chroma to a single pitch-class profile
- Correlate vs. major/minor key profiles over 12 rotations
Stores result in item's `initial_key`.

Update (decoder change):
- Many libraries include .m4a (AAC/MP4) files which libsndfile/PySoundFile often
  cannot decode. Librosa then falls back to `audioread`, which is deprecated and
  produces warnings like "PySoundFile failed. Trying audioread instead."
- To avoid those warnings (and to robustly support .m4a and other codecs),
  this plugin now decodes audio via `ffmpeg` to raw PCM and then analyzes it.
  This is usually more reliable across formats because ffmpeg supports far more
  containers/codecs than libsndfile.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Optional, Tuple

from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, should_write
from beets.util import displayable_path

if TYPE_CHECKING:
    from beets.importer import ImportTask
    from beets.library import Item, Library


# Krumhansl-Kessler key profiles (common defaults).
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _rotate(vec, n: int):
    n = n % len(vec)
    return vec[-n:] + vec[:-n]


def _pearson(a, b) -> float:
    # small, dependency-free Pearson correlation
    import math

    if len(a) != len(b) or not a:
        return float("-inf")
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    den = math.sqrt(da * db) if da > 0 and db > 0 else 0.0
    return num / den if den else float("-inf")


def _load_mono_ffmpeg(
    path: str,
    sr: int,
    offset: float = 0.0,
    duration: Optional[float] = None,
):
    """
    Decode audio using ffmpeg and return (y, sr) where y is mono float32.

    Why:
    - libsndfile/PySoundFile often cannot decode .m4a (AAC/MP4) and some MP3s.
      Librosa then falls back to `audioread`, which is deprecated and produces
      warnings.
    - Decoding with ffmpeg avoids that path and supports more formats reliably.
    """
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err", "-nostdin", "-vn"]

    # Seek (fast enough for analysis; good to keep before -i).
    if offset and offset > 0:
        cmd += ["-ss", str(offset)]

    cmd += ["-i", path]

    # Limit analysis duration.
    if duration and duration > 0:
        cmd += ["-t", str(duration)]

    # Output raw mono PCM float32 at desired sample rate to stdout.
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # FIX: Split error handling into two separate checks.
    #
    # Previously this was a single `if proc.returncode != 0 or not proc.stdout`
    # branch, which meant that when ffmpeg succeeded (returncode 0) but produced
    # no output (e.g. a very short or silent track), stderr was empty too, and
    # the error message was just "ffmpeg decode failed: " with nothing after it.
    #
    # Now we check returncode first (actual ffmpeg error) and empty stdout second
    # (successful decode but no usable audio), each with a distinct message.
    err = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed (rc={proc.returncode}): {err[:500] if err else 'unknown error'}"
        )
    if not proc.stdout:
        raise RuntimeError(
            f"ffmpeg produced no audio data (file may be too short or silent)"
            + (f": {err[:200]}" if err else "")
        )

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    return y, sr


class LibrosaKeyPlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.config.add(
            {
                "auto": True,
                "overwrite": False,
                "t_start": 0.0,      # seconds
                "t_end": 120.0,      # seconds; keep bounded for speed
                "sr": 22050,         # librosa default-ish; lower = faster
                "format": "standard" # standard | compact | camelot | openkey (latter two optional)
            }
        )

        if self.config["auto"].get(bool):
            self.import_stages = [self.imported]

    def commands(self):
        cmd = Subcommand("librosa_key", help="detect and add initial_key from audio using librosa")
        cmd.func = self.command
        return [cmd]

    def command(self, lib: "Library", _opts, args: list[str]) -> None:
        self.find_key(list(lib.items(args)), write=should_write())

    def imported(self, _session, task: "ImportTask") -> None:
        self.find_key(task.imported_items(), write=False)

    def _estimate_key(self, path: str) -> Optional[Tuple[str, str]]:
        """
        Returns (tonic, mode) where mode is "major" or "minor".
        """
        import librosa

        t_start = float(self.config["t_start"].get(float))
        t_end = float(self.config["t_end"].get(float))
        sr = int(self.config["sr"].get(int))

        # Load only a segment (fast + avoids analyzing silence/outros).
        duration = max(0.0, t_end - t_start) if t_end and t_end > t_start else None

        # Decode with ffmpeg to avoid librosa's PySoundFile/audioread warnings and
        # to support formats like .m4a reliably.
        y, sr = _load_mono_ffmpeg(path, sr=sr, offset=t_start, duration=duration)

        # FIX: Tighten minimum-length check. The old `len(y) < sr` guard (< 1 second)
        # let through very short decoded fragments (e.g. 400 samples at sr=22050 is
        # only ~0.02s) that came from studio dialogue or false-start tracks. These
        # then triggered librosa's "n_fft=512 is too large for input signal" warnings
        # from chroma_cqt. Requiring at least 2 seconds filters these out reliably.
        if y is None or len(y) < sr * 2:
            return None

        # Suppress any remaining n_fft warnings from librosa for borderline-length
        # audio that still passes the check above.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"n_fft=\d+ is too large", category=UserWarning)
            # Chroma energy normalized per-frame; then average over time to a single profile.
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        profile = chroma.mean(axis=1)

        # Normalize profile to reduce level effects.
        prof_sum = float(profile.sum())
        if prof_sum <= 0:
            return None
        profile = (profile / prof_sum).astype(float).tolist()

        best = (float("-inf"), 0, "major")  # (score, rotation, mode)

        for r in range(12):
            maj = _pearson(profile, _rotate(MAJOR_PROFILE, r))
            if maj > best[0]:
                best = (maj, r, "major")

            minu = _pearson(profile, _rotate(MINOR_PROFILE, r))
            if minu > best[0]:
                best = (minu, r, "minor")

        tonic = PITCHES[best[1]]
        mode = best[2]
        return tonic, mode

    def _format_key(self, tonic: str, mode: str) -> str:
        fmt = (self.config["format"].as_str() or "standard").lower()

        if fmt == "compact":
            return f"{tonic}m" if mode == "minor" else tonic

        # Keep "standard" close to what beets keyfinder stored historically:
        # major stored as just tonic; minor stored as e.g. "Am" is common,
        # but we'll keep readable: "A minor" unless you want the compact style.
        if fmt == "standard":
            return f"{tonic} minor" if mode == "minor" else tonic

        # Optional: You can add Camelot/OpenKey mapping later if you want.
        return f"{tonic} {mode}"

    def find_key(self, items: list["Item"], write: bool = False) -> None:
        overwrite = self.config["overwrite"].get(bool)

        for item in items:
            shown = displayable_path(item.filepath)

            existing = item.get("initial_key")
            if existing and not overwrite:
                self._log.info("initial_key for {} already exists: {}", shown, existing)
                continue

            audio_path = os.fsdecode(item.filepath)

            try:
                est = self._estimate_key(audio_path)
                if not est:
                    self._log.debug("Skipping {} (too short or silent to detect key)", shown)
                    continue
                tonic, mode = est
                key_str = self._format_key(tonic, mode)
            except FileNotFoundError:
                self._log.error("Failed to detect key for {}: file not found", shown)
                continue
            except RuntimeError as exc:
                # RuntimeError covers our own "too short" / "no audio data" raises
                # from _load_mono_ffmpeg — these are expected skips, not failures.
                self._log.debug("Skipping {} ({})", shown, exc)
                continue
            except Exception as exc:
                self._log.error("Failed to detect key for {}: {}", shown, exc)
                continue

            item["initial_key"] = key_str
            self._log.info("Computed initial_key for {}: {}", shown, key_str)

            if write:
                item.try_write()
            item.store()
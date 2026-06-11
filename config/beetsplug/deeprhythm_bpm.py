"""
DeepRhythm BPM plugin for beets.

Minimal replacement for beets' autobpm (Librosa) using deeprhythm (PyTorch CNN).
Stores integer BPM in the item's `bpm` field.

Update (decoder change):
- Many libraries include .m4a (AAC/MP4) files which libsndfile/PySoundFile often
  cannot decode. deeprhythm uses librosa.load() internally for file decoding;
  when SoundFile fails, librosa falls back to `audioread` (deprecated) and emits
  warnings like "PySoundFile failed. Trying audioread instead."
- To avoid those warnings (and to robustly support .m4a and other codecs),
  this plugin now decodes audio via `ffmpeg` to raw PCM and then calls
  DeepRhythmPredictor.predict_from_audio(...), bypassing deeprhythm's internal
  librosa-based file loading.
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

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Using padding='same' with even kernel lengths and odd dilation.*",
    category=UserWarning,
)


def _load_mono_ffmpeg(
    path: str,
    sr: int = 22050,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple["object", int]:
    """
    Decode audio using ffmpeg and return (y, sr) where y is mono float32.

    Why:
    - deeprhythm uses librosa.load() internally when you call predict(filename).
      For .m4a (AAC/MP4) files, PySoundFile/libsndfile often fails to decode, so
      librosa falls back to audioread (deprecated) and produces warnings.
    - Decoding with ffmpeg + using predict_from_audio avoids that path and is
      more reliable across codecs/containers.
    """
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err", "-nostdin", "-vn"]

    # Seek (fast enough for analysis).
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


class DeepRhythmBPMPlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()

        # Tiny config: only what affects behavior.
        self.config.add(
            {
                "auto": True,        # run on import
                "overwrite": False,  # overwrite existing bpm

                # Decode/analysis controls (kept minimal, but useful):
                "sr": 22050,         # sample rate fed into the model
                "t_start": 0.0,      # seconds; where to start analyzing
                "t_end": 180.0,      # seconds; cap for speed
            }
        )

        self._predictor = None  # lazy init

        if self.config["auto"].get(bool):
            self.import_stages = [self.imported]

    def commands(self) -> list[Subcommand]:
        cmd = Subcommand(
            "bpm",
            help="detect and add bpm from audio using DeepRhythm (CUDA->CPU fallback)",
        )
        cmd.func = self.command
        return [cmd]

    def command(self, lib: Library, _opts, args: list[str]) -> None:
        self.calculate_bpm(list(lib.items(args)), write=should_write())

    def imported(self, _session, task: ImportTask) -> None:
        self.calculate_bpm(task.imported_items(), write=False)

    def _init_predictor_cuda_then_cpu(self):
        from deeprhythm import DeepRhythmPredictor  # type: ignore

        # Prefer CUDA if torch says it's available. If predictor init fails anyway,
        # fall back to CPU.
        try:
            import torch  # type: ignore

            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                self._log.debug("Initializing DeepRhythmPredictor on CUDA")
                try:
                    self._log.info("DeepRhythm using CUDA")
                    return DeepRhythmPredictor(device="cuda")
                except Exception as exc:
                    self._log.warning(
                        "DeepRhythm CUDA init failed ({}); falling back to CPU", exc
                    )
        except Exception as exc:
            # If torch import/inspection fails for any reason, just go CPU.
            self._log.debug("Torch/CUDA check failed ({}); using CPU", exc)

        self._log.debug("Initializing DeepRhythmPredictor on CPU")
        self._log.info("DeepRhythm using CPU")
        return DeepRhythmPredictor(device="cpu")

    def _get_predictor(self):
        if self._predictor is None:
            self._predictor = self._init_predictor_cuda_then_cpu()
        return self._predictor

    def _predict_bpm(self, predictor, sys_path: str) -> float:
        """
        Predict tempo using DeepRhythm, preferring predict_from_audio to avoid
        deeprhythm/librosa decoding warnings on formats like .m4a.
        """
        sr = int(self.config["sr"].get(int))
        t_start = float(self.config["t_start"].get(float))
        t_end = float(self.config["t_end"].get(float))
        duration = max(0.0, t_end - t_start) if t_end and t_end > t_start else None

        # Decode ourselves (ffmpeg) and bypass deeprhythm's librosa.load().
        y, sr = _load_mono_ffmpeg(sys_path, sr=sr, offset=t_start, duration=duration)

        # Guard against extremely short/empty decodes.
        # FIX: Tightened from `< sr` (1 second) to `< sr * 2` (2 seconds). Very
        # short clips like studio dialogue fragments would pass the old check but
        # were still too short for the model to produce a meaningful prediction,
        # causing predict_from_audio to return None.
        if y is None or len(y) < sr * 2:
            raise RuntimeError("audio too short after decoding")

        # FIX: Only catch TypeError for the predict_from_audio signature variation.
        #
        # Previously the outer `except Exception` caught ALL errors from
        # predict_from_audio — including the AttributeError ("'NoneType' object
        # has no attribute 'to'") that PyTorch raises internally when the model
        # can't produce a result for short/silent audio. That funneled every
        # failure into the predictor.predict(sys_path) fallback, which hit the
        # same error again but now uncaught, producing the confusing traceback.
        #
        # Now we only catch TypeError (signature mismatch between deeprhythm
        # versions) and let real prediction failures propagate to the None check
        # below or out to calculate_bpm's exception handler.
        try:
            result = predictor.predict_from_audio(y, sr)
        except TypeError:
            result = predictor.predict_from_audio(y, sr=sr)

        if result is None:
            raise RuntimeError("model returned None (audio may be too short or silent)")
        return float(result)

    def calculate_bpm(self, items: list[Item], write: bool = False) -> None:
        predictor = None

        for item in items:
            path = item.filepath
            shown = displayable_path(path)

            existing = item.get("bpm")
            if existing and not self.config["overwrite"].get(bool):
                self._log.info("BPM for {} already exists: {}", shown, existing)
                continue

            sys_path = os.fsdecode(path)

            try:
                predictor = predictor or self._get_predictor()
                tempo = self._predict_bpm(predictor, sys_path)
                bpm = int(round(float(tempo)))
            except (RuntimeError, AttributeError) as exc:
                # RuntimeError: our own "too short" / "no audio data" /
                #   "model returned None" raises.
                # AttributeError: deeprhythm/PyTorch internals returning None
                #   for unanalyzable audio ("'NoneType' has no attribute 'to'").
                # Both are expected skips for dialogue clips, false starts, etc.
                # Only visible with -v.
                self._log.debug("Skipping {} ({})", shown, exc)
                continue
            except Exception as exc:
                self._log.error("Failed to measure BPM for {}: {}", shown, exc)
                continue

            item["bpm"] = bpm
            self._log.info("Computed BPM for {}: {}", shown, bpm)

            if write:
                item.try_write()
            item.store()
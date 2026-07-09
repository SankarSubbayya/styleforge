"""F1: clip -> evenly sampled frames (base64 data URIs) + Whisper transcript."""

import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class ClipInfo:
    duration: float
    has_audio: bool


def probe(path: Path) -> ClipInfo:
    out = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    meta = json.loads(out)
    streams = meta.get("streams", [])
    # Some containers (webm, fragmented mp4) report no format-level duration.
    duration = 0.0
    for candidate in [meta.get("format", {}).get("duration")] + [
        s.get("duration") for s in streams
    ]:
        try:
            duration = max(duration, float(candidate))
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        duration = 60.0  # guide guarantees 30s-2min; sampling past EOF just skips
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return ClipInfo(duration=duration, has_audio=has_audio)


def extract_frames(path: Path, info: ClipInfo | None = None) -> list[tuple[float, str]]:
    """Return [(timestamp_sec, jpeg data URI)], evenly spaced, capped at MAX_FRAMES.

    Single ffmpeg pass: one decode of the (possibly UHD) source instead of one
    open+seek+decode cycle per frame — this runs on small CPUs under a time cap.
    """
    import tempfile

    info = info or probe(path)
    n = min(config.MAX_FRAMES, max(4, int(info.duration)))
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [
                "ffmpeg", "-v", "quiet", "-i", str(path),
                "-vf", f"fps={n}/{info.duration:.3f},scale={config.FRAME_WIDTH}:-2",
                "-frames:v", str(n), "-q:v", "4", f"{td}/f_%03d.jpg",
            ],
            check=True,
        )
        frames: list[tuple[float, str]] = []
        for i, jpg in enumerate(sorted(Path(td).glob("f_*.jpg"))):
            t = info.duration * (i + 0.5) / n
            uri = "data:image/jpeg;base64," + base64.b64encode(jpg.read_bytes()).decode()
            frames.append((t, uri))
    if not frames:
        raise RuntimeError(f"no frames extracted from {path}")
    return frames


_whisper_model = None


def _get_whisper():
    """Load once per process — model load costs seconds and the harness runs
    several clips through a small worker pool."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # lazy: heavy import

        _whisper_model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type="int8"
        )
    return _whisper_model


def transcribe(path: Path, info: ClipInfo | None = None) -> str:
    """Whisper transcript; empty string for silent clips (Appendix B contingency)."""
    info = info or probe(path)
    if not info.has_audio:
        return ""
    segments, _ = _get_whisper().transcribe(str(path))
    return " ".join(seg.text.strip() for seg in segments).strip()

"""
audio_processor.py — Module 2: Speech Transcription Engine

Responsibilities
----------------
1. Extract audio from the video file (ffmpeg).
2. Transcribe using OpenAI Whisper (ASR).
3. Align transcript segments with slide timestamps.
4. Evaluate transcription quality via Word Error Rate (WER).
5. Persist transcripts as JSON and plain text.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import (
    AUDIO_CHANNELS,
    AUDIO_DIR,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    INPUT_VIDEO,
    TRANSCRIPTS_DIR,
    WHISPER_MODEL,
)
from modules.utils import Timer, clean_text, get_logger, save_json

log = get_logger(__name__)


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """One Whisper segment: a short utterance with timing info."""
    id: int
    start: float        # seconds
    end: float          # seconds
    text: str
    slide_index: Optional[int] = None   # assigned during alignment


@dataclass
class Transcript:
    """Full transcript for a lecture video."""
    language: str
    segments: list[TranscriptSegment]
    full_text: str

    @classmethod
    def from_whisper_result(cls, result: dict) -> "Transcript":
        segments = [
            TranscriptSegment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=clean_text(seg["text"]),
            )
            for seg in result.get("segments", [])
        ]
        return cls(
            language=result.get("language", "en"),
            segments=segments,
            full_text=clean_text(result.get("text", "")),
        )

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "full_text": self.full_text,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "slide_index": s.slide_index,
                }
                for s in self.segments
            ],
        }


# ─── Step 1: Audio extraction ─────────────────────────────────────────────────

def extract_audio(
    video_path: Path = INPUT_VIDEO,
    audio_dir: Path = AUDIO_DIR,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
) -> Path:
    """
    Extract the audio stream from a video file using ffmpeg.

    The output is a mono 16 kHz WAV file — the format expected by Whisper.

    Args:
        video_path:  Source video.
        audio_dir:   Output directory.
        sample_rate: Target sample rate (Hz).
        channels:    1 = mono.

    Returns:
        Path to the extracted WAV file.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"{video_path.stem}.{AUDIO_FORMAT}"

    log.info("Extracting audio → %s", out_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # 16-bit PCM WAV
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(out_path),
    ]

    with Timer() as t:
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    log.info("Audio extracted in %s", t)
    return out_path


# ─── Step 2: Whisper transcription ────────────────────────────────────────────

def transcribe(
    audio_path: Path,
    model_name: str = WHISPER_MODEL,
    transcripts_dir: Path = TRANSCRIPTS_DIR,
) -> Transcript:
    """
    Transcribe audio using OpenAI Whisper.

    Whisper is loaded once per call; for repeated calls, consider caching the
    model object externally.

    Args:
        audio_path:      Path to the WAV file.
        model_name:      Whisper model size: tiny|base|small|medium|large.
        transcripts_dir: Where to save outputs.

    Returns:
        A Transcript object with per-segment timing.
    """
    import whisper  # deferred import — heavy dependency

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading Whisper model '%s'…", model_name)
    model = whisper.load_model(model_name)

    log.info("Transcribing: %s", audio_path)
    with Timer() as t:
        result = model.transcribe(
            str(audio_path),
            verbose=False,
            word_timestamps=False,
            fp16=False,
        )

    transcript = Transcript.from_whisper_result(result)
    log.info(
        "Transcription complete in %s — %d segments, language='%s'",
        t, len(transcript.segments), transcript.language,
    )

    # Persist JSON + plain text
    stem = audio_path.stem
    save_json(transcript.to_dict(), transcripts_dir / f"{stem}.json")
    (transcripts_dir / f"{stem}.txt").write_text(
        transcript.full_text, encoding="utf-8"
    )
    log.info("Transcript saved to %s/", transcripts_dir)

    return transcript


# ─── Step 3: Slide alignment ──────────────────────────────────────────────────

def align_with_slides(
    transcript: Transcript,
    slide_timestamps: list[float],
) -> Transcript:
    """
    Assign each transcript segment to the slide active at that moment.

    Strategy: a slide is active from its start timestamp until the next slide
    starts.  Each segment is assigned to the slide whose window contains the
    segment's midpoint.

    Args:
        transcript:        The full Transcript object.
        slide_timestamps:  List of slide-change timestamps in seconds (sorted).

    Returns:
        The same Transcript with slide_index set on every segment.
    """
    if not slide_timestamps:
        log.warning("No slide timestamps provided; skipping alignment.")
        return transcript

    boundaries = sorted(slide_timestamps) + [float("inf")]

    for seg in transcript.segments:
        mid = (seg.start + seg.end) / 2.0
        # Binary search for the right bucket
        idx = 0
        for i, ts in enumerate(boundaries[:-1]):
            if ts <= mid < boundaries[i + 1]:
                idx = i
                break
        seg.slide_index = idx

    log.info("Segments aligned to %d slides.", len(slide_timestamps))
    return transcript


# ─── Step 4: WER evaluation ───────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Compute Word Error Rate (WER) between a reference and hypothesis transcript.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference word count.

    Uses the `jiwer` library which implements the standard WER pipeline
    (lowercase, strip punctuation, normalise whitespace).

    Args:
        reference:   Ground-truth transcript text.
        hypothesis:  ASR output text.

    Returns:
        Dict with wer, mer (match error rate), wil (word information lost).
    """
    try:
        import jiwer

        measures = jiwer.compute_measures(reference, hypothesis)
        wer  = measures["wer"]
        mer  = measures["mer"]
        wil  = measures["wil"]

        log.info("WER=%.3f  MER=%.3f  WIL=%.3f", wer, mer, wil)
        return {"wer": wer, "mer": mer, "wil": wil}

    except ImportError:
        log.warning("jiwer not installed — skipping WER computation.")
        return {}
    except Exception as exc:
        log.error("WER computation failed: %s", exc)
        return {}

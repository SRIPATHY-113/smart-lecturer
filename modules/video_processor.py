"""
video_processor.py — Module 1: Visual Analysis Engine

Responsibilities
----------------
1. Download a YouTube lecture video via yt-dlp.
2. Extract one frame per second and detect slide changes via histogram comparison.
3. Apply Canny / Laplacian edge detection for OCR pre-processing.
4. Return a list of SlideChange objects with timestamps and frame paths.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    AUDIO_DIR,
    CANNY_HIGH,
    CANNY_LOW,
    FRAMES_DIR,
    HISTOGRAM_THRESHOLD,
    INPUT_VIDEO,
    LAPLACIAN_KSIZE,
    MIN_SLIDE_DURATION_S,
    YT_FORMAT,
    YT_OUTPUT_TEMPLATE,
)
from modules.utils import Timer, get_logger

log = get_logger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class SlideChange:
    """Represents a detected slide transition."""
    index: int
    timestamp_s: float
    frame_path: Path
    edge_path: Path                  # Canny edge image for OCR optimisation
    histogram: list[float] = field(default_factory=list, repr=False)


# ─── YouTube downloader ───────────────────────────────────────────────────────

def download_youtube_video(url: str, output_path: Path = INPUT_VIDEO) -> Path:
    """
    Download a YouTube video using yt-dlp.

    Args:
        url:         YouTube video URL.
        output_path: Where to save the mp4.

    Returns:
        Path to the downloaded video file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading video: %s", url)

    cmd = [
        "yt-dlp",
        "--format", YT_FORMAT,
        "--output", str(output_path),
        "--no-playlist",
        "--merge-output-format", "mp4",
        url,
    ]

    with Timer() as t:
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n{result.stderr}"
        )

    log.info("Download complete in %s → %s", t, output_path)
    return output_path


# ─── Histogram helpers ────────────────────────────────────────────────────────

def _compute_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    """
    Compute a normalised HSV histogram for a BGR frame.

    Using HSV is more robust to illumination changes than raw BGR channels.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared histogram distance (lower → more similar)."""
    return float(cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CHISQR_ALT,
    ))


# ─── Edge detection ───────────────────────────────────────────────────────────

def _canny_edges(frame: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur + Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)


def _laplacian_sharpness(frame: np.ndarray) -> float:
    """Return Laplacian variance — used to skip blurry/transition frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=LAPLACIAN_KSIZE).var())


# ─── Main slide-detection routine ─────────────────────────────────────────────

def extract_slide_changes(
    video_path: Path = INPUT_VIDEO,
    frames_dir: Path = FRAMES_DIR,
    sample_fps: int = 1,
    hist_threshold: float = HISTOGRAM_THRESHOLD,
    min_gap_s: float = MIN_SLIDE_DURATION_S,
) -> list[SlideChange]:
    """
    Extract frames where a slide change is detected.

    Algorithm
    ---------
    1. Sample one frame per `sample_fps` seconds.
    2. Compute HSV histogram for each sampled frame.
    3. Compare consecutive histograms using chi-squared distance.
    4. Flag a slide change when distance > `hist_threshold`.
    5. Enforce `min_gap_s` to suppress flickering transitions.
    6. For each slide change frame, save the raw frame and a Canny edge image.

    Args:
        video_path:     Path to the input video.
        frames_dir:     Where to save extracted frames.
        sample_fps:     How many frames per second to sample.
        hist_threshold: Chi-squared distance threshold for slide change.
        min_gap_s:      Minimum seconds between two detected slide changes.

    Returns:
        Ordered list of SlideChange objects.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps
    step         = max(1, int(fps / sample_fps))

    log.info(
        "Video: %.1fs  |  %.1f fps  |  %d total frames  |  sampling every %d frames",
        duration_s, fps, total_frames, step,
    )

    slide_changes: list[SlideChange] = []
    prev_hist: Optional[np.ndarray] = None
    last_change_s: float = -min_gap_s  # allow first frame
    frame_idx = 0
    slide_idx = 0

    with Timer() as t:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            timestamp_s = frame_idx / fps

            # Skip blurry / transition frames via Laplacian sharpness
            sharpness = _laplacian_sharpness(frame)
            if sharpness < 20.0 and prev_hist is not None:
                frame_idx += 1
                continue

            curr_hist = _compute_histogram(frame)

            is_change = (
                prev_hist is None  # always capture first frame
                or (
                    _histogram_distance(prev_hist, curr_hist) > hist_threshold
                    and (timestamp_s - last_change_s) >= min_gap_s
                )
            )

            if is_change:
                # Save raw slide frame
                frame_file = frames_dir / f"slide_{slide_idx:04d}_{int(timestamp_s):05d}s.jpg"
                cv2.imwrite(str(frame_file), frame)

                # Save Canny edge image for OCR optimisation
                edges = _canny_edges(frame)
                edge_file = frames_dir / f"edge_{slide_idx:04d}_{int(timestamp_s):05d}s.jpg"
                cv2.imwrite(str(edge_file), edges)

                slide_changes.append(SlideChange(
                    index=slide_idx,
                    timestamp_s=round(timestamp_s, 2),
                    frame_path=frame_file,
                    edge_path=edge_file,
                    histogram=curr_hist.tolist(),
                ))

                log.info(
                    "  Slide %d detected @ %ds  (sharpness=%.1f)",
                    slide_idx, int(timestamp_s), sharpness,
                )
                last_change_s = timestamp_s
                slide_idx += 1

            prev_hist = curr_hist
            frame_idx += 1

    cap.release()
    log.info("Extracted %d slide changes in %s", len(slide_changes), t)
    return slide_changes


# ─── Convenience: serialise slide changes for metadata.json ──────────────────

def slide_changes_to_dict(changes: list[SlideChange]) -> list[dict]:
    return [
        {
            "index": sc.index,
            "timestamp_s": sc.timestamp_s,
            "frame_path": str(sc.frame_path),
            "edge_path": str(sc.edge_path),
        }
        for sc in changes
    ]

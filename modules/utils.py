"""
utils.py — Helper utilities for Smart Lecturer.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Text helpers ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and strip non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_tokenize(text: str) -> list[str]:
    """Simple regex-based sentence splitter (no NLTK dependency needed at runtime)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """
    Sliding-window chunking over a list of sentences.

    Args:
        sentences: flat list of sentences.
        size:      number of sentences per chunk.
        overlap:   number of sentences shared between consecutive chunks.

    Returns:
        List of joined text chunks.
    """
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


# ─── Timing ───────────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ─── Numpy helpers ────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows of a 2-D embedding matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms

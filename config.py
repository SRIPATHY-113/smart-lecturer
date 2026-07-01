"""
config.py — Smart Lecturer Configuration
All tunable parameters in one place.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
FRAMES_DIR     = DATA_DIR / "frames"
AUDIO_DIR      = DATA_DIR / "audio"
TRANSCRIPTS_DIR= DATA_DIR / "transcripts"
INDEX_DIR      = BASE_DIR / "index"
VECTOR_DIR     = INDEX_DIR / "vector_store"
METADATA_PATH  = INDEX_DIR / "metadata.json"
INPUT_VIDEO    = DATA_DIR / "input_video.mp4"

# Ensure directories exist at import time
for d in [FRAMES_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, VECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Video / Frame Extraction ─────────────────────────────────────────────────
FRAME_RATE            = 1          # frames per second to sample for slide detection
HISTOGRAM_THRESHOLD   = 0.35       # chi-squared distance threshold for slide change
CANNY_LOW             = 50
CANNY_HIGH            = 150
LAPLACIAN_KSIZE       = 3
MIN_SLIDE_DURATION_S  = 3          # seconds — ignore flicker transitions

# ─── Audio / ASR ─────────────────────────────────────────────────────────────
WHISPER_MODEL         = "base"     # tiny | base | small | medium | large
AUDIO_FORMAT          = "wav"
AUDIO_SAMPLE_RATE     = 16_000     # Hz — Whisper expects 16 kHz
AUDIO_CHANNELS        = 1

# ─── Semantic Indexing ────────────────────────────────────────────────────────
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"   # sentence-transformers model
CHUNK_SIZE            = 5          # sentences per indexing chunk
CHUNK_OVERLAP         = 1          # overlapping sentences between chunks
TFIDF_MAX_FEATURES    = 5_000
PMI_MIN_COUNT         = 2          # minimum co-occurrence for PMI
PMI_WINDOW            = 4          # words in context window

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K                 = 5          # number of results to return per query
SIMILARITY_METRIC     = "cosine"   # cosine | l2

# ─── YouTube Download ────────────────────────────────────────────────────────
YT_FORMAT             = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"
YT_OUTPUT_TEMPLATE    = str(INPUT_VIDEO)

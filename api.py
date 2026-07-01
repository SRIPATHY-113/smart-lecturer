"""
api.py — Smart Lecturer Flask REST API
Wraps the existing pipeline modules with HTTP endpoints.

Run:
    pip install flask flask-cors
    python api.py

Endpoints:
    POST /api/process          { url }          → start pipeline
    GET  /api/status           → pipeline status + progress
    POST /api/query            { query, top_k } → semantic search
    GET  /api/slides           → list of detected slides
    GET  /api/transcript       → full transcript segments
    GET  /api/metadata         → index metadata + PMI
    GET  /api/frame/<index>    → slide frame image (base64)
"""

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── add smart_lecturer root to path ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent / "smart_lecturer"
sys.path.insert(0, str(ROOT))

from config import FRAMES_DIR, METADATA_PATH, TRANSCRIPTS_DIR, VECTOR_DIR
from modules.audio_processor import align_with_slides, extract_audio, transcribe
from modules.semantic_indexer import SemanticIndex, build_chunks_from_transcript
from modules.utils import load_json, save_json
from modules.video_processor import (
    download_youtube_video,
    extract_slide_changes,
    slide_changes_to_dict,
)

app = Flask(__name__)
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
pipeline_state = {
    "status": "idle",          # idle | running | done | error
    "stage": "",
    "progress": 0,             # 0-100
    "message": "",
    "error": None,
    "slides": [],
    "transcript_segments": [],
}

_index: SemanticIndex | None = None


# ── Pipeline runner (background thread) ───────────────────────────────────────

def run_pipeline(url: str):
    global _index, pipeline_state

    def update(stage, progress, message):
        pipeline_state.update(stage=stage, progress=progress, message=message)

    try:
        pipeline_state["status"] = "running"
        pipeline_state["error"] = None

        # Step 1: Download
        update("download", 5, "Downloading lecture video from YouTube…")
        video_path = download_youtube_video(url)

        # Step 2: Slide detection
        update("slides", 20, "Detecting slide changes (histogram analysis)…")
        slides = extract_slide_changes(video_path)
        pipeline_state["slides"] = slide_changes_to_dict(slides)
        slide_ts = [s.timestamp_s for s in slides]

        # Step 3: Audio extraction
        update("audio", 45, "Extracting audio stream (ffmpeg)…")
        audio_path = extract_audio(video_path)

        # Step 4: Transcription
        update("transcribe", 55, "Transcribing with Whisper ASR (this takes a few minutes)…")
        transcript = transcribe(audio_path)
        transcript = align_with_slides(transcript, slide_ts)
        pipeline_state["transcript_segments"] = [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "slide_index": s.slide_index,
            }
            for s in transcript.segments
        ]

        # Step 5: Build index
        update("index", 80, "Building semantic index (embeddings + FAISS + TF-IDF)…")
        chunks = build_chunks_from_transcript(transcript)
        _index = SemanticIndex()
        _index.build(chunks)
        _index.save()

        # Step 6: Save metadata
        update("metadata", 95, "Saving metadata…")
        meta = {
            "video_source": url,
            "num_slides": len(slides),
            "num_segments": len(transcript.segments),
            "num_chunks": len(chunks),
            "language": transcript.language,
            "top_pmi_collocations": [
                {"pair": list(pair), "pmi": round(score, 3)}
                for pair, score in _index.pmi.top_collocations(20)
            ],
            "slide_changes": pipeline_state["slides"],
        }
        save_json(meta, METADATA_PATH)

        update("done", 100, f"Pipeline complete — {len(slides)} slides, {len(chunks)} chunks indexed.")
        pipeline_state["status"] = "done"

    except Exception as exc:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = str(exc)
        pipeline_state["message"] = f"Error: {exc}"


def load_existing_index():
    """Load saved index on startup if it exists."""
    global _index, pipeline_state
    try:
        if (VECTOR_DIR / "faiss.index").exists():
            _index = SemanticIndex()
            _index.load()
            meta = load_json(METADATA_PATH) if METADATA_PATH.exists() else {}
            pipeline_state["status"] = "done"
            pipeline_state["progress"] = 100
            pipeline_state["message"] = f"Loaded existing index — {meta.get('num_chunks', '?')} chunks."
            if "slide_changes" in meta:
                pipeline_state["slides"] = meta["slide_changes"]
            # Load transcript
            for f in TRANSCRIPTS_DIR.glob("*.json"):
                with open(f) as fp:
                    data = json.load(fp)
                pipeline_state["transcript_segments"] = data.get("segments", [])
                break
    except Exception as e:
        pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/process", methods=["POST"])
def process():
    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    if pipeline_state["status"] == "running":
        return jsonify({"error": "Pipeline already running"}), 409

    thread = threading.Thread(target=run_pipeline, args=(url,), daemon=True)
    thread.start()
    return jsonify({"message": "Pipeline started", "status": "running"})


@app.route("/api/status")
def status():
    return jsonify({
        "status": pipeline_state["status"],
        "stage": pipeline_state["stage"],
        "progress": pipeline_state["progress"],
        "message": pipeline_state["message"],
        "error": pipeline_state["error"],
        "num_slides": len(pipeline_state["slides"]),
    })


@app.route("/api/query", methods=["POST"])
def query():
    global _index
    if _index is None:
        return jsonify({"error": "Index not ready. Run the pipeline first."}), 503

    data = request.json or {}
    q = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not q:
        return jsonify({"error": "query is required"}), 400

    results = _index.query(q, top_k=top_k)
    return jsonify({
        "query": q,
        "results": [
            {
                "rank": r.rank,
                "text": r.chunk.text,
                "slide_index": r.chunk.slide_index,
                "timestamp_s": r.chunk.timestamp_s,
                "timestamp_fmt": _fmt_time(r.chunk.timestamp_s),
                "vector_score": r.vector_score,
                "tfidf_score": r.tfidf_score,
                "combined_score": r.combined_score,
            }
            for r in results
        ]
    })


@app.route("/api/slides")
def slides():
    return jsonify({"slides": pipeline_state["slides"]})


@app.route("/api/transcript")
def transcript():
    return jsonify({"segments": pipeline_state["transcript_segments"]})


@app.route("/api/metadata")
def metadata():
    if METADATA_PATH.exists():
        return jsonify(load_json(METADATA_PATH))
    return jsonify({})


@app.route("/api/frame/<int:index>")
def frame(index):
    slides = pipeline_state["slides"]
    if index >= len(slides):
        return jsonify({"error": "Slide not found"}), 404
    path = Path(slides[index]["frame_path"])
    if not path.exists():
        return jsonify({"error": "Frame file not found"}), 404
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return jsonify({"image": f"data:image/jpeg;base64,{encoded}", "slide": slides[index]})


def _fmt_time(seconds):
    if seconds is None:
        return "??:??"
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


if __name__ == "__main__":
    load_existing_index()
    print("\n Smart Lecturer API running at http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)

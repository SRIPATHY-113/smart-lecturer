"""
app.py — Smart Lecturer: Main Application

Usage
-----
    # Full pipeline from a YouTube URL
    python app.py --url "https://www.youtube.com/watch?v=XXXXXXX"

    # Skip download (video already at data/input_video.mp4)
    python app.py --skip-download

    # Query the index after building
    python app.py --query "What is gradient descent?"

    # Evaluate WER with a reference transcript file
    python app.py --url "..." --reference transcript_ref.txt

    # Show top PMI collocations
    python app.py --pmi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    INPUT_VIDEO,
    METADATA_PATH,
    TOP_K,
    TRANSCRIPTS_DIR,
    VECTOR_DIR,
)
from modules.audio_processor import (
    align_with_slides,
    compute_wer,
    extract_audio,
    transcribe,
)
from modules.semantic_indexer import (
    SemanticIndex,
    build_chunks_from_transcript,
)
from modules.utils import Timer, get_logger, load_json, save_json
from modules.video_processor import (
    download_youtube_video,
    extract_slide_changes,
    slide_changes_to_dict,
)

log = get_logger("smart_lecturer")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    url: str | None = None,
    skip_download: bool = False,
    reference_path: Path | None = None,
) -> SemanticIndex:
    """
    End-to-end pipeline:
      1. Download YouTube video (optional).
      2. Extract slide changes (visual analysis).
      3. Extract audio and transcribe (ASR).
      4. Align transcript segments to slides.
      5. Optionally compute WER.
      6. Build semantic index (embeddings + TF-IDF + PMI).
      7. Save index and metadata.

    Args:
        url:             YouTube video URL. Required unless skip_download=True.
        skip_download:   If True, skip downloading and use existing video file.
        reference_path:  Optional path to ground-truth transcript for WER.

    Returns:
        A built and saved SemanticIndex ready for querying.
    """
    with Timer() as total_t:

        # ── Step 1: Download ──────────────────────────────────────────────────
        if not skip_download:
            if not url:
                raise ValueError("--url is required unless --skip-download is set.")
            download_youtube_video(url)
        else:
            if not INPUT_VIDEO.exists():
                raise FileNotFoundError(
                    f"No video found at {INPUT_VIDEO}. Provide --url or place a "
                    f"video at that path."
                )
            log.info("Using existing video: %s", INPUT_VIDEO)

        # ── Step 2: Visual analysis ───────────────────────────────────────────
        log.info("=" * 60)
        log.info("MODULE 1: Visual Analysis Engine")
        log.info("=" * 60)
        slide_changes = extract_slide_changes(INPUT_VIDEO)
        slide_timestamps = [sc.timestamp_s for sc in slide_changes]

        # ── Step 3: Audio extraction & transcription ──────────────────────────
        log.info("=" * 60)
        log.info("MODULE 2: Speech Transcription Engine")
        log.info("=" * 60)
        audio_path = extract_audio(INPUT_VIDEO)
        transcript = transcribe(audio_path)

        # WER evaluation (if reference provided)
        if reference_path and reference_path.exists():
            reference_text = reference_path.read_text(encoding="utf-8")
            wer_metrics = compute_wer(reference_text, transcript.full_text)
            log.info("WER metrics: %s", wer_metrics)
        else:
            wer_metrics = {}

        # ── Step 4: Slide alignment ───────────────────────────────────────────
        transcript = align_with_slides(transcript, slide_timestamps)

        # ── Step 5: Semantic indexing ─────────────────────────────────────────
        log.info("=" * 60)
        log.info("MODULE 3: Semantic Indexing & NLP")
        log.info("=" * 60)
        chunks = build_chunks_from_transcript(transcript)
        index  = SemanticIndex()
        index.build(chunks)

        # ── Step 6: Persist everything ────────────────────────────────────────
        index.save()

        # Write consolidated metadata
        meta = {
            "video_source": url or str(INPUT_VIDEO),
            "num_slides": len(slide_changes),
            "num_segments": len(transcript.segments),
            "num_chunks": len(chunks),
            "language": transcript.language,
            "wer_metrics": wer_metrics,
            "top_pmi_collocations": [
                {"pair": list(pair), "pmi": score}
                for pair, score in index.pmi.top_collocations(20)
            ],
            "slide_changes": slide_changes_to_dict(slide_changes),
        }
        save_json(meta, METADATA_PATH)
        log.info("Metadata saved to %s", METADATA_PATH)

    log.info("Pipeline complete in %s", total_t)
    return index


# ─── Query interface ──────────────────────────────────────────────────────────

def interactive_query(index: SemanticIndex) -> None:
    """Simple REPL for querying the index."""
    print("\n" + "═" * 60)
    print("  Smart Lecturer — Query Mode")
    print("  Type 'exit' to quit.")
    print("═" * 60 + "\n")

    while True:
        try:
            query = input("🎓 Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        results = index.query(query, top_k=TOP_K)
        if not results:
            print("  No results found.\n")
            continue

        print(f"\n  Top {len(results)} result(s) for: '{query}'\n")
        for r in results:
            slide_label = (
                f"slide {r.chunk.slide_index}" if r.chunk.slide_index is not None
                else "unknown slide"
            )
            ts = r.chunk.timestamp_s
            ts_label = f"{int(ts // 60):02d}:{int(ts % 60):02d}" if ts else "??:??"
            print(f"  [{r.rank}] combined={r.combined_score:.3f}  "
                  f"(vec={r.vector_score:.3f}, tfidf={r.tfidf_score:.3f})  "
                  f"@ {ts_label}  ({slide_label})")
            # Truncate long chunks for display
            preview = r.chunk.text[:200].replace("\n", " ")
            if len(r.chunk.text) > 200:
                preview += "…"
            print(f"      {preview}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart Lecturer — NLP lecture indexing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", type=str, default=None,
                   help="YouTube lecture URL to download and process.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip downloading; use existing video file.")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip the full pipeline; load an existing index.")
    p.add_argument("--query", type=str, default=None,
                   help="Run a single query and print results.")
    p.add_argument("--interactive", action="store_true",
                   help="Start an interactive query REPL.")
    p.add_argument("--reference", type=str, default=None,
                   help="Path to a ground-truth transcript for WER evaluation.")
    p.add_argument("--pmi", action="store_true",
                   help="Print top PMI collocations from the index metadata.")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Show PMI from saved metadata ─────────────────────────────────────────
    if args.pmi:
        if not METADATA_PATH.exists():
            log.error("No metadata found at %s. Run the pipeline first.", METADATA_PATH)
            sys.exit(1)
        meta = load_json(METADATA_PATH)
        print("\nTop PMI Collocations:")
        for item in meta.get("top_pmi_collocations", []):
            pair  = tuple(item["pair"])
            score = item["pmi"]
            print(f"  {pair[0]:15s} + {pair[1]:15s}  PMI={score:+.3f}")
        return

    # ── Load or build index ───────────────────────────────────────────────────
    index = SemanticIndex()

    if args.skip_pipeline:
        if not (VECTOR_DIR / "faiss.index").exists():
            log.error("No saved index found. Run the pipeline without --skip-pipeline first.")
            sys.exit(1)
        log.info("Loading existing index…")
        index.load()
    else:
        index = run_pipeline(
            url=args.url,
            skip_download=args.skip_download,
            reference_path=Path(args.reference) if args.reference else None,
        )

    # ── Single query ──────────────────────────────────────────────────────────
    if args.query:
        results = index.query(args.query, top_k=TOP_K)
        for r in results:
            ts = r.chunk.timestamp_s
            ts_label = f"{int(ts // 60):02d}:{int(ts % 60):02d}" if ts else "??:??"
            print(
                f"[{r.rank}] score={r.combined_score:.3f} | "
                f"slide={r.chunk.slide_index} | t={ts_label}\n"
                f"    {r.chunk.text[:300]}\n"
            )

    # ── Interactive REPL ──────────────────────────────────────────────────────
    elif args.interactive:
        interactive_query(index)


if __name__ == "__main__":
    main()

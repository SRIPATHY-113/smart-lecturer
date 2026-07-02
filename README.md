---
title: Smart Lecturer Backend
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: api.py
pinned: false
---

# 🎓 Smart Lecturer

> **A Multimodal AI Pipeline for Lecture Video Indexing and Semantic Search**

Smart Lecturer transforms any YouTube lecture video into a fully searchable, semantically indexed knowledge base — with **zero manual annotation**. Paste a YouTube URL, and the system automatically detects slide changes, transcribes speech, and enables natural language queries that return precise, timestamped answers.

---

## 📌 Project Context

| Field | Detail |
|---|---|
| **Subject** | Natural Language and Image Processing (NLIP) |
| **Type** | End-to-end Multimodal AI Pipeline |
| **Domain** | Educational Technology / Intelligent Tutoring Systems |

---

## ✨ Key Features

- 🖼 **Automated Slide Detection** — HSV histogram comparison + chi-squared distance to detect slide transitions without manual timestamps
- 🎙 **Speech Transcription** — OpenAI Whisper ASR converts lecture audio to timestamped text segments
- 🔗 **Slide-Speech Alignment** — Every spoken sentence is mapped to its active slide using temporal binary search
- 🧠 **Semantic Search** — Dense vector embeddings (Sentence-Transformers) + FAISS for sub-250ms natural language retrieval
- 📊 **TF-IDF + PMI** — Custom-built keyword weighting and collocation discovery for technical vocabulary extraction
- 🌐 **Web UI** — 5-tab interactive interface: Process, Query, Slides, Transcript, PMI Dashboard
- ⚡ **REST API** — Flask backend with 7 endpoints; index persists across sessions

---

## 🏗 System Architecture

```
YouTube URL
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Module 1 — Visual Analysis Engine                  │
│  yt-dlp download → OpenCV frame sampling →          │
│  HSV histogram comparison → Canny edge detection    │
└──────────────────────┬──────────────────────────────┘
                       │  slide timestamps + frame images
                       ▼
┌─────────────────────────────────────────────────────┐
│  Module 2 — Speech Transcription Engine             │
│  ffmpeg audio extraction → OpenAI Whisper ASR →     │
│  timestamped segments → slide alignment             │
└──────────────────────┬──────────────────────────────┘
                       │  aligned transcript
                       ▼
┌─────────────────────────────────────────────────────┐
│  Module 3 — Semantic Indexing & NLP                 │
│  Sentence-Transformers embeddings → FAISS index →   │
│  TF-IDF weighting → PMI collocation discovery       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              Natural Language Query Interface
              (CLI + Flask API + Web UI)
```

---

## 📁 Project Structure

```
smart_lecturer/
├── app.py                    # Main pipeline orchestrator (CLI)
├── api.py                    # Flask REST API (7 endpoints)
├── config.py                 # All tunable parameters
├── index.html                # Web UI (5 tabs, vanilla JS)
├── requirements.txt          # Python dependencies
├── modules/
│   ├── video_processor.py    # Module 1: Visual Analysis Engine
│   ├── audio_processor.py    # Module 2: Speech Transcription Engine
│   ├── semantic_indexer.py   # Module 3: Semantic Indexing & NLP
│   └── utils.py              # Shared utilities
├── data/
│   ├── frames/               # Detected slide images (raw + Canny edge)
│   ├── audio/                # Extracted WAV audio
│   └── transcripts/          # JSON + TXT transcripts
└── index/
    ├── vector_store/         # faiss.index, embeddings.npy, tfidf.pkl
    └── metadata.json         # Full pipeline run record
```

---

## 🛠 Technology Stack

| Component | Technology |
|---|---|
| Video Download | `yt-dlp` |
| Computer Vision | `OpenCV (cv2)` |
| Audio Extraction | `ffmpeg` |
| Speech Recognition | `OpenAI Whisper` (base model) |
| Sentence Embeddings | `Sentence-Transformers` — all-MiniLM-L6-v2 (384-dim) |
| Vector Index | `FAISS` — IndexFlatIP (exact cosine search) |
| TF-IDF | Custom implementation (from scratch) |
| PMI | Custom sliding-window co-occurrence (from scratch) |
| WER Evaluation | `jiwer` |
| Web API | `Flask` + `flask-cors` |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- ffmpeg installed and on PATH
- (Optional) Tesseract OCR for slide text extraction

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org and add to PATH
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smart_lecturer.git
cd smart_lecturer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-cache models (one-time, ~500 MB total)
python cache_model.py
```

---

## 🚀 Usage

### CLI — Full Pipeline

```bash
# Process a YouTube lecture from scratch
python app.py --url "https://www.youtube.com/watch?v=YOUR_LECTURE_ID"

# Interactive query REPL after processing
python app.py --url "https://..." --interactive

# Load existing index and query (no reprocessing)
python app.py --skip-pipeline --interactive

# Single query
python app.py --skip-pipeline --query "What is gradient descent?"

# Show top PMI collocations
python app.py --pmi
```

### Web UI

```bash
# Start the Flask API
python api.py

# Open index.html in your browser (double-click or open directly)
# API runs at http://localhost:5000
```

---

## 🔬 Core Algorithms

### Slide Detection
Frames sampled at 1 fps → HSV histogram (64×64 bins) → chi-squared distance comparison → slide change flagged when `distance > 0.35` and `gap > 3s` → Laplacian variance rejects blurry transition frames.

### Transcription & Alignment
ffmpeg extracts 16 kHz mono WAV → Whisper encoder-decoder Transformer → per-segment timestamps → binary search assigns each segment to its active slide.

### Hybrid Retrieval Scoring
```
combined_score = 0.6 × cosine_similarity(query_emb, chunk_emb)
              + 0.4 × tfidf_cosine(query, chunk)
```

### PMI Collocation Discovery
```
PMI(w1, w2) = log2( P(w1,w2) / (P(w1) × P(w2)) )
```
Sliding window of 4 words; minimum co-occurrence count = 2.

---

## 📊 Sample Results

Evaluated on a 7-minute 51-second YouTube lecture (25 fps, 11,795 frames):

| Metric | Value |
|---|---|
| Slide changes detected | 20 |
| Frames sampled | 471 (1 fps) |
| Transcript segments | 125 |
| Indexed chunks | 125 |
| Whisper transcription time | ~88 seconds (CPU) |
| Query response time | 150–250 ms |
| Estimated WER | ~10% (Whisper base) |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/process` | Start pipeline with `{ url }` |
| GET | `/api/status` | Pipeline progress (0–100%) |
| POST | `/api/query` | Semantic search `{ query, top_k }` |
| GET | `/api/slides` | List of detected slides |
| GET | `/api/transcript` | Full transcript segments |
| GET | `/api/metadata` | Index metadata + PMI collocations |
| GET | `/api/frame/<index>` | Slide frame as base64 image |

---

## 🔧 Configuration

All parameters are in `config.py`:

```python
WHISPER_MODEL         = "base"      # tiny | base | small | medium | large
HISTOGRAM_THRESHOLD   = 0.35        # slide change sensitivity
MIN_SLIDE_DURATION_S  = 3           # min seconds between slides
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"
CHUNK_SIZE            = 5           # sentences per index chunk
CHUNK_OVERLAP         = 1           # overlap between chunks
TFIDF_MAX_FEATURES    = 5_000
PMI_WINDOW            = 4           # co-occurrence window size
TOP_K                 = 5           # default query results count
```

---

## 🔮 Future Enhancements

- [ ] Real-time processing with `faster-whisper` + GPU
- [ ] Automated MCQ generation (T5/GPT fine-tune)
- [ ] OCR integration (Tesseract on Canny edge frames)
- [ ] Speaker diarisation (`pyannote.audio`)
- [ ] Multi-lecture corpus (ChromaDB / Pinecone)
- [ ] Video jump-links in UI (YouTube player seek on click)
- [ ] Knowledge graph from PMI collocations

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition model
- [Sentence-Transformers](https://www.sbert.net/) — Sentence embedding library
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube video downloader
- [OpenCV](https://opencv.org/) — Computer vision library

---

<p align="center">Built as an NLIP project — integrating Computer Vision, ASR, and NLP into a single multimodal pipeline.</p>

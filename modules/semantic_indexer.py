"""
semantic_indexer.py — Module 3: Semantic Indexing & NLP

Responsibilities
----------------
1. Generate dense embeddings via Sentence-Transformers.
2. Build a FAISS vector index for fast similarity retrieval.
3. Apply TF-IDF weighting for keyword relevance scoring.
4. Compute Pointwise Mutual Information (PMI) for collocation/term discovery.
5. Expose a unified query() interface returning ranked results with metadata.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    METADATA_PATH,
    PMI_MIN_COUNT,
    PMI_WINDOW,
    TFIDF_MAX_FEATURES,
    TOP_K,
    VECTOR_DIR,
)
from modules.utils import (
    Timer,
    chunk_sentences,
    get_logger,
    load_json,
    normalise_embeddings,
    save_json,
    sentence_tokenize,
)

log = get_logger(__name__)


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class IndexedChunk:
    """A text chunk stored in the index."""
    chunk_id: int
    text: str
    slide_index: Optional[int]
    timestamp_s: Optional[float]
    tfidf_score: float = 0.0


@dataclass
class RetrievalResult:
    """A single ranked retrieval result."""
    rank: int
    chunk: IndexedChunk
    vector_score: float     # cosine similarity from FAISS
    tfidf_score: float      # TF-IDF keyword relevance
    combined_score: float   # weighted combination


# ─── Embedding model (singleton) ──────────────────────────────────────────────

_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


# ─── TF-IDF ───────────────────────────────────────────────────────────────────

class TFIDFIndex:
    """
    Lightweight TF-IDF implementation.

    Builds an inverted index and stores IDF weights.  Scoring is BM25-style
    in spirit but uses the classic TF-IDF formula for clarity.

    TF(t,d)  = count(t in d) / |d|
    IDF(t)   = log( (1 + N) / (1 + df(t)) ) + 1        (sklearn-style smooth)
    TFIDF    = TF * IDF
    """

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.max_features = max_features
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])
        self._fitted = False

    # ── Tokenisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        import re
        tokens = re.findall(r"[a-z]+", text.lower())
        # Basic stopword removal
        stops = {
            "the","a","an","in","of","to","is","and","for","on","at",
            "with","this","that","are","was","be","as","by","from","or",
            "it","its","also","we","you","i","he","she","they","our",
        }
        return [t for t in tokens if t not in stops and len(t) > 1]

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> "TFIDFIndex":
        """Build vocabulary and IDF weights from a corpus of documents."""
        N = len(corpus)
        df: Counter = Counter()

        tokenised = []
        for doc in corpus:
            tokens = self._tokenise(doc)
            tokenised.append(tokens)
            df.update(set(tokens))

        # Select top-k terms by document frequency
        top_terms = [t for t, _ in df.most_common(self.max_features)]
        self.vocabulary_ = {t: i for i, t in enumerate(top_terms)}

        # Smooth IDF
        idfs = []
        for term in top_terms:
            idf = math.log((1 + N) / (1 + df[term])) + 1.0
            idfs.append(idf)
        self.idf_ = np.array(idfs, dtype=np.float32)

        self._fitted = True
        log.info("TF-IDF fitted: %d terms across %d docs.", len(self.vocabulary_), N)
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), vocab_size) TF-IDF matrix."""
        assert self._fitted, "Call fit() first."
        V = len(self.vocabulary_)
        matrix = np.zeros((len(texts), V), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = self._tokenise(text)
            if not tokens:
                continue
            tf_counts: Counter = Counter(tokens)
            for term, cnt in tf_counts.items():
                if term in self.vocabulary_:
                    col = self.vocabulary_[term]
                    tf = cnt / len(tokens)
                    matrix[row, col] = tf * self.idf_[col]

        return matrix

    def score_query(self, query: str, doc_vectors: np.ndarray) -> np.ndarray:
        """
        Score each document against a query using cosine TF-IDF similarity.

        Args:
            query:       Raw query string.
            doc_vectors: Pre-computed TF-IDF matrix (n_docs × vocab_size).

        Returns:
            1-D array of similarity scores, one per document.
        """
        q_vec = self.transform([query])[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return np.zeros(len(doc_vectors))
        q_vec /= q_norm

        doc_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        normed = doc_vectors / doc_norms

        return normed @ q_vec   # cosine similarities


# ─── PMI ──────────────────────────────────────────────────────────────────────

class PMIAnalyzer:
    """
    Pointwise Mutual Information for collocation discovery.

    PMI(w1, w2) = log2( P(w1,w2) / (P(w1) * P(w2)) )

    A positive PMI indicates that words co-occur more than chance.
    """

    def __init__(self, window: int = PMI_WINDOW, min_count: int = PMI_MIN_COUNT):
        self.window    = window
        self.min_count = min_count
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter  = Counter()
        self.total_tokens: int = 0

    def fit(self, corpus: list[str]) -> "PMIAnalyzer":
        import re
        for doc in corpus:
            tokens = re.findall(r"[a-z]+", doc.lower())
            self.unigram_counts.update(tokens)
            self.total_tokens += len(tokens)

            for i, w1 in enumerate(tokens):
                for w2 in tokens[i + 1 : i + 1 + self.window]:
                    pair = (w1, w2) if w1 < w2 else (w2, w1)
                    self.bigram_counts[pair] += 1

        log.info(
            "PMI fitted: %d unigrams, %d bigrams.",
            len(self.unigram_counts), len(self.bigram_counts),
        )
        return self

    def top_collocations(self, n: int = 20) -> list[tuple[tuple[str, str], float]]:
        """Return the top-n word pairs by PMI score."""
        N = self.total_tokens
        results = []

        for (w1, w2), co_count in self.bigram_counts.items():
            if co_count < self.min_count:
                continue
            p_w1  = self.unigram_counts[w1] / N
            p_w2  = self.unigram_counts[w2] / N
            p_w1w2 = co_count / N

            if p_w1 > 0 and p_w2 > 0:
                pmi = math.log2(p_w1w2 / (p_w1 * p_w2))
                results.append(((w1, w2), round(pmi, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def pmi(self, w1: str, w2: str) -> float:
        """PMI score for a specific word pair."""
        N = self.total_tokens
        if N == 0:
            return 0.0
        pair = (w1, w2) if w1 < w2 else (w2, w1)
        co  = self.bigram_counts.get(pair, 0)
        if co < self.min_count:
            return 0.0
        p_w1  = self.unigram_counts.get(w1, 0) / N
        p_w2  = self.unigram_counts.get(w2, 0) / N
        p_w1w2 = co / N
        if p_w1 == 0 or p_w2 == 0:
            return 0.0
        return math.log2(p_w1w2 / (p_w1 * p_w2))


# ─── Main Semantic Index ──────────────────────────────────────────────────────

class SemanticIndex:
    """
    End-to-end semantic index combining:
      • Sentence-Transformer dense embeddings + FAISS ANN search
      • TF-IDF sparse keyword scoring
      • PMI collocation discovery

    Build:
        idx = SemanticIndex()
        idx.build(chunks)          # list[IndexedChunk]

    Query:
        results = idx.query("what is backpropagation?", top_k=5)

    Persist / Load:
        idx.save()
        idx.load()
    """

    def __init__(
        self,
        vector_dir: Path = VECTOR_DIR,
        metadata_path: Path = METADATA_PATH,
        embedding_weight: float = 0.6,
        tfidf_weight: float = 0.4,
    ):
        self.vector_dir        = vector_dir
        self.metadata_path     = metadata_path
        self.embedding_weight  = embedding_weight
        self.tfidf_weight      = tfidf_weight

        self.chunks: list[IndexedChunk] = []
        self.embeddings: Optional[np.ndarray] = None

        self.tfidf   = TFIDFIndex()
        self.pmi     = PMIAnalyzer()
        self._faiss_index = None
        self._tfidf_matrix: Optional[np.ndarray] = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[IndexedChunk]) -> "SemanticIndex":
        """
        Index a list of text chunks.

        Steps
        -----
        1. Encode chunks with Sentence-Transformers.
        2. Build FAISS flat index (cosine via L2 on normalised vectors).
        3. Fit TF-IDF over all chunk texts.
        4. Fit PMI analyser.
        5. Assign TF-IDF scores per chunk.
        """
        import faiss

        self.chunks = chunks
        texts = [c.text for c in chunks]

        if not texts:
            raise ValueError("No chunks to index.")

        # 1. Dense embeddings
        log.info("Encoding %d chunks with '%s'…", len(texts), EMBEDDING_MODEL)
        encoder = _get_encoder()
        with Timer() as t:
            raw_embeddings = encoder.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True,
            )
        self.embeddings = normalise_embeddings(raw_embeddings).astype(np.float32)
        log.info("Embeddings shape %s computed in %s.", self.embeddings.shape, t)

        # 2. FAISS index (inner product on L2-normalised = cosine)
        dim = self.embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self.embeddings)
        log.info("FAISS index built with %d vectors (dim=%d).", len(texts), dim)

        # 3. TF-IDF
        with Timer() as t:
            self.tfidf.fit(texts)
            self._tfidf_matrix = self.tfidf.transform(texts)
        log.info("TF-IDF matrix %s built in %s.", self._tfidf_matrix.shape, t)

        # 4. PMI
        self.pmi.fit(texts)

        # 5. Assign per-chunk TF-IDF norm (a proxy for content richness)
        tfidf_norms = np.linalg.norm(self._tfidf_matrix, axis=1)
        for i, chunk in enumerate(self.chunks):
            chunk.tfidf_score = float(tfidf_norms[i])

        log.info("Index built for %d chunks.", len(chunks))
        return self

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K,
        alpha: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k most relevant chunks for a query.

        Scoring
        -------
        combined = alpha * cosine_similarity + (1-alpha) * tfidf_cosine

        Args:
            query_text: Natural-language question or keyword string.
            top_k:      Number of results to return.
            alpha:      Embedding weight (overrides instance default if given).

        Returns:
            Ranked list of RetrievalResult.
        """
        if self._faiss_index is None or self.embeddings is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        a = alpha if alpha is not None else self.embedding_weight

        # Dense retrieval — fetch 2× top_k then re-rank
        encoder    = _get_encoder()
        q_emb      = encoder.encode([query_text], convert_to_numpy=True)
        q_emb      = normalise_embeddings(q_emb).astype(np.float32)

        fetch_k    = min(len(self.chunks), top_k * 2)
        scores_vec, indices = self._faiss_index.search(q_emb, fetch_k)
        scores_vec = scores_vec[0]   # (fetch_k,)
        indices    = indices[0]

        # TF-IDF scores for the same candidates
        tfidf_scores_all = self.tfidf.score_query(
            query_text, self._tfidf_matrix[indices]
        )

        results = []
        for rank_pos, (idx, vec_score, tf_score) in enumerate(
            zip(indices, scores_vec, tfidf_scores_all)
        ):
            if idx < 0:
                continue
            combined = a * float(vec_score) + (1 - a) * float(tf_score)
            results.append(RetrievalResult(
                rank=rank_pos + 1,
                chunk=self.chunks[idx],
                vector_score=round(float(vec_score), 4),
                tfidf_score=round(float(tf_score), 4),
                combined_score=round(combined, 4),
            ))

        # Re-rank by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)
        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1

        return results[:top_k]

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save the FAISS index, embeddings, TF-IDF model, and metadata."""
        import faiss

        self.vector_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(
            self._faiss_index, str(self.vector_dir / "faiss.index")
        )
        np.save(str(self.vector_dir / "embeddings.npy"), self.embeddings)

        with open(self.vector_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(
                {
                    "tfidf": self.tfidf,
                    "tfidf_matrix": self._tfidf_matrix,
                    "pmi": self.pmi,
                },
                f,
            )

        metadata = {
            "num_chunks": len(self.chunks),
            "embedding_model": EMBEDDING_MODEL,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "slide_index": c.slide_index,
                    "timestamp_s": c.timestamp_s,
                    "tfidf_score": c.tfidf_score,
                }
                for c in self.chunks
            ],
        }
        save_json(metadata, self.metadata_path)
        log.info("Index saved to %s/", self.vector_dir)

    def load(self) -> "SemanticIndex":
        """Reload index from disk."""
        import faiss

        self._faiss_index = faiss.read_index(
            str(self.vector_dir / "faiss.index")
        )
        self.embeddings = np.load(str(self.vector_dir / "embeddings.npy"))

        with open(self.vector_dir / "tfidf.pkl", "rb") as f:
            blob = pickle.load(f)
            self.tfidf          = blob["tfidf"]
            self._tfidf_matrix  = blob["tfidf_matrix"]
            self.pmi            = blob["pmi"]

        metadata = load_json(self.metadata_path)
        self.chunks = [
            IndexedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                slide_index=c.get("slide_index"),
                timestamp_s=c.get("timestamp_s"),
                tfidf_score=c.get("tfidf_score", 0.0),
            )
            for c in metadata["chunks"]
        ]
        log.info(
            "Index loaded: %d chunks, %d vectors.", len(self.chunks), self._faiss_index.ntotal
        )
        return self


# ─── Helper: build chunks from a Transcript ───────────────────────────────────

def build_chunks_from_transcript(
    transcript,      # audio_processor.Transcript
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> list[IndexedChunk]:
    """
    Convert a Transcript object into IndexedChunk objects for indexing.

    Each chunk is a sliding window of `chunk_size` sentences with `overlap`
    sentences shared between adjacent chunks.  Timing metadata from the
    first sentence in the window is propagated to the chunk.
    """
    chunks: list[IndexedChunk] = []
    chunk_id = 0

    for seg in transcript.segments:
        sentences = sentence_tokenize(seg.text)
        windows   = chunk_sentences(sentences, chunk_size, overlap)

        for text in windows:
            if not text.strip():
                continue
            chunks.append(IndexedChunk(
                chunk_id=chunk_id,
                text=text,
                slide_index=seg.slide_index,
                timestamp_s=seg.start,
            ))
            chunk_id += 1

    log.info("Built %d chunks from transcript.", len(chunks))
    return chunks

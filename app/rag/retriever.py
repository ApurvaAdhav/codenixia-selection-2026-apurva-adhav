"""
app/rag/retriever.py
---------------------
Step: "RAG" using a small LOCAL knowledge base + FAISS (no cloud vector DB,
per the challenge constraints).

How it works:
  1. On startup (or first use), read all .md files in data/knowledge_base/.
  2. Chunk each file by paragraph/section (split on "## " headers - each
     section is already a self-contained topic, which makes for clean,
     coherent chunks without needing a fancy splitter).
  3. Embed each chunk with a local TF-IDF vectorizer (scikit-learn).
  4. Build a FAISS IndexFlatIP (cosine similarity via normalized vectors)
     over the dense TF-IDF vectors.
  5. search_business_knowledge(query) embeds the query with the SAME fitted
     vectorizer and returns the top-k most similar chunks as plain text +
     source file, for the LLM to ground its answer on.

DECISION: TF-IDF instead of a neural embedding model (e.g. sentence-
transformers). This is documented in DECISION_LOG.md - in short: the
knowledge base is small (a handful of markdown files), TF-IDF requires NO
model download / internet access / GPU, keeps the project fully offline
and dependency-light (per the "keep it simple" instruction), and is easily
swappable later - only this file would need to change to plug in a neural
embedding model, since the rest of the app only calls search().

The index is built once and cached in memory (module-level singleton),
since the knowledge base is small and static for this MVP.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import KNOWLEDGE_BASE_DIR, RAG_TOP_K
from app.logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class KnowledgeChunk:
    text: str
    source: str
    heading: str


class KnowledgeBaseRetriever:
    """Loads markdown docs, embeds them with TF-IDF, and serves similarity
    search over FAISS. Fully local/offline - no model download required."""

    def __init__(self, kb_dir: Path = KNOWLEDGE_BASE_DIR):
        self.kb_dir = kb_dir
        self._vectorizer: TfidfVectorizer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[KnowledgeChunk] = []
        self._built = False

    def _embed(self, texts: list[str], fit: bool) -> np.ndarray:
        """Vectorize texts with TF-IDF, then L2-normalize so that FAISS
        inner-product search is equivalent to cosine similarity."""
        if fit:
            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
            matrix = self._vectorizer.fit_transform(texts)
        else:
            matrix = self._vectorizer.transform(texts)

        dense = matrix.toarray().astype("float32")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for empty/no-overlap queries
        return dense / norms

    def _load_chunks(self) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        if not self.kb_dir.exists():
            logger.warning("Knowledge base directory %s does not exist", self.kb_dir)
            return chunks

        for path in sorted(self.kb_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            # Split on level-2 markdown headers ("## Heading") - each section
            # is one coherent, retrievable chunk.
            sections = re.split(r"\n(?=## )", text)
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                heading_match = re.match(r"##?\s*(.+)", section)
                heading = heading_match.group(1).strip() if heading_match else path.stem
                chunks.append(KnowledgeChunk(text=section, source=path.name, heading=heading))

        logger.info("Loaded %d knowledge chunks from %s", len(chunks), self.kb_dir)
        return chunks

    def build(self) -> None:
        """Build (or rebuild) the FAISS index from the knowledge base files."""
        self._chunks = self._load_chunks()
        if not self._chunks:
            logger.warning("No knowledge base chunks found; RAG will return empty results.")
            self._index = None
            self._built = True
            return

        texts = [c.text for c in self._chunks]
        embeddings = self._embed(texts, fit=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
        index.add(embeddings)

        self._index = index
        self._built = True
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(texts), dim)

    def ensure_built(self) -> None:
        if not self._built:
            self.build()

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Return top-k knowledge base chunks most relevant to `query`."""
        self.ensure_built()
        top_k = top_k or RAG_TOP_K

        if self._index is None or not self._chunks:
            return []

        query_emb = self._embed([query], fit=False)

        scores, indices = self._index.search(query_emb, min(top_k, len(self._chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "heading": chunk.heading,
                "relevance_score": round(float(score), 3),
            })
        return results


# Module-level singleton so the FastAPI app and Streamlit app share one
# built index per process instead of rebuilding on every request.
_retriever_singleton: KnowledgeBaseRetriever | None = None


def get_retriever() -> KnowledgeBaseRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = KnowledgeBaseRetriever()
    return _retriever_singleton


def search_business_knowledge(query: str, top_k: int = None) -> list[dict]:
    """Public function used directly by the agent tool of the same name."""
    return get_retriever().search(query, top_k=top_k)

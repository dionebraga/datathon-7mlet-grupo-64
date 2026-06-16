"""Retrieval over the synthetic internal policy documents (RAG).

A dependency-light retriever: it chunks the markdown policies, builds a TF-IDF
index and returns the top-k most relevant chunks for a query. The retrieved
snippets ground the assistant's explanations and are returned as citations, so
answers are traceable to a source document (no ungrounded claims).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICIES_DIR = Path(__file__).resolve().parent / "policies"


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


def _chunk_markdown(text: str, max_chars: int = 500) -> list[str]:
    """Split a doc into section-ish chunks (by headings then by size)."""
    blocks, current = [], []
    for line in text.splitlines():
        if line.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    # further split overly long blocks
    chunks: list[str] = []
    for b in blocks:
        if len(b) <= max_chars:
            chunks.append(b)
        else:
            for i in range(0, len(b), max_chars):
                chunks.append(b[i : i + max_chars])
    return [c for c in chunks if c.strip()]


class PolicyRAG:
    """TF-IDF retriever over the synthetic policy corpus."""

    def __init__(self, policies_dir: Path | None = None) -> None:
        self.dir = policies_dir or POLICIES_DIR
        self.sources: list[str] = []
        self.chunks: list[str] = []
        self._load()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def _load(self) -> None:
        for md in sorted(self.dir.glob("*.md")):
            text = md.read_text(encoding="utf-8")
            for chunk in _chunk_markdown(text):
                self.sources.append(md.name)
                self.chunks.append(chunk)
        if not self.chunks:  # safety: never index an empty corpus
            self.sources = ["empty"]
            self.chunks = ["Nenhuma política sintética encontrada."]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        order = sims.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(source=self.sources[i], text=self.chunks[i], score=round(float(sims[i]), 4))
            for i in order
            if sims[i] > 0
        ]

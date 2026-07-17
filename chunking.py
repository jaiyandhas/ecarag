"""
Semantic chunking for ECA-RAG.

Implements the CA-RAG-style chunking algorithm directly (not via a
LangChain black box) so it can be explained and defended in the review:

  1. Split text into sentences.
  2. Embed each sentence.
  3. Walk through sentences in order; compute cosine similarity between
     the current sentence and the running embedding of the current chunk.
  4. If similarity drops below `threshold`, close the current chunk and
     start a new one.

This mirrors the "Semantic Chunking" algorithm described in Collini et
al. (CA-RAG, IEEE Access 2025), which our project extends with a
cross-encoder re-ranking stage downstream (see validation.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Sentence splitting (regex-based, no spaCy model download required so this
# runs offline / on a demo laptop without extra setup).
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z0-9\"'(])"
)


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter. Good enough for reports/manuals;
    swap in spaCy or nltk.punkt if you have model downloads available."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


@dataclass
class Chunk:
    id: int
    text: str
    sentences: List[str] = field(default_factory=list)


def semantic_chunk(
    text: str,
    embed_fn,
    threshold: float = 0.4,
) -> List[Chunk]:
    """
    Split `text` into semantically coherent chunks.

    Parameters
    ----------
    text : str
        Full document text.
    embed_fn : Callable[[List[str]], np.ndarray]
        Batch embedding function, e.g. SBERT model.encode. Must return an
        (n, d) array of L2-normalizable vectors.
    threshold : float
        Cosine similarity threshold (theta). Below this, a new chunk starts.
        Empirically, paraphrase pairs sit ~0.7-0.9, unrelated pairs ~0,
        so 0.3-0.5 gives reasonable chunk granularity (see CA-RAG paper).

    Returns
    -------
    List[Chunk]
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    
    if len(sentences) < 3:
        return [Chunk(id=0, text=" ".join(sentences), sentences=sentences)]

    embeddings = embed_fn(sentences)
    embeddings = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )

    chunks: List[Chunk] = []
    current_sentences: List[str] = [sentences[0]]
    running_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = float(np.dot(running_embedding, embeddings[i]))
        if sim < threshold:
            chunks.append(
                Chunk(
                    id=len(chunks),
                    text=" ".join(current_sentences),
                    sentences=current_sentences,
                )
            )
            current_sentences = [sentences[i]]
            running_embedding = embeddings[i]
        else:
            current_sentences.append(sentences[i])
            # running embedding = mean of sentences seen so far in this chunk
            idx_start = i - len(current_sentences) + 1
            running_embedding = embeddings[idx_start : i + 1].mean(axis=0)
            running_embedding = running_embedding / (
                np.linalg.norm(running_embedding) + 1e-12
            )

    chunks.append(
        Chunk(
            id=len(chunks),
            text=" ".join(current_sentences),
            sentences=current_sentences,
        )
    )
    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from an uploaded PDF (pypdf, no OCR)."""
    from pypdf import PdfReader
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)

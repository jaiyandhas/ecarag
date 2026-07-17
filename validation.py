"""
Adaptive Similarity Validation — the core ECA-RAG contribution.

Two-stage scoring of (candidate_answer, source_chunk) pairs:

  Stage 1 (CA-RAG baseline): bi-encoder cosine similarity between the
      SBERT embedding of the candidate answer and its source chunk.
      Candidates below `theta` are discarded as unsupported / hallucinated.

  Stage 2 (ECA-RAG addition): surviving candidates are re-scored with a
      cross-encoder, which jointly encodes (answer, chunk) rather than
      embedding them independently — this catches cases where two texts
      share vocabulary (high bi-encoder similarity) but the answer isn't
      actually well-grounded in the chunk's claims.

Final ranking = weighted combination of both scores. We keep the
CA-RAG-only ranking (bi-encoder alone) alongside the ECA-RAG ranking so
the demo can show, side by side, when the cross-encoder stage changes
which answer wins — that comparison is the whole point of the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ScoredCandidate:
    chunk_id: int
    chunk_text: str
    answer: str
    bi_encoder_score: float
    cross_encoder_score: Optional[float] = None
    combined_score: Optional[float] = None
    passed_bi_filter: bool = False


def bi_encoder_scores(
    answers: List[str], chunks: List[str], embed_fn
) -> List[float]:
    """Cosine similarity between each (answer, chunk) pair, batched."""
    a_emb = embed_fn(answers)
    c_emb = embed_fn(chunks)
    a_emb = a_emb / (np.linalg.norm(a_emb, axis=1, keepdims=True) + 1e-12)
    c_emb = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-12)
    return [float(np.dot(a, c)) for a, c in zip(a_emb, c_emb)]


def cross_encoder_scores(
    answers: List[str], chunks: List[str], cross_encoder_predict_fn
) -> List[float]:
    """Score each (answer, chunk) pair with a cross-encoder.

    `cross_encoder_predict_fn` should behave like
    CrossEncoder(...).predict(list_of_pairs) -> np.ndarray of raw scores.
    Cross-encoder relevance scores are unbounded logits, not [0,1]
    similarities, so we squash with a sigmoid before combining with the
    bounded bi-encoder score.
    """
    pairs = [[a, c] for a, c in zip(answers, chunks)]
    raw = cross_encoder_predict_fn(pairs)
    return [1.0 / (1.0 + np.exp(-float(s))) for s in raw]


def validate_and_rank(
    chunk_texts: List[str],
    candidate_answers: List[str],
    embed_fn,
    cross_encoder_predict_fn,
    theta: float = 0.6,
    cross_encoder_weight: float = 0.5,
) -> dict:
    """
    Run the full two-stage validation pipeline.

    Returns a dict with:
      - "ca_rag_only": ranking using bi-encoder score alone (baseline)
      - "eca_rag": ranking using the combined bi+cross-encoder score
      - "disagree": bool, True if the top pick differs between the two
    """
    bi_scores = bi_encoder_scores(candidate_answers, chunk_texts, embed_fn)

    candidates = [
        ScoredCandidate(
            chunk_id=i,
            chunk_text=chunk_texts[i],
            answer=candidate_answers[i],
            bi_encoder_score=bi_scores[i],
            passed_bi_filter=bi_scores[i] >= theta,
        )
        for i in range(len(chunk_texts))
    ]

    # --- CA-RAG-only baseline ranking (bi-encoder score, no filter applied
    # to ranking itself — matches CA-RAG's Technique 1/2 of picking highest
    # similarity among survivors of the threshold) ---
    ca_rag_survivors = [c for c in candidates if c.passed_bi_filter] or candidates
    ca_rag_ranked = sorted(
        ca_rag_survivors, key=lambda c: c.bi_encoder_score, reverse=True
    )

    # --- ECA-RAG: cross-encoder re-rank on top of bi-encoder survivors ---
    survivors = [c for c in candidates if c.passed_bi_filter] or candidates
    if survivors:
        ce_scores = cross_encoder_scores(
            [c.answer for c in survivors],
            [c.chunk_text for c in survivors],
            cross_encoder_predict_fn,
        )
        for c, ce in zip(survivors, ce_scores):
            c.cross_encoder_score = ce
            c.combined_score = (
                (1 - cross_encoder_weight) * c.bi_encoder_score
                + cross_encoder_weight * ce
            )
    eca_ranked = sorted(
        survivors, key=lambda c: (c.combined_score or 0.0), reverse=True
    )

    top_ca_rag = ca_rag_ranked[0] if ca_rag_ranked else None
    top_eca_rag = eca_ranked[0] if eca_ranked else None
    disagree = (
        top_ca_rag is not None
        and top_eca_rag is not None
        and top_ca_rag.chunk_id != top_eca_rag.chunk_id
    )

    return {
        "all_candidates": candidates,
        "ca_rag_only": ca_rag_ranked,
        "eca_rag": eca_ranked,
        "disagree": disagree,
    }

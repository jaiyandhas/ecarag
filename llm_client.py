"""
Candidate answer generation.

Given a question and a chunk of context, produce a candidate answer.
Supports two backends:
  1. Google Gemini 1.5 Flash (via google-generativeai API)
  2. Smart Similarity Extractor (instant local fallback using SBERT)
"""

from __future__ import annotations

import os
import sys

_CANDIDATE_PROMPT = """You are answering a question using ONLY the context \
chunk below. If the chunk does not contain enough information to answer, \
say so explicitly rather than guessing.

Context chunk:
\"\"\"
{chunk}
\"\"\"

Question: {question}

Answer in 2-4 sentences, using only the context above:"""


def smart_similarity_extractor(question: str, chunk_text: str, embed_fn) -> str:
    """
    Zero-setup, instant offline fallback.
    Extracts the top 2 sentences from the chunk that have the highest cosine similarity
    to the question, preserving their original order in the chunk.
    """
    from chunking import split_sentences
    import numpy as np

    sentences = split_sentences(chunk_text)
    if not sentences:
        return "No text available in this chunk."
    if len(sentences) <= 2:
        return chunk_text

    # Embed question and sentences
    q_emb = embed_fn([question])[0]
    s_embs = embed_fn(sentences)

    # Cosine similarities
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    s_norms = s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-12)

    sims = np.dot(s_norms, q_norm)
    
    # Pick top 2 sentences (preserving original order)
    top_indices = np.argsort(sims)[-2:]
    top_indices = sorted(top_indices)

    selected = [sentences[idx] for idx in top_indices]
    return " ".join(selected)


def generate_candidate_answer(
    question: str,
    chunk_text: str,
    backend: str = "gemini",
    api_key: str | None = None,
    embed_fn = None,
    model: str = "gemini-3.5-flash",
) -> str:
    """Generate one candidate answer conditioned on a single chunk using selected backend."""
    if backend == "extractor":
        if embed_fn is None:
            raise ValueError("embed_fn is required for the similarity extractor backend.")
        return smart_similarity_extractor(question, chunk_text, embed_fn)

    # Gemini backend
    # Default to the key provided by the user if not specified or in environment
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Please provide it in the sidebar or set the GEMINI_API_KEY env var."
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Configure model
        model_instance = genai.GenerativeModel(model)
        prompt = _CANDIDATE_PROMPT.format(chunk=chunk_text, question=question)
        
        response = model_instance.generate_content(
            prompt,
            generation_config={"max_output_tokens": 300},
            request_options={"timeout": 15.0}
        )
        return response.text.strip()
    except Exception as e:
        return f"[Generation failed: {str(e)}]"


#!/usr/bin/env python3
"""
Pre-download and cache sentence-transformer models for ECA-RAG.
Ensures no network delays or timeouts during a live demo presentation.
"""

import sys

def main():
    print("Pre-downloading SentenceTransformer bi-encoder model ('all-MiniLM-L6-v2')...")
    try:
        from sentence_transformers import SentenceTransformer
        bi_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Bi-encoder loaded successfully!")
    except Exception as e:
        print(f"Error loading bi-encoder: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nPre-downloading CrossEncoder model ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
    try:
        from sentence_transformers import CrossEncoder
        ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Cross-encoder loaded successfully!")
    except Exception as e:
        print(f"Error loading cross-encoder: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAll models cached successfully. Ready for demo!")

if __name__ == "__main__":
    main()

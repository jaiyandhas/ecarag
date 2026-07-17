# ECA-RAG

**Enhanced Context-Aware Retrieval Augmented Generation with Adaptive Similarity Validation**

ECA-RAG extends [CA-RAG (Collini et al., IEEE Access 2025)](https://doi.org/10.1109/ACCESS.2025.3526209) by adding a **cross-encoder re-ranking stage** on top of CA-RAG's bi-encoder similarity filter. The result is a two-stage validation pipeline that catches cases where high lexical overlap fools the bi-encoder — cases the cross-encoder, attending to both texts jointly, correctly identifies as low-support answers.

> **Review note:** See [`NOVELTY.md`](NOVELTY.md) for an honest side-by-side comparison of what this project does vs. what CA-RAG already does, and what you need before Module 2/3 evaluations.

---

## Architecture

```
Document
   │
   ▼
Semantic Chunking          ← Algorithm 1 (Collini et al.), implemented from scratch
   │                          in chunking.py — not a LangChain wrapper
   ▼
Per-Chunk Candidate Gen    ← one LLM call per chunk (Gemini Flash or offline extractor)
   │
   ▼
Stage 1 — Bi-encoder       ← cosine similarity (answer ↔ chunk); discard below θ
   │                                           ↑ CA-RAG stops here
   ▼
Stage 2 — Cross-encoder    ← jointly encodes (answer, chunk); re-scores survivors
   │                                           ↑ ECA-RAG addition
   ▼
Combined Ranking + Side-by-side Comparison (CA-RAG vs ECA-RAG)
```

---

## Models used

| Role | Model |
|---|---|
| Bi-encoder (Stage 1) | `all-MiniLM-L6-v2` (sentence-transformers) |
| Cross-encoder (Stage 2) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Google Gemini Flash · or offline similarity extractor |

Both sentence-transformer models are downloaded and cached on first run (or pre-cached via `warmup.py`).

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/jaiyandhas/ecarag.git
cd ecarag

# 2. Create virtualenv
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Recommended) Pre-cache models — avoids download delay on first run
python warmup.py

# 5. Set your Gemini API key (optional — the offline extractor works without it)
export GEMINI_API_KEY=your_key_here
# or copy .env.example → .env and fill it in

# 6. Run
streamlit run app.py
```

Opens at **http://localhost:8501**. A bundled sample document is pre-selected so you can demo immediately without uploading anything.

---

## Features

| Feature | Description |
|---|---|
| **Semantic chunking** | Sentence-level cosine similarity walk; configurable threshold |
| **PDF + TXT upload** | Drag-and-drop document upload or use the bundled sample |
| **Two answer backends** | Gemini Flash (online) or similarity extractor (100% offline) |
| **Side-by-side comparison** | CA-RAG pick vs ECA-RAG pick, with score breakdown |
| **Ablation scan** | Batch-runs sample questions and reports which ones diverge between stages |
| **Tunable parameters** | θ (bi-encoder threshold), chunking threshold, cross-encoder weight — all in the sidebar |

---

## Project structure

```
app.py          — Streamlit UI; wires chunking → generation → validation
chunking.py     — Semantic chunking (Algorithm 1 style, written from scratch)
llm_client.py   — Candidate answer generation (Gemini Flash or offline extractor)
validation.py   — Bi-encoder filter + cross-encoder re-rank + CA-RAG/ECA-RAG comparison
warmup.py       — Pre-downloads and caches sentence-transformer models
sample_doc.txt  — Demo document (self-referential RAG explainer)
NOVELTY.md      — Honest positioning of ECA-RAG vs CA-RAG baseline
```

---

## Configuration

| Parameter | Default | What it controls |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key (env var or sidebar input) |
| Bi-encoder threshold θ | 0.6 | Candidates below this cosine score are discarded |
| Chunking threshold | 0.4 | Lower = larger chunks (new chunk starts when similarity drops below this) |
| Cross-encoder weight | 0.5 | 0 = bi-encoder only (CA-RAG), 1 = cross-encoder only |

---

## Known limitations

- **No benchmark evaluation yet** — quantitative ablation (TriviaQA/SQuAD) is Module 2/3 work.
- **Static thresholds** — θ and chunking threshold are manually set; adaptive learning is a stretch goal and flagged as future work.
- **Single LLM call per chunk, no batching** — fine for demo documents; would need batching for large inputs.

---

## Reference

Collini, L., Kurniadi, F. I., Nesi, P., & Pantaleo, G. (2025). *Context-Aware Retrieval Augmented Generation using Similarity Validation*. IEEE Access. https://doi.org/10.1109/ACCESS.2025.3526209

"""
ECA-RAG — Enhanced Context-Aware Retrieval Augmented Generation
with Adaptive Similarity Validation.

Extends CA-RAG (Collini et al., IEEE Access 2025) with a cross-encoder
re-ranking stage on top of the bi-encoder similarity filter.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from chunking import semantic_chunk, extract_text_from_pdf
from llm_client import generate_candidate_answer
from validation import validate_and_rank

st.set_page_config(
    page_title="ECA-RAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Remove Streamlit default top padding */
    .block-container { padding-top: 2rem; }

    .page-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .method-card {
        padding: 1.25rem 1.5rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        margin-bottom: 0.75rem;
    }
    .method-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .method-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.6rem;
    }
    .method-score {
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 0.6rem;
    }
    .method-answer {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.6;
        border-top: 1px solid #f1f5f9;
        padding-top: 0.6rem;
        margin-top: 0.4rem;
    }
    .diverge-note {
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        background: #fffbeb;
        border: 1px solid #fcd34d;
        color: #78350f;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }
    .agree-note {
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def embed_fn_factory(model):
    return lambda texts: model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cross_encoder_fn_factory(model):
    return lambda pairs: model.predict(pairs, show_progress_bar=False)


SAMPLE_QUESTIONS = [
    "What is semantic drift in traditional RAG?",
    "How does CA-RAG validate candidate answers after generation?",
    "What is the key limitation of using only a bi-encoder for similarity validation?",
    "Why is a cross-encoder better at catching deceptively high similarity scores?",
    "Does CA-RAG use a cross-encoder for re-ranking?",
]

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">ECA-RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Enhanced Context-Aware Retrieval Augmented Generation '
    'with Adaptive Similarity Validation &nbsp;·&nbsp; '
    'Extends CA-RAG (Collini et al., IEEE Access 2025)</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    backend = st.radio(
        "Answer generation",
        options=["gemini", "extractor"],
        format_func=lambda x: (
            "Gemini 3.5 Flash" if x == "gemini" else "Similarity Extractor (offline)"
        ),
    )

    gemini_key = ""
    if backend == "gemini":
        gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="Paste your Gemini API key here…",
        )

    st.divider()

    theta = st.slider(
        "Bi-encoder threshold (θ)", 0.0, 1.0, 0.6, 0.05,
        help="Candidates scoring below θ are discarded before re-ranking.",
    )
    chunk_threshold = st.slider(
        "Chunking threshold", 0.0, 1.0, 0.4, 0.05,
        help="Cosine similarity at which a new chunk begins. Lower = larger chunks.",
    )
    ce_weight = st.slider(
        "Cross-encoder weight", 0.0, 1.0, 0.5, 0.05,
        help="Weight of cross-encoder score in final combined rank. "
             "0 = bi-encoder only (CA-RAG baseline), 1 = cross-encoder only.",
    )

# ── Document + Query ─────────────────────────────────────────────────────────
col_doc, col_q = st.columns([4, 5])

with col_doc:
    st.subheader("Document")
    uploaded = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    use_sample = st.checkbox("Use bundled sample document", value=not bool(uploaded))

    doc_text = ""
    if use_sample:
        try:
            with open("sample_doc.txt") as f:
                doc_text = f.read()
        except FileNotFoundError:
            st.error("sample_doc.txt not found. Upload a document instead.")
    elif uploaded is not None:
        if uploaded.type == "application/pdf":
            try:
                doc_text = extract_text_from_pdf(uploaded.read())
            except Exception as e:
                st.error(f"Could not read PDF: {e}")
        else:
            try:
                doc_text = uploaded.read().decode("utf-8")
            except Exception as e:
                st.error(f"Could not decode file: {e}")

    if doc_text:
        st.text_area("Preview", doc_text, height=150, disabled=True, label_visibility="collapsed")

with col_q:
    st.subheader("Query")
    question = st.text_input(
        "Question",
        value="What is the key limitation of using only a bi-encoder for similarity validation?",
        label_visibility="collapsed",
    )
    run_pipeline = st.button("Run pipeline", type="primary", width="stretch")

# ── Model loading ─────────────────────────────────────────────────────────────
models_loaded = False
if doc_text:
    _status = st.empty()
    _status.caption("Loading models…")
    bi_model = load_bi_encoder()
    cross_model = load_cross_encoder()
    embed_fn = embed_fn_factory(bi_model)
    cross_fn = cross_encoder_fn_factory(cross_model)
    _status.empty()
    models_loaded = True

# ── Ablation scan ─────────────────────────────────────────────────────────────
if models_loaded and doc_text:
    with st.expander("Ablation scan — bi-encoder vs. cross-encoder divergence"):
        st.markdown(
            "Runs the two-stage validation pipeline on a set of sample questions using the "
            "offline similarity extractor and reports which questions produce a different "
            "top-ranked answer between CA-RAG (bi-encoder only) and ECA-RAG (combined score)."
        )
        if st.button("Run scan"):
            scan_chunks = semantic_chunk(doc_text, embed_fn, threshold=chunk_threshold)
            scan_chunk_texts = [c.text for c in scan_chunks]
            rows = []
            prog = st.progress(0)
            for qi, q in enumerate(SAMPLE_QUESTIONS):
                cands = [
                    generate_candidate_answer(
                        question=q,
                        chunk_text=chunk.text,
                        backend="extractor",
                        embed_fn=embed_fn,
                    )
                    for chunk in scan_chunks
                ]
                res = validate_and_rank(
                    chunk_texts=scan_chunk_texts,
                    candidate_answers=cands,
                    embed_fn=embed_fn,
                    cross_encoder_predict_fn=cross_fn,
                    theta=theta,
                    cross_encoder_weight=ce_weight,
                )
                ca_top = res["ca_rag_only"][0].chunk_id if res["ca_rag_only"] else "—"
                eca_top = res["eca_rag"][0].chunk_id if res["eca_rag"] else "—"
                rows.append({
                    "Question": q,
                    "CA-RAG top chunk": ca_top,
                    "ECA-RAG top chunk": eca_top,
                    "Diverges?": "Yes" if res["disagree"] else "No",
                })
                prog.progress((qi + 1) / len(SAMPLE_QUESTIONS))
            prog.empty()
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_pipeline and doc_text and question:
    if len(doc_text.split()) < 15:
        st.warning("Document is too short for meaningful chunking.")
        st.stop()

    st.divider()

    # Chunking
    t0 = time.time()
    chunks = semantic_chunk(doc_text, embed_fn, threshold=chunk_threshold)
    chunking_time = time.time() - t0

    if not chunks:
        st.error("No chunks produced — check the document text.")
        st.stop()

    # Generation
    bar = st.progress(0, text="Generating candidates…")
    candidate_answers = []
    for idx, chunk in enumerate(chunks):
        bar.progress(idx / len(chunks), text=f"Chunk {idx + 1} / {len(chunks)}")
        try:
            ans = generate_candidate_answer(
                question=question,
                chunk_text=chunk.text,
                backend=backend,
                api_key=gemini_key if backend == "gemini" else None,
                embed_fn=embed_fn,
            )
        except Exception as e:
            ans = f"[error: {e}]"
        candidate_answers.append(ans)
    bar.progress(1.0, text="Done")
    time.sleep(0.2)
    bar.empty()

    # Validation
    result = validate_and_rank(
        chunk_texts=[c.text for c in chunks],
        candidate_answers=candidate_answers,
        embed_fn=embed_fn,
        cross_encoder_predict_fn=cross_fn,
        theta=theta,
        cross_encoder_weight=ce_weight,
    )

    # ── Output tabs ───────────────────────────────────────────────────────────
    tab_result, tab_scores, tab_chunks = st.tabs([
        "Results", "Validation scores", "Chunks",
    ])

    with tab_result:
        top_ca = result["ca_rag_only"][0] if result["ca_rag_only"] else None
        top_eca = result["eca_rag"][0] if result["eca_rag"] else None

        if result["disagree"]:
            st.markdown(
                '<div class="diverge-note">'
                "The two stages select different answers. The bi-encoder ranked a candidate "
                "higher based on embedding similarity; the cross-encoder, attending to both "
                "texts jointly, reversed the order."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="agree-note">Both stages select the same answer.</div>',
                unsafe_allow_html=True,
            )

        col_ca, col_eca = st.columns(2)
        with col_ca:
            ca_score = f"{top_ca.bi_encoder_score:.4f}" if top_ca else "—"
            ca_chunk = f"Chunk {top_ca.chunk_id}" if top_ca else "—"
            ca_answer = top_ca.answer if top_ca else "No candidates passed the threshold."
            st.markdown(
                f"""<div class="method-card">
                    <div class="method-label">Baseline</div>
                    <div class="method-title">CA-RAG &nbsp;·&nbsp; bi-encoder only</div>
                    <div class="method-score">Top chunk: {ca_chunk} &nbsp;·&nbsp; Score: {ca_score}</div>
                    <div class="method-answer">{ca_answer}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_eca:
            eca_score = f"{top_eca.combined_score:.4f}" if top_eca else "—"
            bi_part = f"{top_eca.bi_encoder_score:.3f}" if top_eca else ""
            ce_part = f"{top_eca.cross_encoder_score:.3f}" if top_eca else ""
            eca_chunk = f"Chunk {top_eca.chunk_id}" if top_eca else "—"
            eca_answer = top_eca.answer if top_eca else "No candidates passed the threshold."
            score_detail = f"(bi {bi_part} + cross {ce_part})" if top_eca else ""
            st.markdown(
                f"""<div class="method-card" style="border-left: 3px solid #059669;">
                    <div class="method-label">Proposed</div>
                    <div class="method-title">ECA-RAG &nbsp;·&nbsp; + cross-encoder re-rank</div>
                    <div class="method-score">Top chunk: {eca_chunk} &nbsp;·&nbsp; Score: {eca_score} <span style="color:#94a3b8;font-size:0.8rem;">{score_detail}</span></div>
                    <div class="method-answer">{eca_answer}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with tab_scores:
        rows = []
        for c in result["all_candidates"]:
            rows.append({
                "Chunk": c.chunk_id,
                "Candidate": c.answer[:110] + ("…" if len(c.answer) > 110 else ""),
                "Bi-encoder": round(c.bi_encoder_score, 4),
                "θ filter": "pass" if c.passed_bi_filter else "filtered",
                "Cross-encoder": round(c.cross_encoder_score, 4) if c.cross_encoder_score is not None else None,
                "Combined": round(c.combined_score, 4) if c.combined_score is not None else None,
            })

        df = pd.DataFrame(rows)

        def _style(val):
            if val == "pass":
                return "color:#059669;font-weight:600"
            if val == "filtered":
                return "color:#94a3b8;font-style:italic"
            return ""

        st.dataframe(
            df.style.map(_style, subset=["θ filter"]),
            width="stretch",
            hide_index=True,
        )

    with tab_chunks:
        st.caption(f"{len(chunks)} chunks · {chunking_time:.2f} s")
        for c in chunks:
            st.markdown(f"**Chunk {c.id}** · {len(c.sentences)} sentence{'s' if len(c.sentences) != 1 else ''}")
            st.write(c.text)
            if c.id < len(chunks) - 1:
                st.divider()

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.data import platt_prob, truncate_words, eval_retrieval
from src.arms import (
    df_from,
    iterative_trace,
    sel_gated,
    sel_fixed,
    adaptive_gate_marginal,
    sel_arm6,
    train_sufficiency_stopper,
)
from src.stats import pboot, mcnemar, holm

TYPES = ["bridge", "comparison"]

def build_extension_dataset(start_idx=900, end_idx=3000, seed=42, cache_path="./results_full/examples_extension_900_3000.pkl"):
    if os.path.exists(cache_path):
        print(f"Loading cached extension examples from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Loading HotpotQA distractor validation split and selecting [{start_idx}:{end_idx}]...")
    raw = load_dataset("hotpotqa/hotpot_qa", "distractor")
    split = raw["validation"].shuffle(seed=seed).select(range(start_idx, end_idx))

    def build_example(ex):
        titles = ex["context"]["title"]
        sentences_lists = ex["context"]["sentences"]
        chunks = [{"title": t, "text": f"{t}: " + " ".join(s)} for t, s in zip(titles, sentences_lists)]
        gold_titles = set(ex["supporting_facts"]["title"])
        gold_chunk_ids = [i for i, c in enumerate(chunks) if c["title"] in gold_titles]
        return {
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "chunks": chunks,
            "gold_chunk_ids": gold_chunk_ids,
            "type": ex.get("type"),
        }

    examples = [build_example(ex) for ex in split]
    print(f"Built {len(examples)} extension examples. Saving to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(examples, f)
    return examples

def run_dense_and_ce(examples, embedder, cross_encoder, cache_path="./results_full/arm2_ce_scores_extension_900_3000.pkl", batch_save=100):
    if os.path.exists(cache_path):
        print(f"Loading cached CE scores from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Running dense retrieval and cross-encoder re-ranking for {len(examples)} examples...")
    ce_scores = {}
    arm1_results = []
    total = len(examples)
    t0 = time.time()

    for idx, ex in enumerate(examples):
        # 1. Dense encoding
        q_emb = embedder.encode(ex["question"], normalize_embeddings=True)
        c_embs = embedder.encode([c["text"] for c in ex["chunks"]], normalize_embeddings=True)
        sims = np.dot(c_embs, q_emb)
        dense_ranked = list(np.argsort(-sims))

        # Arm 1 top-4
        arm1_top4 = dense_ranked[:4]
        m1 = eval_retrieval(arm1_top4, ex["gold_chunk_ids"])
        arm1_results.append({"id": ex["id"], "type": ex["type"], "k_used": 4, **m1})

        # Arm 2 candidate pool = top-10 (all chunks)
        pairs = [(ex["question"], ex["chunks"][i]["text"]) for i in range(len(ex["chunks"]))]
        sc = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-sc)
        ce_scores[ex["id"]] = {
            "ranked_ids": [int(o) for o in order],
            "ranked_scores": [float(sc[o]) for o in order],
            "arm1_metrics": m1,
        }

        if (idx + 1) % batch_save == 0 or idx + 1 == total:
            print(f"  Processed {idx + 1}/{total} examples ({time.time() - t0:.1f}s)...")
            with open(cache_path, "wb") as f:
                pickle.dump(ce_scores, f)

    return ce_scores

def build_extension_traces(examples, platt_by_type, cross_encoder, cache_path="./results_full/traces_extension_900_3000.pkl", batch_save_every=50):
    traces = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                traces = pickle.load(f)
            print(f"Loaded {len(traces)} existing traces from {cache_path}")
        except Exception as e:
            print(f"Could not load cache {cache_path}: {e}")
            traces = {}

    total = len(examples)
    to_process = [ex for ex in examples if ex["id"] not in traces]
    if not to_process:
        print(f"All {total} extension traces already cached!")
        return traces

    print(f"Building {len(to_process)}/{total} iterative traces for extension split...")
    t0 = time.time()
    for idx, ex in enumerate(to_process):
        a, b = platt_by_type[ex["type"]]
        traces[ex["id"]] = iterative_trace(ex, a, b, cross_encoder)
        if (idx + 1) % batch_save_every == 0 or (idx + 1) == len(to_process):
            with open(cache_path, "wb") as f:
                pickle.dump(traces, f)
            print(f"  Processed {idx + 1}/{len(to_process)} traces ({len(traces)} total, elapsed: {time.time() - t0:.1f}s)...")

    return traces

def main():
    print("=" * 70)
    print("EXTENSION EVALUATION: HOTPOTQA VALIDATION EXAMPLES 900 TO 3000")
    print("=" * 70)

    # Load frozen calibration config and sufficiency model
    with open("./results_full/calibration_config.pkl", "rb") as f:
        cal_cfg = pickle.load(f)
    platt_by_type = cal_cfg["platt_by_type"]
    theta_star = 0.18
    tau_star = 0.95

    # Train sufficiency stopper on calibration traces to freeze weights
    with open("./results_full/examples.pkl", "rb") as f:
        orig_examples = pickle.load(f)
    cal_ids = set(cal_cfg["cal_ids"])
    ex_cal = [ex for ex in orig_examples if ex["id"] in cal_ids]
    with open("./results_full/traces_cal.pkl", "rb") as f:
        tr_cal = pickle.load(f)
    suff_model = train_sufficiency_stopper(ex_cal, tr_cal)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Initializing models on {device}...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    # 1. Dataset
    ext_examples = build_extension_dataset(start_idx=900, end_idx=3000, seed=42)
    print(f"Extension dataset count: {len(ext_examples)}")
    counts = pd.Series([e["type"] for e in ext_examples]).value_counts()
    print("Question type distribution:")
    print(counts)

    # 2. Dense & CE Scores
    ce_scores = run_dense_and_ce(ext_examples, embedder, ce)

    # 3. Traces
    traces = build_extension_traces(ext_examples, platt_by_type, ce)

    # 4. Evaluate Arms
    print("\nEvaluating all arms on extension dataset...")
    arms = {}

    # Arm 1
    arms["Arm1 dense k=4"] = pd.DataFrame([
        {"id": ex["id"], "type": ex["type"], "k_used": 4, **ce_scores[ex["id"]]["arm1_metrics"]}
        for ex in ext_examples
    ])

    # Arm 2
    arms["Arm2 CE k=4"] = df_from(ext_examples, lambda ex: ce_scores[ex["id"]]["ranked_ids"][:4])
    arms["Arm2 CE k=5"] = df_from(ext_examples, lambda ex: ce_scores[ex["id"]]["ranked_ids"][:5])

    # Iterative without gating
    arms["Iter k=4 (no gate)"] = df_from(ext_examples, lambda ex: sel_fixed(traces[ex["id"]], 4))
    arms["Iter k=5 (no gate)"] = df_from(ext_examples, lambda ex: sel_fixed(traces[ex["id"]], 5))

    # Arm 4b per-type
    def eval_arm4b(th):
        rows = []
        for ex in ext_examples:
            a, b = platt_by_type[ex["type"]]
            sc = ce_scores[ex["id"]]["ranked_scores"]
            r_ids = ce_scores[ex["id"]]["ranked_ids"]
            probs = platt_prob(sc, a, b)
            sel, k_used = adaptive_gate_marginal(r_ids, probs, th, min_k=1, max_k=6)
            rows.append({"id": ex["id"], "type": ex["type"], "k_used": k_used, **eval_retrieval(sel, ex["gold_chunk_ids"])})
        return pd.DataFrame(rows)

    arms["Arm4b per-type (theta=0.18)"] = eval_arm4b(0.18)
    arms["Arm4b per-type (theta=0.15, sensitivity)"] = eval_arm4b(0.15)

    # Arm 5 iter+gate
    arms["Arm5 iter+gate (theta=0.18)"] = df_from(ext_examples, lambda ex: sel_gated(traces[ex["id"]], theta=0.18, min_k=1))
    arms["Arm5 iter+gate (theta=0.15, sensitivity)"] = df_from(ext_examples, lambda ex: sel_gated(traces[ex["id"]], theta=0.15, min_k=1))

    # Hybrid (Iter k=4 on bridge, Arm 2 on comparison) [Oracle]
    iter_k4_df = arms["Iter k=4 (no gate)"]
    arm2_k4_df = arms["Arm2 CE k=4"]
    arms["Hybrid: Iter k=4 bridge / Arm2 comp. (oracle)"] = pd.concat([
        iter_k4_df[iter_k4_df["type"] == "bridge"],
        arm2_k4_df[arm2_k4_df["type"] == "comparison"],
    ], ignore_index=True)

    # Arm 6
    arms["Arm6 sufficiency stop (tau=0.95, theta=0.18)"] = df_from(
        ext_examples,
        lambda ex: sel_arm6(traces[ex["id"]], suff_model, tau_star, int(ex["type"] == "bridge"), theta=0.18)
    )

    # Summary Table
    t_rows = []
    for name, df in arms.items():
        for t in TYPES:
            s = df[df["type"] == t]
            t_rows.append({
                "arm": name,
                "type": t,
                "n": len(s),
                "recall": s["recall"].mean(),
                "precision": s["precision"].mean(),
                "full_match": s["full_match"].mean(),
                "k": s["k_used"].mean(),
            })
    table_ext = pd.DataFrame(t_rows)
    table_path = "./results_full/TABLE_extension_900_3000.csv"
    table_ext.to_csv(table_path, index=False)
    print("\n" + "=" * 70)
    print("EXTENSION RESULTS (900-3000, N=2100) SAVED TO TABLE_extension_900_3000.csv:")
    print("=" * 70)
    print(table_ext.round(4).to_string(index=False))

    # Significance tests on extension split
    PAIRS_EXT = [
        ("Arm5 iter+gate (theta=0.18)", "Arm2 CE k=4", "Arm 5 vs Arm 2 k=4"),
        ("Iter k=4 (no gate)", "Arm2 CE k=4", "Iter k=4 vs Arm 2 k=4"),
        ("Iter k=5 (no gate)", "Arm2 CE k=5", "Iter k=5 vs Arm 2 k=5"),
        ("Arm5 iter+gate (theta=0.18)", "Iter k=4 (no gate)", "Arm 5 vs Iter k=4 (gating effect)"),
        ("Arm4b per-type (theta=0.18)", "Arm2 CE k=4", "Arm 4b vs Arm 2 (calib single-pass vs fixed-k)"),
        ("Arm5 iter+gate (theta=0.18)", "Arm4b per-type (theta=0.18)", "Arm 5 vs Arm 4b (iter vs single-pass gating)"),
        ("Arm2 CE k=4", "Arm1 dense k=4", "Arm 2 vs Arm 1 (re-ranking effect)"),
    ]
    srows = []
    for A, B, why in PAIRS_EXT:
        for t in TYPES:
            a_df = arms[A][arms[A]["type"] == t]
            b_df = arms[B][arms[B]["type"] == t]
            d, lo, hi, p_boot = pboot(a_df, b_df, metric="recall", n_boot=10000, seed=42)
            p_mcnemar = mcnemar(a_df, b_df)
            srows.append({
                "A": A,
                "B": B,
                "type": t,
                "comparator": why,
                "n": len(a_df),
                "delta_recall": d,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_recall": p_boot,
                "p_fullmatch_mcnemar": p_mcnemar,
                "delta_k": a_df["k_used"].mean() - b_df["k_used"].mean(),
            })
    sig_ext = pd.DataFrame(srows)
    sig_ext["p_recall_holm"] = holm(sig_ext["p_recall"].values)
    sig_ext["p_fullmatch_holm"] = holm(sig_ext["p_fullmatch_mcnemar"].values)
    sig_ext_path = "./results_full/TABLE2_extension_significance.csv"
    sig_ext.to_csv(sig_ext_path, index=False)
    print("\nSaved TABLE2_extension_significance.csv")

if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data import eval_retrieval, platt_prob, truncate_words

def df_from(exs, selector):
    rows = []
    for ex in exs:
        sel = selector(ex)
        rows.append({
            "id": ex["id"],
            "type": ex["type"],
            "k_used": len(sel),
            **eval_retrieval(sel, ex["gold_chunk_ids"]),
        })
    return pd.DataFrame(rows)

def iterative_trace(ex, a, b, cross_encoder, max_hops=6, aug_words=None):
    """Run the iterative loop with NO stopping; record every hop."""
    chunks, q = ex["chunks"], ex["question"]
    remaining, trace, cur = list(range(len(chunks))), [], q
    for hop in range(max_hops):
        if not remaining:
            break
        pairs = [(cur, chunks[i]["text"]) for i in remaining]
        sc = cross_encoder.predict(pairs, show_progress_bar=False)
        j = int(np.argmax(sc))
        s = float(sc[j])
        idx = remaining[j]
        p = float(platt_prob(np.array([s]), a, b)[0])
        trace.append({"hop": hop, "chunk": idx, "score": s, "prob": p})
        remaining.remove(idx)
        addition = truncate_words(chunks[idx]["text"], aug_words)
        cur = f"{q} {addition}"
    return trace

def build_traces(exs, platt_by_type, cross_encoder, aug_words=None, cache_path=None, batch_save_every=50):
    traces = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                traces = pickle.load(f)
            print(f"Loaded {len(traces)} existing traces from {cache_path}")
        except Exception as e:
            print(f"Could not load cache {cache_path}: {e}")
            traces = {}

    total = len(exs)
    to_process = [ex for ex in exs if ex["id"] not in traces]
    if not to_process:
        print(f"All {total} traces already present in cache {cache_path}!")
        return traces

    print(f"Building {len(to_process)}/{total} traces (aug_words={aug_words})...")
    for idx, ex in enumerate(to_process):
        a, b = platt_by_type[ex["type"]]
        traces[ex["id"]] = iterative_trace(ex, a, b, cross_encoder, aug_words=aug_words)
        if (idx + 1) % batch_save_every == 0 or (idx + 1) == len(to_process):
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(traces, f)
            print(f"  Processed {idx + 1}/{len(to_process)} traces (total cached: {len(traces)})...")
    return traces

def sel_gated(trace, theta, min_k=1):
    out = []
    for t in trace:
        if t["hop"] >= min_k and t["prob"] < theta:
            break
        out.append(t["chunk"])
    return out

def sel_fixed(trace, k):
    return [t["chunk"] for t in trace[:k]]

def adaptive_gate_marginal(ranked_ids, ranked_scores, threshold, min_k=1, max_k=6):
    n = min(len(ranked_ids), max_k)
    selected = []
    for i in range(n):
        if i < min_k or ranked_scores[i] >= threshold:
            selected.append(ranked_ids[i])
        else:
            break
    res = selected or ranked_ids[:min_k]
    return res, len(res)

def fixed_ce(exs, arm2_ce_scores, k):
    return df_from(exs, lambda ex: arm2_ce_scores[ex["id"]]["ranked_ids"][:k])

def run_single_pass_arm3(exs, arm2_ce_scores, raw_threshold, min_k=1, max_k=6):
    rows = []
    for ex in exs:
        ce = arm2_ce_scores[ex["id"]]
        sel, k_used = adaptive_gate_marginal(ce["ranked_ids"], ce["ranked_scores"], raw_threshold, min_k=min_k, max_k=max_k)
        rows.append({
            "id": ex["id"],
            "type": ex["type"],
            "k_used": k_used,
            **eval_retrieval(sel, ex["gold_chunk_ids"]),
        })
    return pd.DataFrame(rows)

def run_single_pass_arm4(exs, arm2_ce_scores, params_lookup, threshold, per_type=True, min_k=1, max_k=6):
    rows = []
    for ex in exs:
        ce = arm2_ce_scores[ex["id"]]
        a, b = params_lookup[ex["type"]] if per_type else params_lookup
        probs = platt_prob(np.array(ce["ranked_scores"]), a, b)
        sel, k_used = adaptive_gate_marginal(ce["ranked_ids"], probs, threshold, min_k=min_k, max_k=max_k)
        rows.append({
            "id": ex["id"],
            "type": ex["type"],
            "k_used": k_used,
            **eval_retrieval(sel, ex["gold_chunk_ids"]),
        })
    return pd.DataFrame(rows)

def feats(trace, h, is_bridge):
    cand, prev = trace[h], trace[h - 1]
    return [
        cand["prob"],
        cand["score"],
        h,
        prev["prob"] - cand["prob"],
        min(t["prob"] for t in trace[:h]),
        is_bridge,
    ]

def train_sufficiency_stopper(ex_cal, tr_cal):
    X, y = [], []
    for ex in ex_cal:
        tr = tr_cal[ex["id"]]
        gold = set(ex["gold_chunk_ids"])
        isb = int(ex["type"] == "bridge")
        for h in range(1, len(tr)):
            X.append(feats(tr, h, isb))
            y.append(int(gold <= {t["chunk"] for t in tr[:h]}))
    suff = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    print(f"Sufficiency stopper fit on {len(y)} cal rows, positive rate {np.mean(y):.3f}")
    return suff

def sel_arm6(trace, suff_model, tau, is_bridge, theta):
    out = []
    for h, t in enumerate(trace):
        if h >= 1:
            if t["prob"] < theta:
                break
            f = feats(trace, h, is_bridge)
            if suff_model.predict_proba([f])[0, 1] >= tau:
                break
        out.append(t["chunk"])
    return out

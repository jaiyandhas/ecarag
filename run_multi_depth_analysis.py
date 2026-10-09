import pickle
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import platt_prob, eval_retrieval
from src.arms import sel_gated, adaptive_gate_marginal, feats, sel_arm6, train_sufficiency_stopper
from src.stats import holm

# 1. Load data
print("Loading data...")
with open('results_full/calibration_config.pkl', 'rb') as f:
    cal_cfg = pickle.load(f)
with open('results_full/examples.pkl', 'rb') as f:
    ex_900 = pickle.load(f)
with open('results_full/examples_extension_900_3000.pkl', 'rb') as f:
    ex_ext = pickle.load(f)

with open('results_full/traces_cal.pkl', 'rb') as f:
    tr_cal = pickle.load(f)
with open('results_full/traces_eval.pkl', 'rb') as f:
    tr_eval = pickle.load(f)
with open('results_full/traces_extension_900_3000.pkl', 'rb') as f:
    tr_ext = pickle.load(f)

with open('results_full/arm2_ce_scores.pkl', 'rb') as f:
    ce_eval = pickle.load(f)
with open('results_full/arm2_ce_scores_extension_900_3000.pkl', 'rb') as f:
    ce_ext = pickle.load(f)

platt_by_type = cal_cfg['platt_by_type']
cal_ids = set(cal_cfg['cal_ids'])
eval_ids = set(cal_cfg['eval_ids'])
ex_cal = [ex for ex in ex_900 if ex['id'] in cal_ids]
ex_eval = [ex for ex in ex_900 if ex['id'] in eval_ids]

# Sufficiency model trained on calibration
suff_model = train_sufficiency_stopper(ex_cal, tr_cal)

# Pooled N=2640
exs_pooled = ex_eval + ex_ext
traces_pooled = {**tr_eval, **tr_ext}
ce_pooled = {**ce_eval, **ce_ext}

print(f"Total pooled examples: {len(exs_pooled)}")

# Sweeps
thetas_sweep = [0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
taus_sweep = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999]
target_depths = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

def precompute_arrays(examples):
    n = len(examples)
    # Fixed-k (k=1..6)
    r_fixed = np.zeros((n, 6))
    for i, ex in enumerate(examples):
        r_ids = ce_pooled[ex['id']]['ranked_ids']
        g = ex['gold_chunk_ids']
        for k in range(1, 7):
            r_fixed[i, k - 1] = eval_retrieval(r_ids[:k], g)['recall']
    
    # Arm 4b
    k_4b = np.zeros((n, len(thetas_sweep)))
    r_4b = np.zeros((n, len(thetas_sweep)))
    for i, ex in enumerate(examples):
        a, b = platt_by_type[ex['type']]
        sc = ce_pooled[ex['id']]['ranked_scores']
        r_ids = ce_pooled[ex['id']]['ranked_ids']
        probs = platt_prob(sc, a, b)
        g = ex['gold_chunk_ids']
        for j, th in enumerate(thetas_sweep):
            sel, k_used = adaptive_gate_marginal(r_ids, probs, th, min_k=1, max_k=6)
            k_4b[i, j] = k_used
            r_4b[i, j] = eval_retrieval(sel, g)['recall']
            
    # Arm 5
    k_5 = np.zeros((n, len(thetas_sweep)))
    r_5 = np.zeros((n, len(thetas_sweep)))
    for i, ex in enumerate(examples):
        tr = traces_pooled[ex['id']]
        g = ex['gold_chunk_ids']
        for j, th in enumerate(thetas_sweep):
            sel = sel_gated(tr, theta=th, min_k=1)
            k_5[i, j] = len(sel)
            r_5[i, j] = eval_retrieval(sel, g)['recall']
            
    # Arm 6
    k_6 = np.zeros((n, len(taus_sweep)))
    r_6 = np.zeros((n, len(taus_sweep)))
    for i, ex in enumerate(examples):
        tr = traces_pooled[ex['id']]
        g = ex['gold_chunk_ids']
        isb = int(ex['type'] == 'bridge')
        for j, tau in enumerate(taus_sweep):
            sel = sel_arm6(tr, suff_model, tau, isb, theta=0.18)
            k_6[i, j] = len(sel)
            r_6[i, j] = eval_retrieval(sel, g)['recall']
            
    # Iterative ungated points at k=4 and k=5
    r_iter_k4 = np.zeros(n)
    r_iter_k5 = np.zeros(n)
    for i, ex in enumerate(examples):
        tr = traces_pooled[ex['id']]
        g = ex['gold_chunk_ids']
        r_iter_k4[i] = eval_retrieval([t['chunk'] for t in tr[:4]], g)['recall']
        r_iter_k5[i] = eval_retrieval([t['chunk'] for t in tr[:5]], g)['recall']
        
    return {
        'r_fixed': r_fixed,
        'k_fixed': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        'k_4b': k_4b,
        'r_4b': r_4b,
        'k_5': k_5,
        'r_5': r_5,
        'k_6': k_6,
        'r_6': r_6,
        'r_iter_k4': r_iter_k4,
        'r_iter_k5': r_iter_k5,
    }

print("Precomputing arrays for Bridge...")
bridge_exs = [e for e in exs_pooled if e['type'] == 'bridge']
arr_bridge = precompute_arrays(bridge_exs)

print("Precomputing arrays for Comparison...")
comp_exs = [e for e in exs_pooled if e['type'] == 'comparison']
arr_comp = precompute_arrays(comp_exs)

# Save precomputed arrays to scratch
with open('results_full/precomputed_multi_depth_arrays.pkl', 'wb') as f:
    pickle.dump({'bridge': arr_bridge, 'comp': arr_comp}, f)
print("Arrays precomputed and saved.")

#!/usr/bin/env python3
"""
Computes calibration metrics on the N=2100 extension split (20,901 chunks)
with frozen calibration parameters from calibration_config.pkl.
Writes results_full/TABLE_calibration_extension.csv.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pickle
import numpy as np
import pandas as pd
from src.stats import ece, brier
from src.data import platt_prob

RESULTS_DIR = 'results_full'

def main():
    with open(os.path.join(RESULTS_DIR, 'calibration_config.pkl'), 'rb') as f:
        cal_cfg = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'examples_extension_900_3000.pkl'), 'rb') as f:
        ex_ext = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'arm2_ce_scores_extension_900_3000.pkl'), 'rb') as f:
        ce_ext = pickle.load(f)

    a_pool = cal_cfg['a_pooled']
    b_pool = cal_cfg['b_pooled']
    platt_type = cal_cfg['platt_by_type']

    all_scores = []
    all_labels = []
    all_types = []

    for ex in ex_ext:
        sc_dict = ce_ext[ex['id']]
        r_ids = sc_dict['ranked_ids']
        r_scores = sc_dict['ranked_scores']
        g_ids = set(ex['gold_chunk_ids'])
        for cid, s in zip(r_ids, r_scores):
            all_scores.append(s)
            all_labels.append(int(cid in g_ids))
            all_types.append(ex['type'])

    all_scores = np.array(all_scores)
    yv = np.array(all_labels)
    n_chunks = len(all_scores)

    raw_p = 1.0 / (1.0 + np.exp(-all_scores))
    pooled_p = platt_prob(all_scores, a_pool, b_pool)
    pertype_p = np.array([platt_prob(np.array([s]), *platt_type[t])[0]
                          for s, t in zip(all_scores, all_types)])

    df = pd.DataFrame({
        'method': ['raw sigmoid(s)', 'Platt pooled', 'Platt per-type'],
        'chunks': [n_chunks, n_chunks, n_chunks],
        'ECE': [ece(raw_p, yv), ece(pooled_p, yv), ece(pertype_p, yv)],
        'Brier': [brier(raw_p, yv), brier(pooled_p, yv), brier(pertype_p, yv)],
    })

    out_path = os.path.join(RESULTS_DIR, 'TABLE_calibration_extension.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

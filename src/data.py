import os
import json
import pickle
import numpy as np
import pandas as pd

def eval_retrieval(retrieved_ids, gold_ids):
    retrieved_set, gold_set = set(retrieved_ids), set(gold_ids)
    if not gold_set:
        return {"recall": None, "precision": None, "full_match": None}
    hit = len(retrieved_set & gold_set)
    return {
        "recall": hit / len(gold_set),
        "precision": hit / len(retrieved_set) if retrieved_set else 0.0,
        "full_match": int(gold_set.issubset(retrieved_set)),
    }

def platt_prob(scores, a, b):
    scores = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-(a * scores + b)))

def truncate_words(text, n):
    if n is None:
        return text
    return " ".join(text.split()[:n])

def load_data(data_dir="./results_full"):
    with open(os.path.join(data_dir, "examples.pkl"), "rb") as f:
        examples = pickle.load(f)
    with open(os.path.join(data_dir, "calibration_config.pkl"), "rb") as f:
        cal_cfg = pickle.load(f)
    with open(os.path.join(data_dir, "arm2_ce_scores.pkl"), "rb") as f:
        arm2_ce_scores = pickle.load(f)
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = json.load(f)

    arm1_df = pd.read_csv(os.path.join(data_dir, "arm1_results.csv"))
    arm2_df = pd.read_csv(os.path.join(data_dir, "arm2_results.csv"))
    chunk_df = pd.read_csv(os.path.join(data_dir, "chunk_level_scores.csv"))

    eval_ids = set(cal_cfg["eval_ids"])
    cal_ids = set(cal_cfg["cal_ids"])
    ex_eval = [ex for ex in examples if ex["id"] in eval_ids]
    ex_cal = [ex for ex in examples if ex["id"] in cal_ids]

    return {
        "examples": examples,
        "ex_eval": ex_eval,
        "ex_cal": ex_cal,
        "eval_ids": eval_ids,
        "cal_ids": cal_ids,
        "cal_cfg": cal_cfg,
        "arm2_ce_scores": arm2_ce_scores,
        "config": config,
        "arm1_df": arm1_df,
        "arm2_df": arm2_df,
        "chunk_df": chunk_df,
    }

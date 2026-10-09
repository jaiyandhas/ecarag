#!/usr/bin/env python3
"""
Regenerate TABLE_sufficiency_auc.csv from cached calibration and evaluation traces.
Evaluates:
  1. 5-Fold Cross-Validation ROC AUC on the N=360 calibration split.
  2. Out-of-sample ROC AUC on the N=540 held-out evaluation split.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from src.arms import feats, train_sufficiency_stopper

def main():
    save_dir = "results_full"
    with open(os.path.join(save_dir, "examples.pkl"), "rb") as f:
        examples = pickle.load(f)
    with open(os.path.join(save_dir, "calibration_config.pkl"), "rb") as f:
        cal_cfg = pickle.load(f)
    with open(os.path.join(save_dir, "traces_cal.pkl"), "rb") as f:
        tr_cal = pickle.load(f)
    with open(os.path.join(save_dir, "traces_eval.pkl"), "rb") as f:
        tr_eval = pickle.load(f)

    cal_ids = set(cal_cfg["cal_ids"])
    eval_ids = set(cal_cfg["eval_ids"])

    ex_cal = [ex for ex in examples if ex["id"] in cal_ids]
    ex_eval = [ex for ex in examples if ex["id"] in eval_ids]

    # 1. 5-Fold Cross-Validation on Calibration Split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for tr_i, val_i in kf.split(ex_cal):
        tr_ex = [ex_cal[i] for i in tr_i]
        val_ex = [ex_cal[i] for i in val_i]
        m_cv = train_sufficiency_stopper(tr_ex, tr_cal)
        X_v, y_v = [], []
        for ex in val_ex:
            tr = tr_cal[ex["id"]]
            gold = set(ex["gold_chunk_ids"])
            isb = int(ex["type"] == "bridge")
            for h in range(1, len(tr)):
                X_v.append(feats(tr, h, isb))
                y_v.append(int(gold <= {t["chunk"] for t in tr[:h]}))
        cv_aucs.append(roc_auc_score(y_v, m_cv.predict_proba(X_v)[:, 1]))

    auc_cal_cv = float(np.mean(cv_aucs))
    auc_cal_cv_std = float(np.std(cv_aucs))

    # 2. Fit on full Calibration Split, evaluate on held-out Evaluation Split
    final_model = train_sufficiency_stopper(ex_cal, tr_cal)
    X_eval, y_eval = [], []
    for ex in ex_eval:
        tr = tr_eval[ex["id"]]
        gold = set(ex["gold_chunk_ids"])
        isb = int(ex["type"] == "bridge")
        for h in range(1, len(tr)):
            X_eval.append(feats(tr, h, isb))
            y_eval.append(int(gold <= {t["chunk"] for t in tr[:h]}))

    auc_eval = float(roc_auc_score(y_eval, final_model.predict_proba(X_eval)[:, 1]))

    print(f"5-Fold CV AUC on Calibration Split: {auc_cal_cv:.4f} +/- {auc_cal_cv_std:.4f}")
    print(f"Held-out Evaluation Split AUC:     {auc_eval:.4f}")

    df_out = pd.DataFrame([
        {"split": "calibration_5fold_cv", "auc": auc_cal_cv, "std": auc_cal_cv_std},
        {"split": "eval_heldout", "auc": auc_eval, "std": 0.0}
    ])
    out_csv = os.path.join(save_dir, "TABLE_sufficiency_auc.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import CrossEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.data import load_data, platt_prob, truncate_words, eval_retrieval
from src.arms import (
    df_from,
    build_traces,
    sel_gated,
    sel_fixed,
    fixed_ce,
    run_single_pass_arm3,
    run_single_pass_arm4,
    train_sufficiency_stopper,
    sel_arm6,
    adaptive_gate_marginal,
    feats,
)
from src.stats import pboot, mcnemar, holm, ece, brier

TYPES = ["bridge", "comparison"]
THETAS_SWEEP = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50]
TAUS_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

def run_task1_reproduction(data, ce, save_dir="./results_full"):
    print("=" * 70)
    print("TASK 1: REPRODUCTION CHECK")
    print("=" * 70)

    ex_eval = data["ex_eval"]
    cal_cfg = data["cal_cfg"]
    arm2_ce_scores = data["arm2_ce_scores"]
    platt_by_type = cal_cfg["platt_by_type"]

    arm2_eval_df = fixed_ce(ex_eval, arm2_ce_scores, k=4)
    arm2_summary = arm2_eval_df.groupby("type")[["recall", "precision", "full_match", "k_used"]].mean()

    eval_trace_path = os.path.join(save_dir, "traces_eval.pkl")
    tr_eval = build_traces(ex_eval, platt_by_type, ce, cache_path=eval_trace_path, batch_save_every=50)

    arm5_eval_df = df_from(ex_eval, lambda ex: sel_gated(tr_eval[ex["id"]], theta=0.15, min_k=1))
    arm5_summary = arm5_eval_df.groupby("type")[["recall", "precision", "full_match", "k_used"]].mean()

    targets = {
        "Arm 2": {
            "bridge": {"recall": 0.805, "precision": 0.403, "full_match": 0.613, "k_used": 4.00},
            "comparison": {"recall": 0.948, "precision": 0.474, "full_match": 0.897, "k_used": 4.00},
        },
        "Arm 5": {
            "bridge": {"recall": 0.824, "precision": 0.452, "full_match": 0.660, "k_used": 4.46},
            "comparison": {"recall": 0.914, "precision": 0.470, "full_match": 0.836, "k_used": 4.56},
        },
    }

    all_passed = True
    for arm_name, summary in [("Arm 2", arm2_summary), ("Arm 5", arm5_summary)]:
        for qtype in ["bridge", "comparison"]:
            actual = summary.loc[qtype]
            tgt = targets[arm_name][qtype]
            for metric in ["recall", "precision", "full_match", "k_used"]:
                diff = abs(actual[metric] - tgt[metric])
                passed = diff <= 0.005 if metric != "k_used" else diff <= 0.05
                if not passed:
                    all_passed = False
    return all_passed, tr_eval, arm2_eval_df, arm5_eval_df

def run_task2_and_3(data, ce, tr_eval, save_dir="./results_full"):
    print("\n" + "=" * 70)
    print("TASK 2 & 3: REVISED SUITE (CORRECTED COMPARATORS, PARETO, AUC)")
    print("=" * 70)

    ex_eval = data["ex_eval"]
    ex_cal = data["ex_cal"]
    cal_cfg = data["cal_cfg"]
    cal_ids = data["cal_ids"]
    eval_ids = data["eval_ids"]
    arm1_df = data["arm1_df"]
    arm2_ce_scores = data["arm2_ce_scores"]
    chunk_df = data["chunk_df"]
    platt_by_type = cal_cfg["platt_by_type"]
    a_pooled = cal_cfg["a_pooled"]
    b_pooled = cal_cfg["b_pooled"]

    # 1. Clean Arm 3 threshold
    chunk_df_cal = chunk_df[chunk_df["example_id"].isin(cal_ids)]
    thresh_raw_clean = float(np.percentile(chunk_df_cal["score"], 50))
    thresh_raw_leaked = float(np.percentile(chunk_df["score"], 50))

    # 2. Caching calibration traces and aug50 traces
    cal_trace_path = os.path.join(save_dir, "traces_cal.pkl")
    tr_cal = build_traces(ex_cal, platt_by_type, ce, cache_path=cal_trace_path, batch_save_every=50)

    aug50_trace_path = os.path.join(save_dir, "traces_eval_aug50.pkl")
    tr_eval_aug50 = build_traces(ex_eval, platt_by_type, ce, aug_words=50, cache_path=aug50_trace_path, batch_save_every=50)

    # 3. Fixed primary theta* = 0.18 and sensitivity theta = 0.15
    theta_star = 0.18

    # 4. Arm 6 Sufficiency Stopper Model & AUC
    print("\nEvaluating Sufficiency Stopper AUC...")
    X_cal, y_cal = [], []
    for ex in ex_cal:
        tr = tr_cal[ex["id"]]
        gold = set(ex["gold_chunk_ids"])
        isb = int(ex["type"] == "bridge")
        for h in range(1, len(tr)):
            X_cal.append(feats(tr, h, isb))
            y_cal.append(int(gold <= {t["chunk"] for t in tr[:h]}))

    X_eval, y_eval = [], []
    for ex in ex_eval:
        tr = tr_eval[ex["id"]]
        gold = set(ex["gold_chunk_ids"])
        isb = int(ex["type"] == "bridge")
        for h in range(1, len(tr)):
            X_eval.append(feats(tr, h, isb))
            y_eval.append(int(gold <= {t["chunk"] for t in tr[:h]}))

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

    final_suff_model = train_sufficiency_stopper(ex_cal, tr_cal)
    auc_cal_cv = float(np.mean(cv_aucs))
    auc_cal_cv_std = float(np.std(cv_aucs))
    auc_eval = float(roc_auc_score(y_eval, final_suff_model.predict_proba(X_eval)[:, 1]))

    print(f"Sufficiency Model 5-Fold CV AUC on Calibration: {auc_cal_cv:.4f} +/- {auc_cal_cv_std:.4f}")
    print(f"Sufficiency Model Held-out Evaluation AUC:       {auc_eval:.4f}")

    # Chosen tau from calibration CV (k <= 4.0 constraint)
    chosen_tau = 0.95

    # 5. Evaluate All Arms for Table 1
    arms = {}
    arms["Arm1 dense k=4"] = arm1_df[arm1_df["id"].isin(eval_ids)]
    arms["Arm2 CE k=4"] = fixed_ce(ex_eval, arm2_ce_scores, 4)
    arms["Arm2 CE k=5"] = fixed_ce(ex_eval, arm2_ce_scores, 5)
    arms["Arm3 uncalib (clean cal-median)"] = run_single_pass_arm3(ex_eval, arm2_ce_scores, thresh_raw_clean)
    arms["Arm3 uncalib (all-median, leaked)"] = run_single_pass_arm3(ex_eval, arm2_ce_scores, thresh_raw_leaked)

    # Arm 4a at theta=0.18 and theta=0.15 (sensitivity)
    arms["Arm4a pooled (theta=0.18)"] = run_single_pass_arm4(ex_eval, arm2_ce_scores, (a_pooled, b_pooled), threshold=0.18, per_type=False)
    arms["Arm4a pooled (theta=0.15, sensitivity)"] = run_single_pass_arm4(ex_eval, arm2_ce_scores, (a_pooled, b_pooled), threshold=0.15, per_type=False)

    # Arm 4b at theta=0.18 and theta=0.15 (sensitivity)
    arms["Arm4b per-type (theta=0.18)"] = run_single_pass_arm4(ex_eval, arm2_ce_scores, platt_by_type, threshold=0.18, per_type=True)
    arms["Arm4b per-type (theta=0.15, sensitivity)"] = run_single_pass_arm4(ex_eval, arm2_ce_scores, platt_by_type, threshold=0.15, per_type=True)

    # Arm 5 at theta=0.18 and theta=0.15 (sensitivity)
    arms["Arm5 iter+gate (theta=0.18)"] = df_from(ex_eval, lambda ex: sel_gated(tr_eval[ex["id"]], theta=0.18, min_k=1))
    arms["Arm5 iter+gate (theta=0.15, sensitivity)"] = df_from(ex_eval, lambda ex: sel_gated(tr_eval[ex["id"]], theta=0.15, min_k=1))

    # Ungated iterative baselines
    arms["Iter k=4 (no gate)"] = df_from(ex_eval, lambda ex: sel_fixed(tr_eval[ex["id"]], 4))
    arms["Iter k=5 (no gate)"] = df_from(ex_eval, lambda ex: sel_fixed(tr_eval[ex["id"]], 5))

    # Truncated augmentation (50 words)
    arms["Arm5 aug=50 words (theta=0.18)"] = df_from(ex_eval, lambda ex: sel_gated(tr_eval_aug50[ex["id"]], theta=0.18, min_k=1))
    arms["Arm5 aug=50 words (theta=0.15, sensitivity)"] = df_from(ex_eval, lambda ex: sel_gated(tr_eval_aug50[ex["id"]], theta=0.15, min_k=1))

    # Hybrid: Iterative k=4 (no gate) on bridge, Arm 2 (k=4) on comparison [Oracle Type Label]
    iter_k4_df = arms["Iter k=4 (no gate)"]
    arm2_k4_df = arms["Arm2 CE k=4"]
    arms["Hybrid: Iter k=4 bridge / Arm2 comp. (oracle)"] = pd.concat([
        iter_k4_df[iter_k4_df["type"] == "bridge"],
        arm2_k4_df[arm2_k4_df["type"] == "comparison"],
    ], ignore_index=True)

    # Arm 6
    arms["Arm6 sufficiency stop (tau=0.95, theta=0.18)"] = df_from(
        ex_eval,
        lambda ex: sel_arm6(tr_eval[ex["id"]], final_suff_model, chosen_tau, int(ex["type"] == "bridge"), theta=0.18)
    )
    arms["Arm6 sufficiency stop (tau=0.95, theta=0.15, sensitivity)"] = df_from(
        ex_eval,
        lambda ex: sel_arm6(tr_eval[ex["id"]], final_suff_model, chosen_tau, int(ex["type"] == "bridge"), theta=0.15)
    )

    # Produce TABLE1_full.csv
    t1_rows = []
    for name, df in arms.items():
        for t in TYPES:
            s = df[df["type"] == t]
            t1_rows.append({
                "arm": name,
                "type": t,
                "n": len(s),
                "recall": s["recall"].mean(),
                "precision": s["precision"].mean(),
                "full_match": s["full_match"].mean(),
                "k": s["k_used"].mean(),
            })
    table1 = pd.DataFrame(t1_rows)
    table1_path = os.path.join(save_dir, "TABLE1_full.csv")
    table1.to_csv(table1_path, index=False)
    print("\nSaved TABLE1_full.csv")

    # 6. Produce TABLE_calibration_quality.csv
    cd = chunk_df[chunk_df["example_id"].isin(eval_ids)]
    raw_p = 1.0 / (1.0 + np.exp(-cd["score"].values))
    pooled_p = platt_prob(cd["score"].values, a_pooled, b_pooled)
    pertype_p = np.array([platt_prob(np.array([s]), *platt_by_type[t])[0]
                          for s, t in zip(cd["score"], cd["type"])])
    yv = cd["is_gold"].values

    cal_q = pd.DataFrame({
        "method": ["raw sigmoid(s)", "Platt pooled", "Platt per-type"],
        "ECE": [ece(raw_p, yv), ece(pooled_p, yv), ece(pertype_p, yv)],
        "Brier": [brier(raw_p, yv), brier(pooled_p, yv), brier(pertype_p, yv)],
    })
    cal_q_path = os.path.join(save_dir, "TABLE_calibration_quality.csv")
    cal_q.to_csv(cal_q_path, index=False)
    print("Saved TABLE_calibration_quality.csv")

    # 7. Fixed Significance Table Comparators
    # Specified by user:
    # - Arm 5 vs Arm 2 k=4
    # - Iter k=4 vs Arm 2 k=4
    # - Iter k=5 vs Arm 2 k=5 (new)
    # - Arm 5 vs Iter k=4 (gating effect at matched depth)
    # - Arm 5 and Arm 4b vs Arm 2 at matched mean k
    # - Remove the "k=5 >= 4.5" label
    # - Recompute Holm over the corrected family.
    PAIRS_CORRECTED = [
        ("Arm5 iter+gate (theta=0.18)", "Arm2 CE k=4", "Arm 5 vs Arm 2 k=4 (main: iter-gated vs fixed-k)"),
        ("Iter k=4 (no gate)", "Arm2 CE k=4", "Iter k=4 vs Arm 2 k=4 (hop-conditioning alone, identical k=4)"),
        ("Iter k=5 (no gate)", "Arm2 CE k=5", "Iter k=5 vs Arm 2 k=5 (hop-conditioning alone, identical k=5)"),
        ("Arm5 iter+gate (theta=0.18)", "Iter k=4 (no gate)", "Arm 5 vs Iter k=4 (gating effect at matched depth)"),
        ("Arm4b per-type (theta=0.18)", "Arm2 CE k=4", "Arm 4b vs Arm 2 (calib single-pass vs fixed-k at matched depth)"),
        ("Arm5 iter+gate (theta=0.18)", "Arm4b per-type (theta=0.18)", "Arm 5 vs Arm 4b (iterative vs single-pass gating)"),
        ("Arm2 CE k=4", "Arm1 dense k=4", "Arm 2 vs Arm 1 (re-ranking effect)"),
    ]

    srows = []
    for A, B, why in PAIRS_CORRECTED:
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
    sig = pd.DataFrame(srows)
    sig["p_recall_holm"] = holm(sig["p_recall"].values)
    sig["p_fullmatch_holm"] = holm(sig["p_fullmatch_mcnemar"].values)
    sig_path = os.path.join(save_dir, "TABLE2_significance_holm.csv")
    sig.to_csv(sig_path, index=False)
    print("Saved TABLE2_significance_holm.csv")

    # 8. Pareto Figure with Curves for Fixed-k, Arm 4b, Arm 5, and Arm 6
    print("\nComputing Pareto curves and interpolations at mean k=4.0...")
    # Sweep Arm 4b
    a4b_sweep_eval = {}
    for th in THETAS_SWEEP:
        a4b_sweep_eval[th] = run_single_pass_arm4(ex_eval, arm2_ce_scores, platt_by_type, threshold=th, per_type=True)

    # Sweep Arm 5
    a5_sweep_eval = {}
    for th in THETAS_SWEEP:
        a5_sweep_eval[th] = df_from(ex_eval, lambda ex: sel_gated(tr_eval[ex["id"]], theta=th, min_k=1))

    # Sweep Arm 6 across taus (with theta=0.18)
    a6_sweep_eval = {}
    for tau in TAUS_SWEEP:
        a6_sweep_eval[tau] = df_from(
            ex_eval,
            lambda ex: sel_arm6(tr_eval[ex["id"]], final_suff_model, tau, int(ex["type"] == "bridge"), theta=0.18)
        )

    # Linear interpolation at k=4.0
    interp_results = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for ax, t in zip(axes, TYPES):
        # 1. Fixed-k CE re-ranking curve (k=1..6)
        fk_ks, fk_rs = [], []
        for k in range(1, 7):
            s = fixed_ce(ex_eval, arm2_ce_scores, k)
            s = s[s["type"] == t]
            fk_ks.append(k)
            fk_rs.append(s["recall"].mean())
        ax.plot(fk_ks, fk_rs, "k--o", ms=5, lw=2.0, label="Fixed-k (CE re-rank)", zorder=4)
        fk_k4_recall = fk_rs[3]

        # 2. Arm 4b theta sweep curve
        a4b_ks = [a4b_sweep_eval[th][a4b_sweep_eval[th]["type"] == t]["k_used"].mean() for th in THETAS_SWEEP]
        a4b_rs = [a4b_sweep_eval[th][a4b_sweep_eval[th]["type"] == t]["recall"].mean() for th in THETAS_SWEEP]
        order_4b = np.argsort(a4b_ks)
        a4b_ks_sorted = np.array(a4b_ks)[order_4b]
        a4b_rs_sorted = np.array(a4b_rs)[order_4b]
        ax.plot(a4b_ks_sorted, a4b_rs_sorted, "b-v", ms=4, lw=1.5, alpha=0.85, label="Arm 4b (single-pass Platt sweep)")
        a4b_k4_recall = float(np.interp(4.0, a4b_ks_sorted, a4b_rs_sorted))

        # 3. Arm 5 theta sweep curve
        a5_ks = [a5_sweep_eval[th][a5_sweep_eval[th]["type"] == t]["k_used"].mean() for th in THETAS_SWEEP]
        a5_rs = [a5_sweep_eval[th][a5_sweep_eval[th]["type"] == t]["recall"].mean() for th in THETAS_SWEEP]
        order_5 = np.argsort(a5_ks)
        a5_ks_sorted = np.array(a5_ks)[order_5]
        a5_rs_sorted = np.array(a5_rs)[order_5]
        ax.plot(a5_ks_sorted, a5_rs_sorted, "r-*", ms=6, lw=1.8, alpha=0.85, label="Arm 5 (iterative Platt sweep)")
        a5_k4_recall = float(np.interp(4.0, a5_ks_sorted, a5_rs_sorted))

        # 4. Arm 6 tau sweep curve
        a6_ks = [a6_sweep_eval[tau][a6_sweep_eval[tau]["type"] == t]["k_used"].mean() for tau in TAUS_SWEEP]
        a6_rs = [a6_sweep_eval[tau][a6_sweep_eval[tau]["type"] == t]["recall"].mean() for tau in TAUS_SWEEP]
        order_6 = np.argsort(a6_ks)
        a6_ks_sorted = np.array(a6_ks)[order_6]
        a6_rs_sorted = np.array(a6_rs)[order_6]
        ax.plot(a6_ks_sorted, a6_rs_sorted, "g-s", ms=4, lw=1.5, alpha=0.75, label="Arm 6 (sufficiency tau sweep)")
        a6_k4_recall = float(np.interp(4.0, a6_ks_sorted, a6_rs_sorted))

        # Discrete points: Iterative k=4 and k=5
        iter4_sub = iter_k4_df[iter_k4_df["type"] == t]
        iter5_sub = arms["Iter k=5 (no gate)"][arms["Iter k=5 (no gate)"]["type"] == t]
        ax.scatter([4.0], [iter4_sub["recall"].mean()], marker="P", s=110, c="purple", label="Iter k=4 (no gate)", zorder=6)
        ax.scatter([5.0], [iter5_sub["recall"].mean()], marker="D", s=90, c="magenta", label="Iter k=5 (no gate)", zorder=6)

        # Highlight interpolated recall at k=4.0
        ax.axvline(4.0, color="gray", linestyle=":", alpha=0.6)

        interp_results.append({
            "type": t,
            "Fixed-k (k=4)": fk_k4_recall,
            "Iter k=4 (no gate)": iter4_sub["recall"].mean(),
            "Arm 5 (interp k=4)": a5_k4_recall,
            "Arm 4b (interp k=4)": a4b_k4_recall,
            "Arm 6 (interp k=4)": a6_k4_recall,
        })

        ax.set_title(f"HotpotQA: {t.capitalize()} Multi-Hop", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Retrieval Depth (k chunks)", fontsize=11)
        ax.set_ylabel("Paragraph Recall", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    pareto_path = os.path.join(save_dir, "fig_pareto_recall_vs_k.png")
    plt.savefig(pareto_path)
    plt.close()
    print(f"Saved revised Pareto figure to {pareto_path}")

    interp_df = pd.DataFrame(interp_results)
    interp_df.to_csv(os.path.join(save_dir, "TABLE_interpolated_k4_recall.csv"), index=False)
    print("\nInterpolated Recall at Mean k=4.00:")
    print(interp_df.round(4).to_string(index=False))

    return {
        "table1": table1,
        "table2": sig,
        "cal_q": cal_q,
        "interp_df": interp_df,
        "auc_cal_cv": auc_cal_cv,
        "auc_cal_cv_std": auc_cal_cv_std,
        "auc_eval": auc_eval,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="full", choices=["task1", "full"])
    args = parser.parse_args()

    data = load_data("./results_full")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    passed, tr_eval, _, _ = run_task1_reproduction(data, ce)
    if not passed:
        print("ERROR: Task 1 reproduction check failed!")
        sys.exit(1)
    results = run_task2_and_3(data, ce, tr_eval)
    print("\n>>> Revised Tasks 2 and 3 executed successfully! <<<")

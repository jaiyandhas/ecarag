import numpy as np
import pandas as pd
from scipy.stats import binomtest

def pboot(a, b, metric="recall", n_boot=10000, seed=42):
    m = a[["id", metric]].merge(b[["id", metric]], on="id", suffixes=("_a", "_b"))
    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).values
    bm = np.random.default_rng(seed).choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = 2 * min((bm <= 0).mean(), (bm >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / n_boot)))

def mcnemar(a, b):
    m = a[["id", "full_match"]].merge(b[["id", "full_match"]], on="id", suffixes=("_a", "_b"))
    n10 = int(((m["full_match_a"] == 1) & (m["full_match_b"] == 0)).sum())
    n01 = int(((m["full_match_a"] == 0) & (m["full_match_b"] == 1)).sum())
    return 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)

def holm(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m, dtype=float)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj

def ece(p, yv, bins=10):
    p, yv = np.asarray(p, float), np.asarray(yv, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(yv[idx == b].mean() - p[idx == b].mean()) * (idx == b).sum() / len(p)
                     for b in range(bins) if (idx == b).any()))

def brier(p, yv):
    p, yv = np.asarray(p, float), np.asarray(yv, float)
    return float(np.mean((p - yv) ** 2))

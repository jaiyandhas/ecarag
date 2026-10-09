# ECA-RAG: Calibrated Confidence, Hop-Conditioned Re-Ranking, and the Limits of Adaptive-$k$ Retrieval in Multi-Hop RAG

Official research codebase, data artifacts, and evaluation pipeline for the paper:
**"ECA-RAG: Calibrated Confidence, Hop-Conditioned Re-Ranking, and the Limits of Adaptive-$k$ Retrieval in Multi-Hop RAG"**.

---

## Authors & Affiliation

- **Jaiyandh Amutha Sugumar** (`jaiyandh@gmail.com`) — *Conceptualization, Methodology, Software, Formal Analysis*
- **Adwaith Murali** (`adwaithmurali2005@gmail.com`) — *Writing – original draft, Writing – review & editing, Supervision, Project administration*
- **Ariya Kumar** (`ariyakumar258@gmail.com`) — *Investigation, Data curation, Validation, Visualization*

*Department of Computer Science and Engineering, Kongu Engineering College, Perundurai, Erode, Tamil Nadu, India*

---

## Overview

Retrieval-Augmented Generation (RAG) typically employs a static retrieval depth ($k$) across all user queries regardless of reasoning complexity. In multi-hop reasoning, static $k$ induces a sharp dilemma: low $k$ leads to missing necessary intermediate evidence (under-retrieval), while high $k$ injects extraneous distractors that degrade generation quality and escalate computational cost.

This repository implements the complete six-arm experimental framework of **ECA-RAG**, evaluating calibrated confidence gating, score-based stopping, hop-conditioned query reformulation, and learned sufficiency classification on the **HotpotQA** benchmark ($N=540$ core evaluation split and $N=2,100$ independent extension split).

---

## Experimental Framework (The Six Arms)

| Arm | Method | Retrieval Depth ($k$) | Description |
|:---:|:---|:---:|:---|
| **Arm 1** | CA-RAG Bi-Encoder Baseline | Static ($k=2$) | Dense bi-encoder retrieval using semantic chunking. |
| **Arm 2** | Cross-Encoder Direct Re-Ranking | Static ($k=2$) | Full cross-encoder scoring over bi-encoder candidates. |
| **Arm 3** | Calibrated Cross-Encoder | Static ($k=2$) | Type-conditioned Platt scaling mapped to empirical precision. |
| **Arm 4a / 4b** | Score-Gated Adaptive Depth | Adaptive ($k \in [1, 5]$) | Confidence-gated halting using fixed/type-specific thresholds ($\theta^*$). |
| **Arm 5** | Hop-Conditioned Iterative Re-Ranking | Adaptive ($k \in [1, 6]$) | Reformulates query with retrieved chunk text at each hop with calibrated threshold gating ($\theta^*$). |
| **Arm 6** | Learned Sufficiency-Gated Retrieval | Dynamic ($k \in [1, 6]$) | Hop-conditioned iteration paired with a calibrated logistic sufficiency classifier ($\tau^*$). |

---

## Key Empirical Findings

1. **Hop-Conditioning Drives Asymmetric Gains on Bridge Questions**:
   - On bridge questions requiring sequential link traversal, hop-conditioned query reformulation (**Arm 5**) attains **92.59% recall** at mean depth $k = 3.99$, outperforming fixed-$k=4$ retrieval (89.07%, $\Delta = +0.0352$, $p < 0.001$) and ungated iteration at $k=4$ (90.00%, $\Delta = +0.0259$, $p = 0.002$).
2. **Comparison Split Disconnect**:
   - Multi-hop comparison questions saturate rapidly at $k=2$ (**98.89% recall** in Arm 2), because both entity candidates appear in the top initial pool. Gated iteration yields minimal benefit on comparisons and can introduce drift if forced into additional hops.
3. **Calibrated Confidence Gating**:
   - Type-conditioned Platt scaling achieves tight probability alignment with Expected Calibration Error (ECE) of **0.057** on bridge queries and **0.081** on comparison queries (validated at **0.053** and **0.086** on the $N=2,100$ extension split).
4. **Sufficiency Stopper Efficiency**:
   - Learned sufficiency halting (**Arm 6**) prunes early hops when sufficient evidence has been accumulated, reducing mean depth from $k=3.99$ to $k=3.30$ on bridge queries while retaining **89.63% recall**.

---

## Repository Structure

```text
EcaRag/
├── ECA-RAG_final.tex               # Primary LaTeX source of truth (10-page camera-ready)
├── ECA-RAG_final.pdf               # Compiled camera-ready PDF manuscript
├── ECA-RAG_final.docx              # Synchronized IEEE-format Word manuscript
├── references.bib                  # BibTeX bibliography
├── ieee.csl                        # IEEE Citation Style Language definition
├── custom-reference.docx           # Pandoc Word styling reference
├── docx-build.tex                  # Preprocessed LaTeX intermediate for Pandoc
├── CLAIMS_TRACE.md                 # Complete provenance mapping paper claims to CSV rows
├── number_audit_tokens_provenance.csv  # Token-level audit trace
├── number_audit.py                 # Automated numerical consistency auditor
├── requirements.txt                # Python environment dependencies
│
├── figures/                        # High-resolution Pareto frontier plots
│   ├── fig_pareto_recall_vs_k.pdf  # Vector PDF of Figure 1
│   └── fig_pareto_recall_vs_k.png  # PNG raster of Figure 1
│
├── results_full/                   # Authoritative ground-truth data and result tables
│   ├── TABLE1_full.csv             # Full benchmark results (Arms 1–5, N=540)
│   ├── TABLE_extension_900_3000.csv# Extension benchmark results (N=2,100)
│   ├── TABLE_calibration_quality.csv   # Calibration metrics (ECE, Brier, log-loss)
│   ├── TABLE_calibration_extension.csv # Extension calibration metrics
│   ├── TABLE_matched_depth_k4_bootstrap.csv # Matched-depth k=4 bootstrap contrasts
│   ├── TABLE_matched_depth_k4_point_estimates.csv # Point estimates at k=4
│   ├── TABLE_multi_depth_significance.csv   # Multi-depth bootstrap significance
│   ├── TABLE_multi_depth_bridge.csv         # Bridge-specific multi-depth sweep
│   ├── TABLE_multi_depth_comparison.csv     # Comparison-specific multi-depth sweep
│   ├── TABLE_multi_depth_point_estimates.csv# Multi-depth point estimates
│   ├── TABLE_sufficiency_auc.csv            # Sufficiency classifier ROC-AUC
│   ├── TABLE_sweep_arm5_bridge.csv          # Arm 5 threshold sweep data
│   ├── arm1_results.csv / arm2_results.csv  # Raw per-query evaluation scores
│   ├── calibration_config.pkl               # Fitted Platt scaling parameters
│   └── precomputed_multi_depth_arrays.pkl   # Precomputed bootstrap arrays
│
├── src/                            # Modular core implementation package
│   ├── __init__.py
│   ├── arms.py                     # Execution logic for Arms 1 through 6
│   ├── data.py                     # Dataset loading, Platt probability, metric utils
│   └── stats.py                    # Bootstrap CI estimation and Holm-Bonferroni correction
│
└── scripts/                        # Reproducibility and build utilities
    ├── recompute_matched_and_multi_depth.py # Master unified script for Tables 4, 5 & Fig. 1
    ├── compute_calibration_extension.py     # Extension split calibration recomputation
    ├── generate_sufficiency_auc.py          # Sufficiency stopper AUC analysis
    ├── hand_coded_number_check.py           # Strict precision claim verifier
    ├── count_abstract.py                    # LaTeX abstract word counter
    ├── build_docx_tex.py                    # Pandoc LaTeX preprocessor
    └── study_constants.json                 # Canonical evaluation thresholds and seeds
```

---

## Installation & Setup

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/jaiyandhas/ecarag.git
cd ecarag
pip install -r requirements.txt
```

---

## Reproduction Guide

### 1. Unified Matched-Depth & Multi-Depth Analysis (Tables IV, V & Figure 1)
To reproduce the bootstrap confidence intervals ($B=10,000$), Holm-adjusted $p$-values, and Pareto frontier curves:

```bash
python scripts/recompute_matched_and_multi_depth.py
```
*Outputs generated:*
- `results_full/TABLE_matched_depth_k4_bootstrap.csv`
- `results_full/TABLE_matched_depth_k4_point_estimates.csv`
- `results_full/TABLE_multi_depth_significance.csv`
- `results_full/TABLE_multi_depth_bridge.csv`
- `results_full/TABLE_multi_depth_comparison.csv`
- `figures/fig_pareto_recall_vs_k.pdf`
- `figures/fig_pareto_recall_vs_k.png`

### 2. Verify Claims Traceability and Numeric Precision
To audit all quantitative assertions in the paper against raw CSV files:

```bash
python scripts/hand_coded_number_check.py
python number_audit.py
```

### 3. Recompute Calibration Metrics (Table III)
To re-evaluate Platt scaling ECE, Brier score, and log-loss:

```bash
python scripts/compute_calibration_extension.py
python scripts/generate_sufficiency_auc.py
```

### 4. Compile Paper Manuscripts
- **Camera-Ready LaTeX PDF (Exactly 10 pages)**:
  ```bash
  tectonic ECA-RAG_final.tex
  ```
- **IEEE-Formatted Word (.docx)**:
  ```bash
  python scripts/build_docx_tex.py
  pandoc docx-build.tex -o ECA-RAG_final.docx --reference-doc=custom-reference.docx --citeproc --csl=ieee.csl
  ```

---

## Traceability & Data Provenance

Every quantitative claim, table cell, confidence interval, and $p$-value reported in the paper is strictly tied to an exact row and column in `results_full/`. See [`CLAIMS_TRACE.md`](file:///Users/jaiyandh/Downloads/EcaRag/CLAIMS_TRACE.md) for the exhaustive cell-by-cell mapping.

---

## Citation

```bibtex
@article{sugumar2026ecarag,
  author    = {Sugumar, Jaiyandh Amutha and Murali, Adwaith and Kumar, Ariya},
  title     = {{ECA-RAG}: Calibrated Confidence, Hop-Conditioned Re-Ranking, and the Limits of Adaptive-$k$ Retrieval in Multi-Hop {RAG}},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.

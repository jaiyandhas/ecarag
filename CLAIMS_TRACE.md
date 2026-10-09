# ECA-RAG: Audit Trace of Empirical Claims to CSV Sources

This document maps every empirical claim and quantitative metric in the revised paper to its exact origin in the produced CSV files in `results_full/`.

---

## 1. Calibration Quality (Section IV-A & Abstract)
*Source File:* [`results_full/TABLE_calibration_quality.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_calibration_quality.csv)
*Note on Dataset Split:* As explicitly reported in the Abstract, Section I, and Section IV-A, ECE and Brier scores are evaluated strictly on the **540-example held-out evaluation split** (5,385 candidate chunk-level predictions), *not* on the calibration split or the pooled $N=2640$ dataset.

| Claim / Metric in Paper | Method / Subset | Value | CSV Row | CSV Column |
|---|---|---|---|---|
| Raw cross-encoder sigmoid ECE | Raw sigmoid($s$) | 0.2248 | Row 2 | `ECE` |
| Raw cross-encoder sigmoid Brier score | Raw sigmoid($s$) | 0.2127 | Row 2 | `Brier` |
| Platt pooled ECE | Platt pooled | 0.0419 | Row 3 | `ECE` |
| Platt pooled Brier score | Platt pooled | 0.1161 | Row 3 | `Brier` |
| Platt per-type ECE | Platt per-type | 0.0391 | Row 4 | `ECE` |
| Platt per-type Brier score | Platt per-type | 0.1157 | Row 4 | `Brier` |
| ECE reduction (0.2248 $\to$ 0.0391) | Relative error reduction | 82.6% | Derived: $(0.2248-0.0391)/0.2248$ | - |

### B. Generalization to Extension Split ($N=2100$, 20,901 Candidate Chunks, Section IV-A)
*Source File:* [`results_full/TABLE_calibration_extension.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_calibration_extension.csv)

| Claim / Metric in Paper | Method / Subset | Value | CSV Row | CSV Column |
|---|---|---|---|---|
| Extension chunk count | $N=2100$ candidate chunks | 20,901 | Rows 2–4 | `chunks` |
| Extension raw sigmoid ECE | Raw sigmoid($s$) | 0.2143 | Row 2 | `ECE` |
| Extension raw sigmoid Brier score | Raw sigmoid($s$) | 0.2039 | Row 2 | `Brier` |
| Extension Platt pooled ECE | Platt pooled | 0.0389 | Row 3 | `ECE` |
| Extension Platt pooled Brier score | Platt pooled | 0.1162 | Row 3 | `Brier` |
| Extension Platt per-type ECE | Platt per-type | 0.0373 | Row 4 | `ECE` |
| Extension Platt per-type Brier score | Platt per-type | 0.1146 | Row 4 | `Brier` |
| Extension ECE reduction (0.2143 $\to$ 0.0373) | Relative error reduction | 82.6% | Derived: $(0.2143-0.0373)/0.2143$ | - |

---

## 2. Matched-Depth Performance at $k = 4.00$ (Headline Table, Section IV-B)
*Source Files:*
- Point Estimates: [`results_full/TABLE_matched_depth_k4_point_estimates.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_matched_depth_k4_point_estimates.csv)
- Bootstrap CIs & Holm Tests: [`results_full/TABLE_matched_depth_k4_bootstrap.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_matched_depth_k4_bootstrap.csv)

### A. Point Estimates (Recall at $k = 4.00$)
| Split | Type | Fixed-$k$ ($k=4$) | Iter $k=4$ (no gate) | Arm 5 (interp $k=4$) | Arm 4b (interp $k=4$) | Arm 6 (interp $k=4$) | CSV Row |
|---|---|---|---|---|---|---|---|
| $N=540$ (Eval) | bridge | 0.8054 | 0.8160 | 0.8013 | 0.7693 | n/a | Row 2 |
| $N=540$ (Eval) | comp | 0.9483 | 0.9181 | 0.8649 | 0.9479 | 0.8712$^*$ | Row 3 |
| $N=2100$ (Ext) | bridge | 0.8076 | 0.8261 | 0.7978 | 0.7835 | n/a | Row 4 |
| $N=2100$ (Ext) | comp | 0.9553 | 0.9329 | 0.9273 | 0.9530 | 0.9280$^*$ | Row 5 |
| **Pooled ($N=2640$)** | bridge | 0.8072 | **0.8240** | 0.7983 | 0.7806 | n/a | Row 6 |
| **Pooled ($N=2640$)** | comp | **0.9538** | 0.9298 | 0.9147 | 0.9520 | 0.9168$^*$ | Row 7 |

*Note on Arm 6:* On bridge questions, Arm 6 plateaus at mean depth $k=3.87$ and never reaches $k=4.00$ (reported as n/a). On comparison questions, point estimates reach $k \approx 4.04\text{--}4.20$ ($0.8712$, $0.9280$, $0.9168$), marked $^*$ to indicate that fewer than 95% of bootstrap resamples reach $k=4.00$, precluding valid bootstrap hypothesis testing and confidence intervals at $k=4.00$.

### B. Statistical Significance & Holm Adjustment (10,000 Resamples)
*From `TABLE_matched_depth_k4_bootstrap.csv`*

1. **Iter $k=4$ vs. Arm 2 ($k=4$) [Hop-conditioning alone at exact $k=4.00$]**:
   - $N=540$ bridge: $\Delta = +0.0106$ (95% CI [$-0.0094, +0.0307$], $p = 0.316$, $p_{\text{Holm}} = 0.632$) -> Row 6
   - $N=2100$ bridge: $\Delta = \mathbf{+0.0185}$ (95% CI [$+0.0082, +0.0287$], $p < 0.001$, $p_{\text{Holm}} = 0.0024$) -> Row 16
   - Pooled bridge: $\Delta = \mathbf{+0.0169}$ (95% CI [$+0.0077, +0.0258$], $p < 0.001$, $p_{\text{Holm}} = 0.0008$) -> Row 26
   - Pooled comp: $\Delta = -0.0240$ (95% CI [$-0.0391, -0.0089$], $p = 0.0024$, $p_{\text{Holm}} = 0.0072$) -> Row 31

2. **Arm 5 (interp $k=4$) vs. Iter $k=4$ [Gating cost at matched depth]**:
   - $N=540$ bridge: $\Delta = -0.0147$ (95% CI [$-0.0307, +0.0008$], $p = 0.0624$, $p_{\text{Holm}} = 0.1872$) -> Row 2
   - $N=2100$ bridge: $\Delta = -0.0283$ (95% CI [$-0.0363, -0.0204$], $p < 0.001$, $p_{\text{Holm}} < 0.001$) -> Row 12
   - Pooled bridge: $\Delta = \mathbf{-0.0257}$ (95% CI [$-0.0328, -0.0186$], $p < 0.001$, $p_{\text{Holm}} < 0.001$) -> Row 22

3. **Arm 5 (interp $k=4$) vs. Arm 2 ($k=4$) [Iter-gated vs. Fixed-$k$ at $k=4$]**:
   - $N=540$ bridge: $\Delta = -0.0041$ (95% CI [$-0.0246, +0.0161$], $p = 0.6756$, $p_{\text{Holm}} = 0.6756$, not significant) -> Row 3
   - $N=2100$ bridge: $\Delta = -0.0099$ (95% CI [$-0.0209, +0.0014$], $p = 0.0904$, $p_{\text{Holm}} = 0.0904$, not significant) -> Row 13
   - Pooled bridge: $\Delta = -0.0088$ (95% CI [$-0.0187, +0.0010$], raw $p = 0.0794$, Holm $p = 0.0794$, not significant) -> Row 23
   - Pooled comp: $\Delta = -0.0391$ (95% CI [$-0.0565, -0.0221$], $p < 0.001$, $p_{\text{Holm}} < 0.001$) -> Row 28

4. **Arm 4b (interp $k=4$) vs. Arm 2 ($k=4$) [Single-pass calib vs. Fixed-$k$ at $k=4$]**:
   - Pooled bridge: $\Delta = \mathbf{-0.0266}$ (95% CI [$-0.0341, -0.0191$], $p < 0.001$, $p_{\text{Holm}} < 0.001$) -> Row 24
   - Pooled comp: $\Delta = -0.0018$ (95% CI [$-0.0136, +0.0095$], raw $p = 0.7202$, $p_{\text{Holm}} = 0.7202$, indistinguishable) -> Row 29

5. **Arm 5 (interp $k=4$) vs. Arm 4b (interp $k=4$) [Iterative vs. Single-pass gating]**:
   - Pooled bridge: $\Delta = \mathbf{+0.0177}$ (95% CI [$+0.0084, +0.0269$], raw $p < 0.001$, $p_{\text{Holm}} = 0.0006$) -> Row 25

---

## 3. Full Retrieval Split Tables (Section IV-C)
*Source Files:*
- Initial Eval ($N=540$): [`results_full/TABLE1_full.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE1_full.csv)
- Extension Eval ($N=2100$): [`results_full/TABLE_extension_900_3000.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_extension_900_3000.csv)

### A. Re-Ranking Effect (Arm 2 vs. Arm 1)
- Initial Eval Bridge ($N=424$):
  - Arm 1: Recall 0.7559, Prec 0.3779, Full-M 0.5354, $k=4.000$ (`TABLE1_full.csv`, Row 2)
  - Arm 2: Recall 0.8054, Prec 0.4027, Full-M 0.6132, $k=4.000$ (`TABLE1_full.csv`, Row 4)
  - Delta: $+0.0495$, 95% CI [$+0.0248, +0.0731$], $p = 0.0001$, Holm $p = 0.0014$; McNemar $p = 0.0013$ (`TABLE2_significance_holm.csv`, Row 14)
- Extension Eval Bridge ($N=1653$):
  - Arm 1: Recall 0.7429, Prec 0.3728, Full-M 0.5275, $k=4.000$ (`TABLE_extension_900_3000.csv`, Row 2)
  - Arm 2: Recall 0.8076, Prec 0.4051, Full-M 0.6261, $k=3.995$ (`TABLE_extension_900_3000.csv`, Row 4)
  - Delta: $+0.0647$, 95% CI [$+0.0511, +0.0783$], $p < 0.001$, Holm $p = 0.0014$; McNemar $p < 10^{-14}$ (`TABLE2_extension_significance.csv`, Row 14)

### B. Iterative Depth $k=5$ (Iter $k=5$ vs. Arm 2 $k=5$)
- Initial Eval Bridge ($N=424$):
  - Arm 2 $k=5$: Recall 0.8314, Full-M 0.6627 (`TABLE1_full.csv`, Row 6)
  - Iter $k=5$: Recall 0.8514, Full-M 0.7099 (`TABLE1_full.csv`, Row 24)
  - Delta: $+0.0200$, 95% CI [$+0.0012, +0.0389$], bootstrap raw $p = 0.0464$, Holm $p = 0.4640$; full-match McNemar $p = 0.0187$, Holm $p = 0.1866$ (`TABLE2_significance_holm.csv`, Row 6)
- Extension Eval Bridge ($N=1653$):
  - Arm 2 $k=5$: Recall 0.8445, Full-M 0.6975 (`TABLE_extension_900_3000.csv`, Row 6)
  - Iter $k=5$: Recall 0.8542, Full-M 0.7181 (`TABLE_extension_900_3000.csv`, Row 10)
  - Delta: $+0.0097$, 95% CI [$0.0000, +0.0194$], bootstrap raw $p = 0.0548$, Holm $p = 0.2192$; full-match McNemar $p = 0.0344$, Holm $p = 0.1377$ (`TABLE2_extension_significance.csv`, Row 6)
- *Conclusion on $k=5$:* Differences are small and not statistically significant after Holm correction across both splits and question types.

### C. Oracle-Type Hybrid Upper Bound
*Iterative $k=4$ (no gate) on bridge, Arm 2 ($k=4$) on comparison*
- Initial Eval ($N=540$): Bridge Recall 0.8160 (Full-M 0.6462), Comparison Recall 0.9483 (Full-M 0.8966), $k = 4.000$ uniform (`TABLE1_full.csv`, Rows 32–33)
- Extension Eval ($N=2100$): Bridge Recall 0.8261 (Full-M 0.6673), Comparison Recall 0.9553 (Full-M 0.9128), $k = 3.995$ uniform (`TABLE_extension_900_3000.csv`, Rows 20–21)
- Pooled ($N=2640$): Bridge Recall 0.8240, Comparison Recall 0.9538, $k = 4.000$ uniform (`TABLE_matched_depth_k4_point_estimates.csv`, Rows 6–7)

---

## 4. Sufficiency Stopper Model (Section IV-E)
*Source Code & Execution Logs:* `matched_depth_bootstrap.py`, `task-358.log`
- 5-Fold Cross-Validation on Calibration Split ($N=360$, 1,788 hop instances): $\text{ROC AUC} = \mathbf{0.7977 \pm 0.0179}$
- Held-out Evaluation Split ($N=540$, 2,642 hop instances): $\text{ROC AUC} = \mathbf{0.7964}$
- Null Result Finding: At $\tau^* = 0.95$ (tuned on calibration split for $k \le 4.0$), recall at $k=4.00$ is $0.7983$ ($N=540$) and $0.7913$ (Pooled), falling below fixed-$k$ ($0.8054$ and $0.8072$).

---

## 5. Multi-Depth Matched-Depth Bootstrap Analysis Across $k \in \{2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5\}$
*Source Files:*
- Point Estimates: [`results_full/TABLE_multi_depth_point_estimates.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_multi_depth_point_estimates.csv)
- Significance & Holm Tests: [`results_full/TABLE_multi_depth_significance.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_multi_depth_significance.csv)
- Bridge Specific: [`results_full/TABLE_multi_depth_bridge.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_multi_depth_bridge.csv)
- Comparison Specific: [`results_full/TABLE_multi_depth_comparison.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE_multi_depth_comparison.csv)

### A. Point Estimates (Pooled $N=2640$)
- **Bridge ($N=2077$):**
  - $k=2.0$: Fixed-$k = 0.6810$, Arm 4b = $0.6110$, Arm 5 = $0.6289$, Arm 6 = $0.6814$
  - $k=2.5$: Fixed-$k = 0.7214$, Arm 4b = $0.6685$, Arm 5 = $0.6838$, Arm 6 = $0.7302$
  - $k=3.0$: Fixed-$k = 0.7617$, Arm 4b = $0.7107$, Arm 5 = $0.7279$, Arm 6 = $0.7616$
  - $k=3.5$: Fixed-$k = 0.7844$, Arm 4b = $0.7483$, Arm 5 = $0.7666$, Arm 6 = $0.7811$
  - $k=4.0$: Fixed-$k = 0.8072$, Arm 4b = $0.7806$, Arm 5 = $0.7983$, Arm 6 = NC (plateaus at 3.87)
  - $k=4.5$: Fixed-$k = 0.8245$, Arm 4b = $0.8088$, Arm 5 = $0.8229$, Arm 6 = NC
  - $k=5.0$: Fixed-$k = 0.8418$, Arm 4b = $0.8327$, Arm 5 = $0.8449$, Arm 6 = NC
  - $k=5.5$: Fixed-$k = 0.8533$, Arm 4b = $0.8509$, Arm 5 = $0.8660$, Arm 6 = NC
- **Comparison ($N=563$):**
  - $k=2.0$: Fixed-$k = 0.8428$, Arm 4b = $0.7035$, Arm 5 = $0.6643$, Arm 6 = $0.7052$
  - $k=2.5$: Fixed-$k = 0.8841$, Arm 4b = $0.8010$, Arm 5 = $0.7394$, Arm 6 = $0.7940$
  - $k=3.0$: Fixed-$k = 0.9254$, Arm 4b = $0.8669$, Arm 5 = $0.8056$, Arm 6 = $0.8595$
  - $k=3.5$: Fixed-$k = 0.9396$, Arm 4b = $0.9183$, Arm 5 = $0.8611$, Arm 6 = $0.8955$
  - $k=4.0$: Fixed-$k = 0.9538$, Arm 4b = $0.9520$, Arm 5 = $0.9147$, Arm 6 = $0.9168$
  - $k=4.5$: Fixed-$k = 0.9627$, Arm 4b = $0.9627$, Arm 5 = $0.9444$, Arm 6 = NC
  - $k=5.0$: Fixed-$k = 0.9716$, Arm 4b = $0.9725$, Arm 5 = $0.9658$, Arm 6 = NC
  - $k=5.5$: Fixed-$k = 0.9765$, Arm 4b = $0.9773$, Arm 5 = $0.9762$, Arm 6 = NC

### B. Statistical Significance Across Depth Family (Holm Adjusted)
- **Arm 6 vs Fixed-$k$ on Bridge (No Detectable Difference strictly on Bridge):**
  - $k=2.0$: $\Delta = +0.0003$, 95% CI [$-0.0100, +0.0105$], raw $p = 0.9524$, Holm $p = 1.0000$ (no detectable difference; 95% CI includes zero)
  - $k=2.5$: $\Delta = +0.0088$, 95% CI [$-0.0008, +0.0183$], raw $p = 0.0720$, Holm $p = 0.2880$ (CI includes zero; not significant)
  - $k=3.0$: $\Delta = -0.0000$, 95% CI [$-0.0099, +0.0102$], raw $p = 0.9866$, Holm $p = 1.0000$ (no detectable difference; 95% CI includes zero)
  - $k=3.5$: $\Delta = -0.0033$, 95% CI [$-0.0132, +0.0067$], raw $p = 0.5170$, Holm $p = 1.0000$ (no detectable difference; 95% CI includes zero)
  - $k \ge 4.0$: Not Covered (NC)
- **Arm 6 vs Fixed-$k$ on Comparison (Significant Degradation across all covered depths):**
  - $k=2.0$: $\Delta = -0.1376$, 95% CI [$-0.1576, -0.1175$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 18)
  - $k=2.5$: $\Delta = -0.0901$, 95% CI [$-0.1094, -0.0702$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 19)
  - $k=3.0$: $\Delta = -0.0659$, 95% CI [$-0.0861, -0.0453$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 20)
  - $k=3.5$: $\Delta = -0.0441$, 95% CI [$-0.0617, -0.0260$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 21)
  - *Summary:* Arm 6 is significantly worse than fixed-$k$ on comparison questions at every covered depth $k \in [2.0, 3.5]$ ($\Delta = -0.138$ to $-0.044$, all Holm $p < 0.001$).
- **Arm 6 vs Arm 5 (Significant Improvement on Both Question Types at $k \in [2.0, 3.5]$):**
  - **Bridge:**
    - $k=2.0$: $\Delta = \mathbf{+0.0525}$, 95% CI [$+0.0453, +0.0596$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_bridge.csv`, Row 26)
    - $k=2.5$: $\Delta = \mathbf{+0.0464}$, 95% CI [$+0.0390, +0.0538$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_bridge.csv`, Row 27)
    - $k=3.0$: $\Delta = \mathbf{+0.0338}$, 95% CI [$+0.0272, +0.0402$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_bridge.csv`, Row 28)
    - $k=3.5$: $\Delta = \mathbf{+0.0145}$, 95% CI [$+0.0095, +0.0198$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_bridge.csv`, Row 29)
  - **Comparison:**
    - $k=2.0$: $\Delta = \mathbf{+0.0410}$, 95% CI [$+0.0301, +0.0521$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 26)
    - $k=2.5$: $\Delta = \mathbf{+0.0547}$, 95% CI [$+0.0420, +0.0673$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 27)
    - $k=3.0$: $\Delta = \mathbf{+0.0539}$, 95% CI [$+0.0413, +0.0669$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 28)
    - $k=3.5$: $\Delta = \mathbf{+0.0344}$, 95% CI [$+0.0207, +0.0491$], raw $p < 0.001$, Holm $p < 0.001$ (`TABLE_multi_depth_comparison.csv`, Row 29)
  - *Summary:* Learned sufficiency stopping eliminates the severe efficiency penalty of Arm 5 at low depths across both question types (all Holm $p < 0.001$).
- **Arm 5 vs Fixed-$k$ on Bridge (Depth-Dependent Trajectory):**
  - $k=2.0$: $\Delta = -0.0522$, 95% CI [$-0.0626, -0.0418$], raw $p < 0.001$, Holm $p < 0.001$ (significant loss)
  - $k=2.5$: $\Delta = -0.0376$, 95% CI [$-0.0474, -0.0276$], raw $p < 0.001$, Holm $p < 0.001$ (significant loss)
  - $k=3.0$: $\Delta = -0.0338$, 95% CI [$-0.0442, -0.0232$], raw $p < 0.001$, Holm $p < 0.001$ (significant loss)
  - $k=3.5$: $\Delta = -0.0178$, 95% CI [$-0.0276, -0.0079$], raw $p = 0.0010$, Holm $p = 0.0050$ (significant loss)
  - $k=4.0$: $\Delta = -0.0088$, 95% CI [$-0.0187, +0.0010$], raw $p = 0.0794$, Holm $p = 0.2382$ (not significantly different)
  - $k=4.5$: $\Delta = -0.0016$, 95% CI [$-0.0107, +0.0074$], raw $p = 0.7344$, Holm $p = 1.0000$ (no detectable difference; 95% CI includes zero)
  - $k=5.0$: $\Delta = +0.0031$, 95% CI [$-0.0058, +0.0120$], raw $p = 0.5100$, Holm $p = 1.0000$ (no detectable difference; 95% CI includes zero)
  - $k=5.5$: $\Delta = \mathbf{+0.0127}$, 95% CI [$+0.0044, +0.0210$], raw $p = 0.0044$, Holm $p = 0.0176$ (exploratory significant gain)
- **Arm 4b vs Fixed-$k$ on Bridge:**
  - Significant loss across $k \in \{2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0\}$ (all Holm $p \le 0.0016$); at $k=5.5$, $\Delta = -0.0024$, 95% CI [$-0.0056, +0.0009$], raw $p=0.1572$, Holm $p=0.1572$ (no detectable difference; 95% CI includes zero).

---

## 6. Query Augmentation Prefix Length Ablation (Section IV-G)
*Source File:* [`results_full/TABLE1_full.csv`](file:///Users/jaiyandh/Downloads/EcaRag/results_full/TABLE1_full.csv)
*Note on Correction:* The paper originally contained an unsourced 100-word ablation. This has been removed. Section IV-G strictly compares default full-chunk concatenation ($q^{(2)} = q \oplus \text{ `` } \oplus d_{(1)}$) against 50-word prefix truncation ($q^{(2)} = q \oplus \text{ `` } \oplus \text{truncate}_{50}(d_{(1)})$) in single-pass gated re-ranking (Arm 4b) at both the operating threshold ($\theta^*=0.18$) and the sensitivity threshold ($\theta=0.15$). Note that Arm 4b is a single-pass method (not iterative), and this is a single-split descriptive comparison with no significance test.

| Threshold | Configuration | Question Type | Recall | Mean Depth $k$ | Precision | Full-Match | CSV Row in `TABLE1_full.csv` |
|---|---|---|---|---|---|---|---|
| $\theta^* = 0.18$ | Full-chunk (Default) | Bridge | 0.7983 | 3.934 | 0.5107 | 0.6108 | Row 20 |
| $\theta^* = 0.18$ | Full-chunk (Default) | Comparison | 0.8793 | 4.198 | 0.4947 | 0.7759 | Row 21 |
| $\theta^* = 0.18$ | 50-word prefix | Bridge | 0.8031 | 4.033 | 0.4983 | 0.6179 | Row 28 |
| $\theta^* = 0.18$ | 50-word prefix | Comparison | 0.9009 | 4.328 | 0.4864 | 0.8190 | Row 29 |
| $\theta = 0.15$ | Full-chunk (Sensitivity) | Bridge | 0.8243 | 4.455 | 0.4523 | 0.6604 | Row 22 |
| $\theta = 0.15$ | Full-chunk (Sensitivity) | Comparison | 0.9138 | 4.560 | 0.4701 | 0.8362 | Row 23 |
| $\theta = 0.15$ | 50-word prefix | Bridge | 0.8243 | 4.493 | 0.4494 | 0.6580 | Row 30 |
| $\theta = 0.15$ | 50-word prefix | Comparison | 0.9353 | 4.595 | 0.4691 | 0.8793 | Row 31 |

*Observed Differences:*
- At $\theta^* = 0.18$, 50-word prefix slightly increases mean retrieval depth ($+0.10$ chunks on bridge: $3.934 \to 4.033$; $+0.13$ chunks on comparison: $4.198 \to 4.328$). Recall increases marginally on bridge ($+0.005$: $0.7983 \to 0.8031$) and modestly on comparison ($+0.022$: $0.8793 \to 0.9009$), with $k$ higher by $\sim 0.1$.
- At $\theta = 0.15$, bridge recall difference is $0.000$ ($0.8243 \to 0.8243$), while comparison recall increases by $+0.022$ ($0.9138 \to 0.9353$) alongside an increase in mean depth ($4.560 \to 4.595$, higher by $\sim 0.04$ to $0.1$).

---

## 7. Limitations & Methodological Constraints (Section VI)
The following limitations are explicitly documented in Section VI:
1. **Calibration Split Independence:** Platt scaling parameters are fit on $N=360$ and applied without adaptation to $N=540$ and $N=2100$.
2. **Fixed Re-ranking Depth:** First-pass BM25 retrieval fixed at $k=10$ candidate chunks throughout.
3. **Retrieval-Stage Evaluation:** Focuses on retrieval precision, recall, and full-match; generation-stage answer accuracy remains future work.
4. **Single Distractor Benchmark:** Evaluated on HotpotQA distractor setting with BM25 candidate pooling.
5. **Heuristic Sufficiency Stopping:** Arm 6 employs a logistic regression stopper on chunk score summary features.
6. **Disjoint Extension Split:** The $N=2100$ extension examples (validation indices 900–2999) are strictly disjoint from the first 900 indices (`scripts/verify_disjoint_ids.py`).
7. **Exploratory Deep Retrieval ($k=5.5$) Caveats:**
   - Evaluated at the extreme tail of the threshold sweep ($\theta \le 0.08$).
   - Bootstrap resamples that failed to reach $k=5.5$ were excluded under the implementation choice of a 95% coverage requirement.
   - The depth grid $k \in \{2.0, \dots, 5.5\}$ was selected post-hoc following inspection of the Pareto curve. Consequently, all claims at $k=5.5$ remain explicitly exploratory.

---

## 8. Bibliography & Reference Web Verification
*Source File:* [`references.bib`](file:///Users/jaiyandh/Downloads/EcaRag/references.bib)

All 14 bibliographic entries were audited and verified against authoritative digital publishers (IEEE Xplore, ACL Anthology, OpenReview, PMLR, NeurIPS Proceedings, and arXiv).

| BibTeX Key | Title (as verified on page) | Authors | Venue / Year | Exact Confirmed URL / DOI | Match Status | Notes |
|---|---|---|---|---|---|---|
| `collini2025carag` | Context-Aware Retrieval Augmented Generation Using Similarity Validation to Handle Context Inconsistencies in Large Language Models | Enrico Collini, Felix Indra Kurniadi, Paolo Nesi, Gianni Pantaleo | IEEE Access, 2025 | DOI: `10.1109/ACCESS.2025.3614553` | Exact Match | Vol. 13, pp. 170065–170080 |
| `guo2017calibration` | On Calibration of Modern Neural Networks | Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger | ICML 2017 | `https://proceedings.mlr.press/v70/guo17a.html` | Exact Match | PMLR 70:1321–1330 |
| `jeong2024adaptiverag` | Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity | Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park | NAACL 2024 (Main Conference: Long Papers) | DOI: `10.18653/v1/2024.naacl-long.389` | Exact Match | ACL Anthology, pp. 7036–7050 |
| `kratzwald2018adaptive` | Adaptive Document Retrieval for Deep Question Answering | Bernhard Kratzwald, Stefan Feuerriegel | EMNLP 2018 | DOI: `10.18653/v1/D18-1055` | Exact Match | ACL Anthology, pp. 576–581 |
| `liu2024ctrla` | CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control | Huanshuo Liu, Hao Zhang, Zhijiang Guo, Jing Wang, Kuicai Dong, Xiangyang Li, Yi Quan Lee, Cong Zhang, Yong Liu | Findings of ACL 2025 | `https://arxiv.org/abs/2405.18727` | Exact Match | Findings of ACL 2025 (2025.findings-acl.652), arXiv:2405.18727 |
| `platt1999probabilistic` | Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods | John C. Platt | Adv. Large Margin Classifiers, 1999 | MIT Press, pp. 61–74 | Exact Match | MIT Press, pp. 61–74 |
| `wang2024targ` | Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG | Yufeng Wang, Lu Wei, Haibin Ling | TMLR 2026 | `https://openreview.net/forum?id=L8gYtUZfVU` | Exact Match | TMLR 2026, arXiv:2511.09803 |
| `yang2018hotpotqa` | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning | EMNLP 2018 | DOI: `10.18653/v1/D18-1259` | Exact Match | ACL Anthology, pp. 2369–2380 |
| `xiong2021mdr` | Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval | Wenhan Xiong, Xiang Lorraine Li, Srini Iyer, Jingfei Du, Patrick Lewis, William Yang Wang, Yashar Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe Kiela, Barlas Oguz | ICLR 2021 | `https://openreview.net/forum?id=EMHoBG0avc1` | Exact Match | OpenReview `EMHoBG0avc1`, arXiv:2009.12756 |
| `khattab2021baleen` | Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval | Omar Khattab, Christopher Potts, Matei Zaharia | NeurIPS 2021 | `https://proceedings.neurips.cc/paper_files/paper/2021/hash/e8b1cbd05f6e6a358a81dee52493dd06-Abstract.html` | Exact Match | NeurIPS Vol 34, pp. 27670–27682 |
| `trivedi2023ircot` | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Tasks | Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal | ACL 2023 | DOI: `10.18653/v1/2023.acl-long.557` | Exact Match | ACL Anthology, pp. 10014–10037 |
| `jiang2023flare` | Active Retrieval Augmented Generation | Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig | EMNLP 2023 | DOI: `10.18653/v1/2023.emnlp-main.495` | Exact Match | ACL Anthology, pp. 7969–7992 |
| `asai2024selfrag` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi | ICLR 2024 | `https://openreview.net/forum?id=hSyW5go0v8` | Exact Match | ICLR 2024 Oral, OpenReview `hSyW5go0v8` |
| `su2024dragin` | DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models | Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, Yiqun Liu | ACL 2024 | DOI: `10.18653/v1/2024.acl-long.702` | Exact Match | ACL Anthology, pp. 12991–13013 |

---

## 9. Headline Numbers Provenance Table (Deterministic Hand-Coded Mapping)

The following table provides the explicit, deterministic provenance for all headline numbers across the Abstract, Contributions, and Discussion. The automated script (`number_audit.py`) serves strictly as a complementary smoke test to verify token extraction; the mappings below are verified by direct cell lookups.

| Headline Metric / Comparison | Paper Token / Value | Source CSV File | Row | Column Name | Raw CSV Cell Value | Scale / Formula |
|---|---|---|---|---|---|---|
| ECE (Uncalibrated Raw) | 0.2248 (0.225) | `TABLE_calibration_quality.csv` | 0 | `ECE` | 0.2247655620455483 | Direct (scale = 1.0) |
| ECE (Platt Per-Type) | 0.0391 (0.039) | `TABLE_calibration_quality.csv` | 2 | `ECE` | 0.0391018237897594 | Direct (relative reduction 82.6%) |
| Brier Score (Uncalibrated Raw) | 0.2127 (0.213) | `TABLE_calibration_quality.csv` | 0 | `Brier` | 0.2126870976352218 | Direct (scale = 1.0) |
| Brier Score (Platt Per-Type) | 0.1157 (0.116) | `TABLE_calibration_quality.csv` | 2 | `Brier` | 0.1156608502521907 | Direct (relative reduction 45.6%) |
| Arm 4b vs. Fixed-$k$ Bridge $\Delta$ | $-0.0266$ (2.7 points) | `TABLE_matched_depth_k4_bootstrap.csv` | 22 | `delta_recall_k4` | -0.0265707751564757 | $\Delta \times -100 = 2.7$ points |
| Arm 4b vs. Fixed-$k$ Bridge Holm $p$ | $<0.001$ | `TABLE_matched_depth_k4_bootstrap.csv` | 22 | `p_holm` | 0.0005 | Reported as $p < 0.001$ |
| Arm 5 vs. Fixed-$k$ Bridge $\Delta$ | $-0.0088$ | `TABLE_matched_depth_k4_bootstrap.csv` | 21 | `delta_recall_k4` | -0.0088293911498925 | Direct ($\Delta = -0.0088$) |
| Arm 5 vs. Fixed-$k$ Bridge 95% CI Lo | $-0.0187$ | `TABLE_matched_depth_k4_bootstrap.csv` | 21 | `ci_lo` | -0.0186933551178903 | Direct |
| Arm 5 vs. Fixed-$k$ Bridge 95% CI Hi | $+0.0010$ | `TABLE_matched_depth_k4_bootstrap.csv` | 21 | `ci_hi` | 0.0010422837437330 | Direct (CI includes zero) |
| Arm 5 vs. Fixed-$k$ Bridge Raw $p$ | 0.0794 (0.079) | `TABLE_matched_depth_k4_bootstrap.csv` | 21 | `p_value` | 0.0794 | Direct ($p = 0.079$) |
| Arm 5 vs. Fixed-$k$ Bridge Holm $p$ | 0.0794 (0.079) | `TABLE_matched_depth_k4_bootstrap.csv` | 21 | `p_holm` | 0.0794 | Direct (Holm $p = 0.079$) |
| Arm 5 vs. Iter $k=4$ Bridge $\Delta$ | $-0.0257$ (2.6 points) | `TABLE_matched_depth_k4_bootstrap.csv` | 20 | `delta_recall_k4` | -0.0256806188821987 | $\Delta \times -100 = 2.6$ points |
| Arm 5 vs. Iter $k=4$ Bridge 95% CI Lo | $-0.0328$ | `TABLE_matched_depth_k4_bootstrap.csv` | 20 | `ci_lo` | -0.0328043384541005 | Direct |
| Arm 5 vs. Iter $k=4$ Bridge 95% CI Hi | $-0.0186$ | `TABLE_matched_depth_k4_bootstrap.csv` | 20 | `ci_hi` | -0.0186403069808110 | Direct |
| Arm 5 vs. Iter $k=4$ Bridge Holm $p$ | $<0.001$ | `TABLE_matched_depth_k4_bootstrap.csv` | 20 | `p_holm` | 0.0005 | Reported as $p < 0.001$ |
| Arm 5 vs. Arm 4b Bridge $\Delta$ | $+0.0177$ (+1.8 points) | `TABLE_matched_depth_k4_bootstrap.csv` | 23 | `delta_recall_k4` | 0.0177413840065832 | $\Delta \times 100 = +1.8$ points |
| Arm 5 vs. Arm 4b Bridge 95% CI Lo | $+0.0084$ | `TABLE_matched_depth_k4_bootstrap.csv` | 23 | `ci_lo` | 0.0084259949075610 | Direct |
| Arm 5 vs. Arm 4b Bridge 95% CI Hi | $+0.0269$ | `TABLE_matched_depth_k4_bootstrap.csv` | 23 | `ci_hi` | 0.0268862308917219 | Direct |
| Arm 5 vs. Arm 4b Bridge Holm $p$ | 0.001 (0.0006) | `TABLE_matched_depth_k4_bootstrap.csv` | 23 | `p_holm` | 0.0006 | Reported as $p = 0.001$ |
| Iter $k=4$ vs. Fixed-$k$ Bridge $\Delta$ | $+0.0169$ (+1.7 points) | `TABLE_matched_depth_k4_bootstrap.csv` | 24 | `delta_recall_k4` | 0.0168512277323061 | $\Delta \times 100 = +1.7$ points |
| Iter $k=4$ vs. Fixed-$k$ Bridge 95% CI Lo | $+0.0077$ | `TABLE_matched_depth_k4_bootstrap.csv` | 24 | `ci_lo` | 0.0077034183919114 | Direct |
| Iter $k=4$ vs. Fixed-$k$ Bridge 95% CI Hi | $+0.0258$ | `TABLE_matched_depth_k4_bootstrap.csv` | 24 | `ci_hi` | 0.0257643235435724 | Direct |
| Iter $k=4$ vs. Fixed-$k$ Bridge Holm $p$ | 0.001 | `TABLE_matched_depth_k4_bootstrap.csv` | 24 | `p_holm` | 0.0008 | Reported as $0.001$ |
| Iter $k=4$ vs. Fixed-$k$ Comp $\Delta$ | $-0.0240$ (-2.4 points) | `TABLE_matched_depth_k4_bootstrap.csv` | 29 | `delta_recall_k4` | -0.0239786856127887 | $\Delta \times 100 = -2.4$ points |
| Iter $k=4$ vs. Fixed-$k$ Comp 95% CI Lo | $-0.0391$ | `TABLE_matched_depth_k4_bootstrap.csv` | 29 | `ci_lo` | -0.039076376554174 | Direct |
| Iter $k=4$ vs. Fixed-$k$ Comp 95% CI Hi | $-0.0089$ | `TABLE_matched_depth_k4_bootstrap.csv` | 29 | `ci_hi` | -0.0088809946714031 | Direct |
| Iter $k=4$ vs. Fixed-$k$ Comp Holm $p$ | 0.007 | `TABLE_matched_depth_k4_bootstrap.csv` | 29 | `p_holm` | 0.0072 | Reported as $0.007$ |
| Arm 6 vs. Fixed-$k$ Comp $k=2.0$ $\Delta$ | $-0.138$ ($-0.1376$) | `TABLE_multi_depth_comparison.csv` | 16 | `obs_delta` | -0.1375585338285161 | Direct |
| Arm 6 vs. Fixed-$k$ Comp $k=3.5$ $\Delta$ | $-0.044$ ($-0.0441$) | `TABLE_multi_depth_comparison.csv` | 19 | `obs_delta` | -0.044083973549616 | Direct |
| Arm 6 vs. Arm 5 Bridge $k=3.5$ $\Delta$ | $+1.5$ points ($+0.0145$) | `TABLE_multi_depth_bridge.csv` | 27 | `obs_delta` | 0.0145078987798803 | $\Delta \times 100 = +1.5$ points |
| Arm 6 vs. Arm 5 Bridge $k=2.0$ $\Delta$ | $+5.3$ points ($+0.0525$) | `TABLE_multi_depth_bridge.csv` | 24 | `obs_delta` | 0.0525111735179338 | $\Delta \times 100 = +5.3$ points |
| Arm 6 vs. Arm 5 Comp $k=3.5$ $\Delta$ | $+3.4$ points ($+0.0344$) | `TABLE_multi_depth_comparison.csv` | 27 | `obs_delta` | 0.0343816327801111 | $\Delta \times 100 = +3.4$ points |
| Arm 6 vs. Arm 5 Comp $k=2.5$ $\Delta$ | $+5.5$ points ($+0.0547$) | `TABLE_multi_depth_comparison.csv` | 25 | `obs_delta` | 0.054676218248639 | $\Delta \times 100 = +5.5$ points |
| Arm 5 vs. Fixed-$k$ Bridge $k=5.5$ $\Delta$ | $+0.0127$ | `TABLE_multi_depth_bridge.csv` | 15 | `obs_delta` | 0.0126797820586939 | Direct |
| Arm 5 vs. Fixed-$k$ Bridge $k=5.5$ CI Lo | $+0.0044$ | `TABLE_multi_depth_bridge.csv` | 15 | `ci_lo` | 0.0044436686471266 | Direct |
| Arm 5 vs. Fixed-$k$ Bridge $k=5.5$ CI Hi | $+0.0210$ | `TABLE_multi_depth_bridge.csv` | 15 | `ci_hi` | 0.0210268139146172 | Direct |
| Arm 5 vs. Fixed-$k$ Bridge $k=5.5$ Holm $p$ | 0.0176 | `TABLE_multi_depth_bridge.csv` | 15 | `p_holm` | 0.0176 | Direct |
| Sufficiency Stopper AUC (5-fold CV) | 0.7977 (0.80) | `TABLE_sufficiency_auc.csv` | 0 | `auc` | 0.7976890641768017 | Direct |
| Sufficiency Stopper AUC (Held-out Eval) | 0.7964 (0.80) | `TABLE_sufficiency_auc.csv` | 1 | `auc` | 0.7964322394201111 | Direct |

---

## 10. Automated Number Audit Smoke Test Summary
The paper text (`ECA-RAG_final.tex`) was audited using [`number_audit.py`](file:///Users/jaiyandh/Downloads/EcaRag/number_audit.py) as a smoke test against the CSV data store:
- **Total extracted numbers:** 1,197
- **Sourced directly from CSVs:** 1,090
- **Experimental study design constants:** 106
- **Derived mathematical reductions:** 1
- **Unmatched / unsourced numbers:** **0**




#!/usr/bin/env python3
"""
Preprocesses ECA-RAG_final.tex into docx-build.tex for Pandoc compilation.
Rules:
1. Replace all \\ref and \\cref with literal text (Table 1..5, Fig. 1, Section IV-G).
2. Expand \\multirow and \\multicolumn into explicit cells with cohort labels repeated.
   Table 5 comparison block must say 'Comparison (N=563)' explicitly on every row.
3. Replace \\toprule, \\midrule, \\bottomrule, \\cmidrule, \\cline with \\hline.
4. Replace IEEE author block with plain text containing [email] and [confirm department].
5. Point \\includegraphics to figures/fig_pareto_recall_vs_k.png (300 dpi).
6. Convert IEEEtran documentclass to article for seamless Pandoc processing.
"""

import re
import os

def generate_docx_build_tex():
    with open("ECA-RAG_final.tex", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Document class and preamble simplification for Pandoc
    # Replace documentclass
    content = re.sub(r"\\documentclass\[conference\]\{IEEEtran\}", r"\\documentclass{article}", content)

    # Replace packages / IEEE definitions
    content = content.replace(r"\usepackage{cite}", "% cite")
    content = content.replace(r"\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em", "% bibtex")
    content = content.replace(r"    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}", "")

    # Replace author block
    author_block_target = r"""\author{
    \IEEEauthorblockN{Jaiyandh Amutha Sugumar\IEEEauthorrefmark{1}, Adwaith Murali\IEEEauthorrefmark{2}, Ariya Kumar\IEEEauthorrefmark{3}}
    \IEEEauthorblockA{
        \textit{Department of Computer Science and Engineering}\\
        \textit{Kongu Engineering College}\\
        Perundurai, Erode, Tamil Nadu, India\\
        Email: \IEEEauthorrefmark{1}jaiyandh@gmail.com, \IEEEauthorrefmark{2}adwaithmurali2005@gmail.com, \IEEEauthorrefmark{3}ariyakumar258@gmail.com
    }
}"""
    author_block_replacement = r"""\author{Jaiyandh Amutha Sugumar, Adwaith Murali, Ariya Kumar\\
\textit{Department of Computer Science and Engineering}\\
\textit{Kongu Engineering College}, Perundurai, Erode, Tamil Nadu, India\\
Email: jaiyandh@gmail.com, adwaithmurali2005@gmail.com, ariyakumar258@gmail.com}"""
    content = content.replace(author_block_target, author_block_replacement)

    # 2. Figure replacement: pdf -> png
    content = content.replace("figures/fig_pareto_recall_vs_k.pdf", "figures/fig_pareto_recall_vs_k.png")

    # 3. Reference replacements
    # Check all references
    content = content.replace(r"Section~\ref{sec:results-ablation}", "Section IV-G")
    content = content.replace(r"\ref{sec:results-ablation}", "Section IV-G")
    content = content.replace(r"Table~\ref{tab:calibration}", "Table 1")
    content = content.replace(r"\ref{tab:calibration}", "Table 1")
    content = content.replace(r"Table~\ref{tab:matched_depth}", "Table 2")
    content = content.replace(r"\ref{tab:matched_depth}", "Table 2")
    content = content.replace(r"Table~\ref{tab:full_eval}", "Table 3")
    content = content.replace(r"\ref{tab:full_eval}", "Table 3")
    content = content.replace(r"Table~\ref{tab:significance}", "Table 4")
    content = content.replace(r"\ref{tab:significance}", "Table 4")
    content = content.replace(r"Table~\ref{tab:multi_depth}", "Table 5")
    content = content.replace(r"\ref{tab:multi_depth}", "Table 5")
    content = content.replace(r"Table~V", "Table 5")
    content = content.replace(r"Table~IV", "Table 4")
    content = content.replace(r"Table~III", "Table 3")
    content = content.replace(r"Table~II", "Table 2")
    content = content.replace(r"Table~I", "Table 1")
    content = content.replace(r"Figure~\ref{fig:pareto}", "Fig. 1")
    content = content.replace(r"Fig.~\ref{fig:pareto}", "Fig. 1")
    content = content.replace(r"\ref{fig:pareto}", "Fig. 1")

    # Change table and figure captions to Table 1..5 and Fig. 1 if needed
    content = content.replace(r"\caption{Cross-Encoder Calibration Quality", r"\caption{\textbf{Table 1:} Cross-Encoder Calibration Quality")
    content = content.replace(r"\caption{Matched-Depth Paragraph Recall", r"\caption{\textbf{Table 2:} Matched-Depth Paragraph Recall")
    content = content.replace(r"\caption{Full Retrieval Performance Across", r"\caption{\textbf{Table 3:} Full Retrieval Performance Across")
    content = content.replace(r"\caption{Paired Bootstrap Hypothesis Testing", r"\caption{\textbf{Table 4:} Paired Bootstrap Hypothesis Testing")
    content = content.replace(r"\caption{Exploratory Multi-Depth Matched-Depth Analysis", r"\caption{\textbf{Table 5:} Exploratory Multi-Depth Matched-Depth Analysis")
    content = content.replace(r"\caption{Recall--efficiency Pareto frontiers", r"\caption{\textbf{Fig. 1:} Recall--efficiency Pareto frontiers")

    # 4. Table 1 reconstruction
    table1_target = r"""\begin{tabular}{lccc}
\toprule
\textbf{Calibration Method} & \textbf{ECE} $\downarrow$ & \textbf{Brier Score} $\downarrow$ & \textbf{ECE Red.} \\
\midrule
Raw Sigmoid $\sigma(s)$ & 0.2248 & 0.2127 & Baseline \\
Platt Scaling (Pooled)  & 0.0419 & 0.1161 & 81.4\% \\
Platt Scaling (Per-Type) & \textbf{0.0391} & \textbf{0.1157} & \textbf{82.6\%} \\
\bottomrule
\end{tabular}"""

    table1_replacement = r"""\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Calibration Method} & \textbf{ECE} $\downarrow$ & \textbf{Brier Score} $\downarrow$ & \textbf{ECE Red.} \\
\hline
Raw Sigmoid $\sigma(s)$ & 0.2248 & 0.2127 & Baseline \\
\hline
Platt Scaling (Pooled)  & 0.0419 & 0.1161 & 81.4\% \\
\hline
Platt Scaling (Per-Type) & \textbf{0.0391} & \textbf{0.1157} & \textbf{82.6\%} \\
\hline
\end{tabular}"""
    content = content.replace(table1_target, table1_replacement)

    # 5. Table 2 reconstruction
    table2_target = r"""\begin{tabular}{llccccc}
\toprule
\textbf{Split / Dataset} & \textbf{Type ($N$)} & \textbf{Fixed-$k$ ($k=4$)} & \textbf{Iterative ($k=4$, No Gate)} & \textbf{Arm 5 (Interp. $k=4$)} & \textbf{Arm 4b (Interp. $k=4$)} & \textbf{Arm 6 (Interp. $k=4$)} \\
\midrule
\multirow{2}{*}{$N=540$ Component Split} 
 & Bridge (424)     & 0.8054 & 0.8160 & 0.8013 & 0.7693 & n/a \\
 & Comparison (116) & \textbf{0.9483} & 0.9181 & 0.8649 & 0.9479 & 0.8712$^*$ \\
\midrule
\multirow{2}{*}{$N=2100$ Component Split} 
 & Bridge (1653)     & 0.8076 & \textbf{0.8261} & 0.7978 & 0.7835 & n/a \\
 & Comparison (447)  & \textbf{0.9553} & 0.9329 & 0.9273 & 0.9530 & 0.9280$^*$ \\
\midrule
\multirow{2}{*}{\textbf{Pooled Headline ($N=2640$)}} 
 & \textbf{Bridge (2077)}     & 0.8072 & \textbf{0.8240} & 0.7983 & 0.7806 & n/a \\
 & \textbf{Comparison (563)} & \textbf{0.9538} & 0.9298 & 0.9147 & 0.9520 & 0.9168$^*$ \\
\bottomrule
\end{tabular}"""

    table2_replacement = r"""\begin{tabular}{|l|l|c|c|c|c|c|}
\hline
\textbf{Split / Dataset} & \textbf{Type ($N$)} & \textbf{Fixed-$k$ ($k=4$)} & \textbf{Iterative ($k=4$, No Gate)} & \textbf{Arm 5 (Interp. $k=4$)} & \textbf{Arm 4b (Interp. $k=4$)} & \textbf{Arm 6 (Interp. $k=4$)} \\
\hline
$N=540$ Component Split & Bridge (424) & 0.8054 & 0.8160 & 0.8013 & 0.7693 & n/a \\
\hline
$N=540$ Component Split & Comparison (116) & \textbf{0.9483} & 0.9181 & 0.8649 & 0.9479 & 0.8712$^*$ \\
\hline
$N=2100$ Component Split & Bridge (1653) & 0.8076 & \textbf{0.8261} & 0.7978 & 0.7835 & n/a \\
\hline
$N=2100$ Component Split & Comparison (447) & \textbf{0.9553} & 0.9329 & 0.9273 & 0.9530 & 0.9280$^*$ \\
\hline
\textbf{Pooled Headline ($N=2640$)} & \textbf{Bridge (2077)} & 0.8072 & \textbf{0.8240} & 0.7983 & 0.7806 & n/a \\
\hline
\textbf{Pooled Headline ($N=2640$)} & \textbf{Comparison (563)} & \textbf{0.9538} & 0.9298 & 0.9147 & 0.9520 & 0.9168$^*$ \\
\hline
\end{tabular}"""
    content = content.replace(table2_target, table2_replacement)

    # 6. Table 3 reconstruction
    table3_target = r"""\begin{tabular}{llcccccccc}
\toprule
& & \multicolumn{4}{c}{\textbf{Initial Eval Split ($N=540$)}} & \multicolumn{4}{c}{\textbf{Extension Split ($N=2100$)}} \\
\cline{3-6} \cline{7-10}
\textbf{Method} & \textbf{Type} & \textbf{Recall} & \textbf{Precision} & \textbf{Full-Match} & \textbf{Mean $k$} & \textbf{Recall} & \textbf{Precision} & \textbf{Full-Match} & \textbf{Mean $k$} \\
\midrule
\multirow{2}{*}{Arm 1 (Dense Baseline $k=4$)}
 & Bridge     & 0.7559 & 0.3779 & 0.5354 & 4.000 & 0.7429 & 0.3728 & 0.5275 & 4.000 \\
 & Comparison & 0.9138 & 0.4569 & 0.8362 & 4.000 & 0.9306 & 0.4691 & 0.8613 & 4.000 \\
\midrule
\multirow{2}{*}{Arm 2 (Fixed-$k$ CE, $k=4$)}
 & Bridge     & 0.8054 & 0.4027 & 0.6132 & 4.000 & 0.8076 & 0.4051 & 0.6261 & 3.995 \\
 & Comparison & 0.9483 & 0.4741 & 0.8966 & 4.000 & 0.9553 & 0.4814 & 0.9128 & 3.984 \\
\midrule
\multirow{2}{*}{Arm 2 (Fixed-$k$ CE, $k=5$)}
 & Bridge     & 0.8314 & 0.3328 & 0.6627 & 4.998 & 0.8445 & 0.3394 & 0.6975 & 4.992 \\
 & Comparison & 0.9698 & 0.3879 & 0.9397 & 5.000 & 0.9720 & 0.3939 & 0.9441 & 4.971 \\
\midrule
\multirow{2}{*}{Arm 3 (Uncalib. Gating, $s \ge -1.925$)}
 & Bridge     & 0.7818 & 0.4895 & 0.5684 & 4.158 & --- & --- & --- & --- \\
 & Comparison & 0.9612 & 0.4983 & 0.9224 & 4.534 & --- & --- & --- & --- \\
\midrule
\multirow{2}{*}{Arm 4a (Calib. Gating, $\theta^*=0.18$)}
 & Bridge     & 0.7394 & 0.5670 & 0.4858 & 3.535 & --- & --- & --- & --- \\
 & Comparison & 0.9353 & 0.6046 & 0.8793 & 3.845 & --- & --- & --- & --- \\
\midrule
\multirow{2}{*}{Arm 4b (Calib. Gating, $\theta^*=0.18$)}
 & Bridge     & 0.7417 & 0.5581 & 0.4906 & 3.594 & 0.7577 & 0.5659 & 0.5227 & 3.598 \\
 & Comparison & 0.9224 & 0.6345 & 0.8534 & 3.629 & 0.9128 & 0.6673 & 0.8277 & 3.398 \\
\midrule
\multirow{2}{*}{Arm 4b (Sensitivity, $\theta=0.15$)}
 & Bridge     & 0.7677 & 0.5113 & 0.5401 & 3.976 & 0.7834 & 0.5148 & 0.5729 & 3.998 \\
 & Comparison & 0.9353 & 0.6046 & 0.8793 & 3.845 & 0.9295 & 0.6358 & 0.8613 & 3.640 \\
\midrule
\multirow{2}{*}{Arm 5 (Iter. Gated, $\theta^*=0.18$)}
 & Bridge     & 0.7983 & 0.5107 & 0.6108 & 3.934 & 0.7895 & 0.5204 & 0.5935 & 3.850 \\
 & Comparison & 0.8793 & 0.4947 & 0.7759 & 4.198 & 0.9295 & 0.5559 & 0.8591 & 4.038 \\
\midrule
\multirow{2}{*}{Arm 5 (Sensitivity, $\theta=0.15$)}
 & Bridge     & 0.8243 & 0.4523 & 0.6604 & 4.455 & 0.8173 & 0.4582 & 0.6461 & 4.399 \\
 & Comparison & 0.9138 & 0.4701 & 0.8362 & 4.560 & 0.9407 & 0.5333 & 0.8814 & 4.237 \\
\midrule
\multirow{2}{*}{Iterative Ungated ($k=4$)}
 & Bridge     & 0.8160 & 0.4080 & 0.6462 & 4.000 & 0.8261 & 0.4143 & 0.6673 & 3.995 \\
 & Comparison & 0.9181 & 0.4591 & 0.8534 & 4.000 & 0.9329 & 0.4702 & 0.8658 & 3.984 \\
\midrule
\multirow{2}{*}{Iterative Ungated ($k=5$)}
 & Bridge     & 0.8514 & 0.3408 & 0.7099 & 4.998 & 0.8542 & 0.3433 & 0.7181 & 4.992 \\
 & Comparison & 0.9526 & 0.3810 & 0.9052 & 5.000 & 0.9620 & 0.3899 & 0.9239 & 4.971 \\
\midrule
\multirow{2}{*}{Arm 6 (Sufficiency Stopper, $\tau^*=0.95$)}
 & Bridge     & 0.7983 & 0.5107 & 0.6108 & 3.934 & 0.7895 & 0.5204 & 0.5935 & 3.850 \\
 & Comparison & 0.8793 & 0.4947 & 0.7759 & 4.198 & 0.9295 & 0.5559 & 0.8591 & 4.038 \\
\midrule
\multirow{2}{*}{Oracle Hybrid Upper Bound}
 & Bridge     & 0.8160 & 0.4080 & 0.6462 & 4.000 & 0.8261 & 0.4143 & 0.6673 & 3.995 \\
 & Comparison & 0.9483 & 0.4741 & 0.8966 & 4.000 & 0.9553 & 0.4814 & 0.9128 & 3.984 \\
\bottomrule
\end{tabular}"""

    table3_replacement = r"""\begin{tabular}{|l|l|c|c|c|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Type} & \textbf{Eval Recall} & \textbf{Eval Prec.} & \textbf{Eval Full-M.} & \textbf{Eval $k$} & \textbf{Ext. Recall} & \textbf{Ext. Prec.} & \textbf{Ext. Full-M.} & \textbf{Ext. $k$} \\
\hline
Arm 1 (Dense Baseline $k=4$) & Bridge & 0.7559 & 0.3779 & 0.5354 & 4.000 & 0.7429 & 0.3728 & 0.5275 & 4.000 \\
\hline
Arm 1 (Dense Baseline $k=4$) & Comparison & 0.9138 & 0.4569 & 0.8362 & 4.000 & 0.9306 & 0.4691 & 0.8613 & 4.000 \\
\hline
Arm 2 (Fixed-$k$ CE, $k=4$) & Bridge & 0.8054 & 0.4027 & 0.6132 & 4.000 & 0.8076 & 0.4051 & 0.6261 & 3.995 \\
\hline
Arm 2 (Fixed-$k$ CE, $k=4$) & Comparison & 0.9483 & 0.4741 & 0.8966 & 4.000 & 0.9553 & 0.4814 & 0.9128 & 3.984 \\
\hline
Arm 2 (Fixed-$k$ CE, $k=5$) & Bridge & 0.8314 & 0.3328 & 0.6627 & 4.998 & 0.8445 & 0.3394 & 0.6975 & 4.992 \\
\hline
Arm 2 (Fixed-$k$ CE, $k=5$) & Comparison & 0.9698 & 0.3879 & 0.9397 & 5.000 & 0.9720 & 0.3939 & 0.9441 & 4.971 \\
\hline
Arm 3 (Uncalib. Gating, $s \ge -1.925$) & Bridge & 0.7818 & 0.4895 & 0.5684 & 4.158 & --- & --- & --- & --- \\
\hline
Arm 3 (Uncalib. Gating, $s \ge -1.925$) & Comparison & 0.9612 & 0.4983 & 0.9224 & 4.534 & --- & --- & --- & --- \\
\hline
Arm 4a (Calib. Gating, $\theta^*=0.18$) & Bridge & 0.7394 & 0.5670 & 0.4858 & 3.535 & --- & --- & --- & --- \\
\hline
Arm 4a (Calib. Gating, $\theta^*=0.18$) & Comparison & 0.9353 & 0.6046 & 0.8793 & 3.845 & --- & --- & --- & --- \\
\hline
Arm 4b (Calib. Gating, $\theta^*=0.18$) & Bridge & 0.7417 & 0.5581 & 0.4906 & 3.594 & 0.7577 & 0.5659 & 0.5227 & 3.598 \\
\hline
Arm 4b (Calib. Gating, $\theta^*=0.18$) & Comparison & 0.9224 & 0.6345 & 0.8534 & 3.629 & 0.9128 & 0.6673 & 0.8277 & 3.398 \\
\hline
Arm 4b (Sensitivity, $\theta=0.15$) & Bridge & 0.7677 & 0.5113 & 0.5401 & 3.976 & 0.7834 & 0.5148 & 0.5729 & 3.998 \\
\hline
Arm 4b (Sensitivity, $\theta=0.15$) & Comparison & 0.9353 & 0.6046 & 0.8793 & 3.845 & 0.9295 & 0.6358 & 0.8613 & 3.640 \\
\hline
Arm 5 (Iter. Gated, $\theta^*=0.18$) & Bridge & 0.7983 & 0.5107 & 0.6108 & 3.934 & 0.7895 & 0.5204 & 0.5935 & 3.850 \\
\hline
Arm 5 (Iter. Gated, $\theta^*=0.18$) & Comparison & 0.8793 & 0.4947 & 0.7759 & 4.198 & 0.9295 & 0.5559 & 0.8591 & 4.038 \\
\hline
Arm 5 (Sensitivity, $\theta=0.15$) & Bridge & 0.8243 & 0.4523 & 0.6604 & 4.455 & 0.8173 & 0.4582 & 0.6461 & 4.399 \\
\hline
Arm 5 (Sensitivity, $\theta=0.15$) & Comparison & 0.9138 & 0.4701 & 0.8362 & 4.560 & 0.9407 & 0.5333 & 0.8814 & 4.237 \\
\hline
Iterative Ungated ($k=4$) & Bridge & 0.8160 & 0.4080 & 0.6462 & 4.000 & 0.8261 & 0.4143 & 0.6673 & 3.995 \\
\hline
Iterative Ungated ($k=4$) & Comparison & 0.9181 & 0.4591 & 0.8534 & 4.000 & 0.9329 & 0.4702 & 0.8658 & 3.984 \\
\hline
Iterative Ungated ($k=5$) & Bridge & 0.8514 & 0.3408 & 0.7099 & 4.998 & 0.8542 & 0.3433 & 0.7181 & 4.992 \\
\hline
Iterative Ungated ($k=5$) & Comparison & 0.9526 & 0.3810 & 0.9052 & 5.000 & 0.9620 & 0.3899 & 0.9239 & 4.971 \\
\hline
Arm 6 (Sufficiency Stopper, $\tau^*=0.95$) & Bridge & 0.7983 & 0.5107 & 0.6108 & 3.934 & 0.7895 & 0.5204 & 0.5935 & 3.850 \\
\hline
Arm 6 (Sufficiency Stopper, $\tau^*=0.95$) & Comparison & 0.8793 & 0.4947 & 0.7759 & 4.198 & 0.9295 & 0.5559 & 0.8591 & 4.038 \\
\hline
Oracle Hybrid Upper Bound & Bridge & 0.8160 & 0.4080 & 0.6462 & 4.000 & 0.8261 & 0.4143 & 0.6673 & 3.995 \\
\hline
Oracle Hybrid Upper Bound & Comparison & 0.9483 & 0.4741 & 0.8966 & 4.000 & 0.9553 & 0.4814 & 0.9128 & 3.984 \\
\hline
\end{tabular}"""
    content = content.replace(table3_target, table3_replacement)

    # 7. Table 4 reconstruction
    table4_target = r"""\begin{tabular}{lllccccc}
\toprule
\textbf{Cohort} & \textbf{Hypothesis Comparator} & \textbf{Contrast Description} & $\Delta$ \textbf{Recall} & \textbf{95\% Bootstrap CI} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{Verdict} \\
\midrule
\multirow{5}{*}{Pooled Bridge ($N=2077$)}
 & Iter $k=4$ vs. Arm 2 ($k=4$) & Hop-conditioning alone at exact $k=4.00$ & $+0.0169$ & $[+0.0077, +0.0258]$ & $<0.001$ & $0.001$ & Significant Gain \\
 & Arm 5 (interp) vs. Iter $k=4$ & Recall cost of calibrated gating & $-0.0257$ & $[-0.0328, -0.0186]$ & $<0.001$ & $<0.001$ & Significant Cost \\
 & Arm 5 (interp) vs. Arm 2 ($k=4$) & Iterative gated vs. fixed-$k$ at $k=4.00$ & $-0.0088$ & $[-0.0187, +0.0010]$ & $0.079$ & $0.079$ & No Detectable Diff. (CI includes 0) \\
 & Arm 4b (interp) vs. Arm 2 ($k=4$) & Single-pass gating vs. fixed-$k$ at $k=4.00$ & $-0.0266$ & $[-0.0341, -0.0191]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & Arm 5 (interp) vs. Arm 4b (interp) & Iterative vs. single-pass gating & $+0.0177$ & $[+0.0084, +0.0269]$ & $<0.001$ & $0.001$ & Significant Gain \\
\midrule
\multirow{3}{*}{Pooled Comp ($N=563$)}
 & Iter $k=4$ vs. Arm 2 ($k=4$) & Hop-conditioning on comparison & $-0.0240$ & $[-0.0391, -0.0089]$ & $0.002$ & $0.007$ & Significant Loss \\
 & Arm 4b (interp) vs. Arm 2 ($k=4$) & Single-pass gating on comparison & $-0.0018$ & $[-0.0136, +0.0095]$ & $0.720$ & $0.720$ & Indistinguishable \\
 & Arm 5 (interp) vs. Arm 2 ($k=4$) & Iterative gating on comparison & $-0.0391$ & $[-0.0565, -0.0221]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\bottomrule
\end{tabular}"""

    table4_replacement = r"""\begin{tabular}{|l|l|l|c|c|c|c|l|}
\hline
\textbf{Cohort} & \textbf{Hypothesis Comparator} & \textbf{Contrast Description} & $\Delta$ \textbf{Recall} & \textbf{95\% Bootstrap CI} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{Verdict} \\
\hline
Pooled Bridge ($N=2077$) & Iter $k=4$ vs. Arm 2 ($k=4$) & Hop-conditioning alone at exact $k=4.00$ & $+0.0169$ & $[+0.0077, +0.0258]$ & $<0.001$ & $0.001$ & Significant Gain \\
\hline
Pooled Bridge ($N=2077$) & Arm 5 (interp) vs. Iter $k=4$ & Recall cost of calibrated gating & $-0.0257$ & $[-0.0328, -0.0186]$ & $<0.001$ & $<0.001$ & Significant Cost \\
\hline
Pooled Bridge ($N=2077$) & Arm 5 (interp) vs. Arm 2 ($k=4$) & Iterative gated vs. fixed-$k$ at $k=4.00$ & $-0.0088$ & $[-0.0187, +0.0010]$ & $0.079$ & $0.079$ & No Detectable Diff. (CI includes 0) \\
\hline
Pooled Bridge ($N=2077$) & Arm 4b (interp) vs. Arm 2 ($k=4$) & Single-pass gating vs. fixed-$k$ at $k=4.00$ & $-0.0266$ & $[-0.0341, -0.0191]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Pooled Bridge ($N=2077$) & Arm 5 (interp) vs. Arm 4b (interp) & Iterative vs. single-pass gating & $+0.0177$ & $[+0.0084, +0.0269]$ & $<0.001$ & $0.001$ & Significant Gain \\
\hline
Pooled Comp ($N=563$) & Iter $k=4$ vs. Arm 2 ($k=4$) & Hop-conditioning on comparison & $-0.0240$ & $[-0.0391, -0.0089]$ & $0.002$ & $0.007$ & Significant Loss \\
\hline
Pooled Comp ($N=563$) & Arm 4b (interp) vs. Arm 2 ($k=4$) & Single-pass gating on comparison & $-0.0018$ & $[-0.0136, +0.0095]$ & $0.720$ & $0.720$ & Indistinguishable \\
\hline
Pooled Comp ($N=563$) & Arm 5 (interp) vs. Arm 2 ($k=4$) & Iterative gating on comparison & $-0.0391$ & $[-0.0565, -0.0221]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
\end{tabular}"""
    content = content.replace(table4_target, table4_replacement)

    # 8. Table 5 reconstruction
    table5_target = r"""\begin{tabular}{lllcccccc}
\toprule
\textbf{Cohort} & \textbf{Comparison} & \textbf{Target Depth $k$} & \textbf{Covered} & $\Delta$ \textbf{Recall} & \textbf{95\% Bootstrap CI} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{Verdict} \\
\midrule
Bridge ($N=2077$) & Arm 6 vs. Fixed-$k$
   & $k=2.0$ & Yes & $+0.0003$ & $[-0.0100, +0.0105]$ & $0.9524$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
 & & $k=2.5$ & Yes & $+0.0088$ & $[-0.0008, +0.0183]$ & $0.0720$ & $0.2880$ & No Detectable Diff. (CI includes 0) \\
 & & $k=3.0$ & Yes & $-0.0000$ & $[-0.0099, +0.0102]$ & $0.9866$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
 & & $k=3.5$ & Yes & $-0.0033$ & $[-0.0132, +0.0067]$ & $0.5170$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
\midrule
Bridge ($N=2077$) & Arm 6 vs. Arm 5
   & $k=2.0$ & Yes & $+0.0525$ & $[+0.0453, +0.0596]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
 & & $k=2.5$ & Yes & $+0.0464$ & $[+0.0390, +0.0538]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
 & & $k=3.0$ & Yes & $+0.0338$ & $[+0.0272, +0.0402]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
 & & $k=3.5$ & Yes & $+0.0145$ & $[+0.0095, +0.0198]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\midrule
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$
   & $k=2.0$ & Yes & $-0.0522$ & $[-0.0626, -0.0418]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=2.5$ & Yes & $-0.0376$ & $[-0.0474, -0.0276]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=3.0$ & Yes & $-0.0338$ & $[-0.0442, -0.0232]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=3.5$ & Yes & $-0.0178$ & $[-0.0276, -0.0079]$ & $0.0010$ & $0.0050$ & Significant Loss \\
 & & $k=4.0$ & Yes & $-0.0088$ & $[-0.0187, +0.0010]$ & $0.0794$ & $0.2382$ & No Detectable Diff. (CI includes 0) \\
 & & $k=4.5$ & Yes & $-0.0016$ & $[-0.0107, +0.0074]$ & $0.7344$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
 & & $k=5.0$ & Yes & $+0.0031$ & $[-0.0058, +0.0120]$ & $0.5100$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
 & & $k=5.5$ & Yes & $+0.0127$ & $[+0.0044, +0.0210]$ & $0.0044$ & $0.0176$ & Exploratory Gain \\
\midrule
Comparison ($N=563$) & Arm 6 vs. Fixed-$k$
   & $k=2.0$ & Yes & $-0.1376$ & $[-0.1576, -0.1175]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=2.5$ & Yes & $-0.0901$ & $[-0.1094, -0.0702]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=3.0$ & Yes & $-0.0659$ & $[-0.0861, -0.0453]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=3.5$ & Yes & $-0.0441$ & $[-0.0617, -0.0260]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\midrule
Comparison ($N=563$) & Arm 6 vs. Arm 5
   & $k=2.0$ & Yes & $+0.0410$ & $[+0.0301, +0.0521]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
 & & $k=2.5$ & Yes & $+0.0547$ & $[+0.0420, +0.0673]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
 & & $k=3.0$ & Yes & $+0.0539$ & $[+0.0413, +0.0669]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
 & & $k=3.5$ & Yes & $+0.0344$ & $[+0.0207, +0.0491]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\midrule
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$
   & $k=2.0$ & Yes & $-0.1786$ & $[-0.1978, -0.1592]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=2.5$ & Yes & $-0.1447$ & $[-0.1632, -0.1259]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=3.0$ & Yes & $-0.1198$ & $[-0.1397, -0.0992]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=3.5$ & Yes & $-0.0785$ & $[-0.0974, -0.0593]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=4.0$ & Yes & $-0.0391$ & $[-0.0565, -0.0221]$ & $<0.001$ & $<0.001$ & Significant Loss \\
 & & $k=4.5$ & Yes & $-0.0183$ & $[-0.0329, -0.0043]$ & $0.0110$ & $0.0330$ & Significant Loss \\
 & & $k=5.0$ & Yes & $-0.0058$ & $[-0.0181, +0.0060]$ & $0.3470$ & $0.6940$ & No Detectable Diff. (CI includes 0) \\
\bottomrule
\end{tabular}"""

    table5_replacement = r"""\begin{tabular}{|l|l|l|c|c|c|c|c|l|}
\hline
\textbf{Cohort} & \textbf{Comparison} & \textbf{Target Depth $k$} & \textbf{Covered} & $\Delta$ \textbf{Recall} & \textbf{95\% Bootstrap CI} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{Verdict} \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Fixed-$k$ & $k=2.0$ & Yes & $+0.0003$ & $[-0.0100, +0.0105]$ & $0.9524$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Fixed-$k$ & $k=2.5$ & Yes & $+0.0088$ & $[-0.0008, +0.0183]$ & $0.0720$ & $0.2880$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Fixed-$k$ & $k=3.0$ & Yes & $-0.0000$ & $[-0.0099, +0.0102]$ & $0.9866$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Fixed-$k$ & $k=3.5$ & Yes & $-0.0033$ & $[-0.0132, +0.0067]$ & $0.5170$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Arm 5 & $k=2.0$ & Yes & $+0.0525$ & $[+0.0453, +0.0596]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Arm 5 & $k=2.5$ & Yes & $+0.0464$ & $[+0.0390, +0.0538]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Arm 5 & $k=3.0$ & Yes & $+0.0338$ & $[+0.0272, +0.0402]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Bridge ($N=2077$) & Arm 6 vs. Arm 5 & $k=3.5$ & Yes & $+0.0145$ & $[+0.0095, +0.0198]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=2.0$ & Yes & $-0.0522$ & $[-0.0626, -0.0418]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=2.5$ & Yes & $-0.0376$ & $[-0.0474, -0.0276]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=3.0$ & Yes & $-0.0338$ & $[-0.0442, -0.0232]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=3.5$ & Yes & $-0.0178$ & $[-0.0276, -0.0079]$ & $0.0010$ & $0.0050$ & Significant Loss \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=4.0$ & Yes & $-0.0088$ & $[-0.0187, +0.0010]$ & $0.0794$ & $0.2382$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=4.5$ & Yes & $-0.0016$ & $[-0.0107, +0.0074]$ & $0.7344$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=5.0$ & Yes & $+0.0031$ & $[-0.0058, +0.0120]$ & $0.5100$ & $1.0000$ & No Detectable Diff. (CI includes 0) \\
\hline
Bridge ($N=2077$) & Arm 5 vs. Fixed-$k$ & $k=5.5$ & Yes & $+0.0127$ & $[+0.0044, +0.0210]$ & $0.0044$ & $0.0176$ & Exploratory Gain \\
\hline
Comparison ($N=563$) & Arm 6 vs. Fixed-$k$ & $k=2.0$ & Yes & $-0.1376$ & $[-0.1576, -0.1175]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 6 vs. Fixed-$k$ & $k=2.5$ & Yes & $-0.0901$ & $[-0.1094, -0.0702]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 6 vs. Fixed-$k$ & $k=3.0$ & Yes & $-0.0659$ & $[-0.0861, -0.0453]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 6 vs. Fixed-$k$ & $k=3.5$ & Yes & $-0.0441$ & $[-0.0617, -0.0260]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 6 vs. Arm 5 & $k=2.0$ & Yes & $+0.0410$ & $[+0.0301, +0.0521]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Comparison ($N=563$) & Arm 6 vs. Arm 5 & $k=2.5$ & Yes & $+0.0547$ & $[+0.0420, +0.0673]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Comparison ($N=563$) & Arm 6 vs. Arm 5 & $k=3.0$ & Yes & $+0.0539$ & $[+0.0413, +0.0669]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Comparison ($N=563$) & Arm 6 vs. Arm 5 & $k=3.5$ & Yes & $+0.0344$ & $[+0.0207, +0.0491]$ & $<0.001$ & $<0.001$ & Significant Gain (Early Stopping) \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=2.0$ & Yes & $-0.1786$ & $[-0.1978, -0.1592]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=2.5$ & Yes & $-0.1447$ & $[-0.1632, -0.1259]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=3.0$ & Yes & $-0.1198$ & $[-0.1397, -0.0992]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=3.5$ & Yes & $-0.0785$ & $[-0.0974, -0.0593]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=4.0$ & Yes & $-0.0391$ & $[-0.0565, -0.0221]$ & $<0.001$ & $<0.001$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=4.5$ & Yes & $-0.0183$ & $[-0.0329, -0.0043]$ & $0.0110$ & $0.0330$ & Significant Loss \\
\hline
Comparison ($N=563$) & Arm 5 vs. Fixed-$k$ & $k=5.0$ & Yes & $-0.0058$ & $[-0.0181, +0.0060]$ & $0.3470$ & $0.6940$ & No Detectable Diff. (CI includes 0) \\
\hline
\end{tabular}"""
    content = content.replace(table5_target, table5_replacement)

    # 9. Clean up any remaining booktabs macros or table* environments
    content = content.replace(r"\begin{table*}", r"\begin{table}")
    content = content.replace(r"\end{table*}", r"\end{table}")
    content = content.replace(r"\begin{figure*}", r"\begin{figure}")
    content = content.replace(r"\end{figure*}", r"\end{figure}")
    content = content.replace(r"\toprule", r"\hline")
    content = content.replace(r"\midrule", r"\hline")
    content = content.replace(r"\bottomrule", r"\hline")

    with open("docx-build.tex", "w", encoding="utf-8") as f:
        f.write(content)

    print("docx-build.tex generated successfully.")

if __name__ == "__main__":
    generate_docx_build_tex()

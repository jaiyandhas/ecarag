#!/usr/bin/env python3
"""
Unified script for Matched-Depth (k=4.00) and Multi-Depth (k in [2.0, 5.5]) Bootstrap Analysis.
Recomputes both Table 4 and Table 5 with:
- One unified sweep grid (23 thetas, 24 taus)
- One interpolation routine (linear interpolation without extrapolation)
- B = 10,000 resamples
- Seed = 42
- Exact matching between k=4 cells in both tables.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.stats import holm

RESULTS_DIR = 'results_full'
FIGURES_DIR = 'figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_BOOT = 10000
SEED = 42
TARGET_DEPTHS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

def interpolate_curve(k_pts, r_pts, target_k, enforce_monotone=False):
    """Linearly interpolate r at target_k. If target_k is out of range, return NaN."""
    order = np.argsort(k_pts)
    ks = k_pts[order]
    rs = r_pts[order]
    if enforce_monotone:
        rs = np.maximum.accumulate(rs)
    if target_k < ks[0] or target_k > ks[-1]:
        return np.nan
    return float(np.interp(target_k, ks, rs))

def run_multi_depth(data):
    """Recomputes Table 5 and Pareto figures."""
    print("=" * 70)
    print("RUNNING MULTI-DEPTH ANALYSIS (TABLE 5 & FIGURES)")
    print("=" * 70)

    def analyze_subset(arr, subset_name):
        n = arr['r_fixed'].shape[0]
        obs_k_fixed = arr['k_fixed']
        obs_r_fixed = arr['r_fixed'].mean(axis=0)

        obs_k_4b = arr['k_4b'].mean(axis=0)
        obs_r_4b = arr['r_4b'].mean(axis=0)

        obs_k_5 = arr['k_5'].mean(axis=0)
        obs_r_5 = arr['r_5'].mean(axis=0)

        obs_k_6 = arr['k_6'].mean(axis=0)
        obs_r_6 = arr['r_6'].mean(axis=0)

        boot_r_fixed = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
        boot_r_4b = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
        boot_r_5 = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
        boot_r_6 = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)

        grid_k = np.linspace(1.5, 5.5, 81)
        band_r_fixed = np.full((N_BOOT, len(grid_k)), np.nan)
        band_r_4b = np.full((N_BOOT, len(grid_k)), np.nan)
        band_r_5 = np.full((N_BOOT, len(grid_k)), np.nan)
        band_r_6 = np.full((N_BOOT, len(grid_k)), np.nan)

        rng = np.random.default_rng(SEED)
        boot_indices = rng.integers(0, n, size=(N_BOOT, n))

        for b in range(N_BOOT):
            idx = boot_indices[b]
            b_r_fix = arr['r_fixed'][idx].mean(axis=0)
            b_k_4b = arr['k_4b'][idx].mean(axis=0)
            b_r_4b = arr['r_4b'][idx].mean(axis=0)
            b_k_5 = arr['k_5'][idx].mean(axis=0)
            b_r_5 = arr['r_5'][idx].mean(axis=0)
            b_k_6 = arr['k_6'][idx].mean(axis=0)
            b_r_6 = arr['r_6'][idx].mean(axis=0)

            for d_idx, td in enumerate(TARGET_DEPTHS):
                boot_r_fixed[b, d_idx] = interpolate_curve(obs_k_fixed, b_r_fix, td)
                boot_r_4b[b, d_idx] = interpolate_curve(b_k_4b, b_r_4b, td)
                boot_r_5[b, d_idx] = interpolate_curve(b_k_5, b_r_5, td)
                boot_r_6[b, d_idx] = interpolate_curve(b_k_6, b_r_6, td)

            for g_idx, gk in enumerate(grid_k):
                band_r_fixed[b, g_idx] = interpolate_curve(obs_k_fixed, b_r_fix, gk, enforce_monotone=True)
                band_r_4b[b, g_idx] = interpolate_curve(b_k_4b, b_r_4b, gk, enforce_monotone=True)
                band_r_5[b, g_idx] = interpolate_curve(b_k_5, b_r_5, gk, enforce_monotone=True)
                band_r_6[b, g_idx] = interpolate_curve(b_k_6, b_r_6, gk, enforce_monotone=True)

        # Point estimates
        point_estimates = []
        for d_idx, td in enumerate(TARGET_DEPTHS):
            r_fix = interpolate_curve(obs_k_fixed, obs_r_fixed, td)
            r_4b = interpolate_curve(obs_k_4b, obs_r_4b, td)
            r_5 = interpolate_curve(obs_k_5, obs_r_5, td)
            r_6 = interpolate_curve(obs_k_6, obs_r_6, td)
            point_estimates.append({
                'k': td,
                'Fixed-k': r_fix,
                'Arm 4b': r_4b,
                'Arm 5': r_5,
                'Arm 6': r_6,
            })
        df_pts = pd.DataFrame(point_estimates)

        comparisons = [
            ('Arm 4b vs Fixed-k', boot_r_4b, boot_r_fixed, 'Arm 4b', 'Fixed-k'),
            ('Arm 5 vs Fixed-k', boot_r_5, boot_r_fixed, 'Arm 5', 'Fixed-k'),
            ('Arm 6 vs Fixed-k', boot_r_6, boot_r_fixed, 'Arm 6', 'Fixed-k'),
            ('Arm 6 vs Arm 5', boot_r_6, boot_r_5, 'Arm 6', 'Arm 5'),
        ]

        comp_rows = []
        for comp_name, boot_a, boot_b, name_a, name_b in comparisons:
            depth_results = []
            for d_idx, td in enumerate(TARGET_DEPTHS):
                obs_a = df_pts.loc[df_pts['k'] == td, name_a].values[0]
                obs_b = df_pts.loc[df_pts['k'] == td, name_b].values[0]
                diffs = boot_a[:, d_idx] - boot_b[:, d_idx]
                valid = ~np.isnan(diffs)

                if np.isnan(obs_a) or np.isnan(obs_b) or (valid.mean() < 0.95):
                    depth_results.append({
                        'subset': subset_name,
                        'comparison': comp_name,
                        'depth_k': td,
                        'covered': False,
                        'obs_delta': np.nan,
                        'ci_lo': np.nan,
                        'ci_hi': np.nan,
                        'p_raw': np.nan,
                    })
                else:
                    diffs_valid = diffs[valid]
                    obs_diff = obs_a - obs_b
                    ci_lo, ci_hi = np.percentile(diffs_valid, [2.5, 97.5])
                    p_val = 2 * min(float((diffs_valid <= 0).mean()), float((diffs_valid >= 0).mean()))
                    p_val = min(1.0, max(p_val, 1.0 / len(diffs_valid)))
                    depth_results.append({
                        'subset': subset_name,
                        'comparison': comp_name,
                        'depth_k': td,
                        'covered': True,
                        'obs_delta': obs_diff,
                        'ci_lo': ci_lo,
                        'ci_hi': ci_hi,
                        'p_raw': p_val,
                    })

            # Holm adjustment across the 8-depth family per contrast
            covered_indices = [i for i, r in enumerate(depth_results) if r['covered']]
            p_raws = [depth_results[i]['p_raw'] for i in covered_indices]
            p_holms = holm(p_raws)

            for idx_c, orig_idx in enumerate(covered_indices):
                depth_results[orig_idx]['p_holm'] = p_holms[idx_c]
            for orig_idx in range(len(depth_results)):
                if not depth_results[orig_idx]['covered']:
                    depth_results[orig_idx]['p_holm'] = np.nan

            comp_rows.extend(depth_results)

        df_comp = pd.DataFrame(comp_rows)
        return {
            'df_pts': df_pts,
            'df_comp': df_comp,
            'grid_k': grid_k,
            'band_r_fixed': band_r_fixed,
            'band_r_4b': band_r_4b,
            'band_r_5': band_r_5,
            'band_r_6': band_r_6,
            'obs_k_fixed': obs_k_fixed,
            'obs_r_fixed': obs_r_fixed,
            'obs_k_4b': obs_k_4b,
            'obs_r_4b': obs_r_4b,
            'obs_k_5': obs_k_5,
            'obs_r_5': obs_r_5,
            'obs_k_6': obs_k_6,
            'obs_r_6': obs_r_6,
            'r_iter_k4': arr['r_iter_k4'].mean(),
            'r_iter_k5': arr['r_iter_k5'].mean(),
        }

    res_bridge = analyze_subset(data['bridge'], 'bridge')
    res_comp = analyze_subset(data['comp'], 'comparison')

    df_comp_all = pd.concat([res_bridge['df_comp'], res_comp['df_comp']], ignore_index=True)
    df_comp_all.to_csv(os.path.join(RESULTS_DIR, 'TABLE_multi_depth_significance.csv'), index=False)
    res_bridge['df_comp'].to_csv(os.path.join(RESULTS_DIR, 'TABLE_multi_depth_bridge.csv'), index=False)
    res_comp['df_comp'].to_csv(os.path.join(RESULTS_DIR, 'TABLE_multi_depth_comparison.csv'), index=False)

    df_pts_all = pd.concat([
        res_bridge['df_pts'].assign(subset='bridge'),
        res_comp['df_pts'].assign(subset='comparison')
    ], ignore_index=True)
    df_pts_all.to_csv(os.path.join(RESULTS_DIR, 'TABLE_multi_depth_point_estimates.csv'), index=False)

    # Plot Pareto curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for idx, (res, title) in enumerate([(res_bridge, 'Bridge Multi-Hop Questions (N=2077)'),
                                        (res_comp, 'Comparison Multi-Hop Questions (N=563)')]):
        ax = axes[idx]
        grid_k = res['grid_k']

        # 1. Fixed-k
        lo_fix = np.nanpercentile(res['band_r_fixed'], 2.5, axis=0)
        hi_fix = np.nanpercentile(res['band_r_fixed'], 97.5, axis=0)
        ax.fill_between(grid_k, lo_fix, hi_fix, color='#1f77b4', alpha=0.15)
        fix_mask = (grid_k >= 1.0) & (grid_k <= 6.0)
        fix_interp_r = np.interp(grid_k[fix_mask], res['obs_k_fixed'], res['obs_r_fixed'])
        ax.plot(grid_k[fix_mask], fix_interp_r, '-', color='#1f77b4', lw=2.2, label='Fixed-k (Arm 2 interpolated)')
        ax.scatter(res['obs_k_fixed'], res['obs_r_fixed'], color='#1f77b4', s=45, zorder=4, edgecolor='black', lw=0.8)

        # 2. Arm 4b
        lo_4b = np.nanpercentile(res['band_r_4b'], 2.5, axis=0)
        hi_4b = np.nanpercentile(res['band_r_4b'], 97.5, axis=0)
        valid_4b = ~np.isnan(lo_4b)
        ax.fill_between(grid_k[valid_4b], lo_4b[valid_4b], hi_4b[valid_4b], color='#ff7f0e', alpha=0.15)
        ord_4b = np.argsort(res['obs_k_4b'])
        ax.plot(res['obs_k_4b'][ord_4b], np.maximum.accumulate(res['obs_r_4b'][ord_4b]), 's--', color='#ff7f0e', lw=1.8, ms=4, label='Arm 4b (Calib gating sweep)')

        # 3. Arm 5
        lo_5 = np.nanpercentile(res['band_r_5'], 2.5, axis=0)
        hi_5 = np.nanpercentile(res['band_r_5'], 97.5, axis=0)
        valid_5 = ~np.isnan(lo_5)
        ax.fill_between(grid_k[valid_5], lo_5[valid_5], hi_5[valid_5], color='#2ca02c', alpha=0.15)
        ord_5 = np.argsort(res['obs_k_5'])
        ax.plot(res['obs_k_5'][ord_5], np.maximum.accumulate(res['obs_r_5'][ord_5]), '^-.', color='#2ca02c', lw=1.8, ms=4, label='Arm 5 (Iterative gated sweep)')

        # 4. Arm 6
        lo_6 = np.nanpercentile(res['band_r_6'], 2.5, axis=0)
        hi_6 = np.nanpercentile(res['band_r_6'], 97.5, axis=0)
        valid_6 = ~np.isnan(lo_6)
        ax.fill_between(grid_k[valid_6], lo_6[valid_6], hi_6[valid_6], color='#d62728', alpha=0.15)
        ord_6 = np.argsort(res['obs_k_6'])
        ax.plot(res['obs_k_6'][ord_6], np.maximum.accumulate(res['obs_r_6'][ord_6]), 'd:', color='#d62728', lw=1.8, ms=4, label='Arm 6 (Sufficiency sweep)')

        # 5. Iterative ungated points
        ax.scatter([4.0], [res['r_iter_k4']], marker='X', color='#9467bd', s=110, zorder=6, edgecolor='black', lw=1.0, label='Iterative ungated (k=4, 5)')
        ax.scatter([5.0], [res['r_iter_k5']], marker='X', color='#9467bd', s=110, zorder=6, edgecolor='black', lw=1.0)

        ax.axvline(x=4.0, color='gray', linestyle=':', lw=1.5, alpha=0.8, label='Reference Depth k=4.0')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Mean Retrieval Depth ($k$ chunks)', fontsize=12)
        ax.set_ylabel('Supporting Paragraph Recall', fontsize=12)
        ax.set_xlim(1.5, 5.5)
        ax.set_ylim(0.55, 0.92 if idx == 0 else 1.00)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.legend(fontsize=9, loc='lower right', framealpha=0.92)

    plt.tight_layout()
    plt.savefig('figures/fig_pareto_recall_vs_k.pdf', bbox_inches='tight')
    plt.savefig('figures/fig_pareto_recall_vs_k.png', bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_pareto_recall_vs_k.png'), bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_pareto_recall_vs_k.pdf'), bbox_inches='tight')
    plt.close()
    print("Multi-depth analysis and figures generated.")

def run_matched_depth_k4(data):
    """Recomputes Table 2 and Table 4."""
    print("\n" + "=" * 70)
    print("RUNNING MATCHED-DEPTH k=4.00 BOOTSTRAP ANALYSIS (TABLES 2 & 4)")
    print("=" * 70)

    subsets = [
        ('N=540 (Original Eval)', 'bridge', slice(0, 424)),
        ('N=540 (Original Eval)', 'comparison', slice(0, 116)),
        ('N=2100 (Extension)', 'bridge', slice(424, 2077)),
        ('N=2100 (Extension)', 'comparison', slice(116, 563)),
        ('Pooled (N=2640)', 'bridge', slice(0, 2077)),
        ('Pooled (N=2640)', 'comparison', slice(0, 563)),
    ]

    all_sig_rows = []
    point_rows = []

    for split_name, qtype, sl in subsets:
        arr = data['bridge'] if qtype == 'bridge' else data['comp']
        n = arr['r_fixed'][sl].shape[0]

        sub_r_fixed = arr['r_fixed'][sl, 3] # k=4
        sub_r_iter = arr['r_iter_k4'][sl]
        sub_k_4b = arr['k_4b'][sl]
        sub_r_4b = arr['r_4b'][sl]
        sub_k_5 = arr['k_5'][sl]
        sub_r_5 = arr['r_5'][sl]
        sub_k_6 = arr['k_6'][sl]
        sub_r_6 = arr['r_6'][sl]

        # Observed point estimates
        obs_r_fixed = float(np.mean(sub_r_fixed))
        obs_r_iter = float(np.mean(sub_r_iter))
        obs_r_4b = interpolate_curve(sub_k_4b.mean(axis=0), sub_r_4b.mean(axis=0), 4.0)
        obs_r_5 = interpolate_curve(sub_k_5.mean(axis=0), sub_r_5.mean(axis=0), 4.0)
        obs_r_6 = interpolate_curve(sub_k_6.mean(axis=0), sub_r_6.mean(axis=0), 4.0)

        point_rows.append({
            'split': split_name,
            'type': qtype,
            'n': n,
            'Fixed-k (k=4)': obs_r_fixed,
            'Iter k=4 (no gate)': obs_r_iter,
            'Arm 5 (interp k=4)': obs_r_5,
            'Arm 4b (interp k=4)': obs_r_4b,
            'Arm 6 (interp k=4)': obs_r_6,
        })

        rng = np.random.default_rng(SEED)
        boot_indices = rng.integers(0, n, size=(N_BOOT, n))

        boot_r_fixed = sub_r_fixed[boot_indices].mean(axis=1)
        boot_r_iter = sub_r_iter[boot_indices].mean(axis=1)

        boot_k_4b = np.take(sub_k_4b, boot_indices, axis=0).mean(axis=1)
        boot_r_4b = np.take(sub_r_4b, boot_indices, axis=0).mean(axis=1)
        boot_interp_4b = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            boot_interp_4b[b] = interpolate_curve(boot_k_4b[b], boot_r_4b[b], 4.0)

        boot_k_5 = np.take(sub_k_5, boot_indices, axis=0).mean(axis=1)
        boot_r_5 = np.take(sub_r_5, boot_indices, axis=0).mean(axis=1)
        boot_interp_5 = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            boot_interp_5[b] = interpolate_curve(boot_k_5[b], boot_r_5[b], 4.0)

        comps = [
            ('Arm 5 (interp k=4)', 'Iter k=4 (no gate)', obs_r_5 - obs_r_iter, boot_interp_5 - boot_r_iter, 'Arm 5 vs Iter k=4 (gating effect at k=4)'),
            ('Arm 5 (interp k=4)', 'Arm 2 (fixed k=4)', obs_r_5 - obs_r_fixed, boot_interp_5 - boot_r_fixed, 'Arm 5 vs Arm 2 (iter-gated vs fixed-k at k=4)'),
            ('Arm 4b (interp k=4)', 'Arm 2 (fixed k=4)', obs_r_4b - obs_r_fixed, boot_interp_4b - boot_r_fixed, 'Arm 4b vs Arm 2 (single-pass calib vs fixed-k at k=4)'),
            ('Arm 5 (interp k=4)', 'Arm 4b (interp k=4)', obs_r_5 - obs_r_4b, boot_interp_5 - boot_interp_4b, 'Arm 5 vs Arm 4b (iterative vs single-pass gating at k=4)'),
            ('Iter k=4 (no gate)', 'Arm 2 (fixed k=4)', obs_r_iter - obs_r_fixed, boot_r_iter - boot_r_fixed, 'Iter k=4 vs Arm 2 (hop-conditioning alone at k=4)'),
        ]

        rows = []
        p_raws = []
        for a_name, b_name, obs_diff, boot_diffs, desc in comps:
            valid = ~np.isnan(boot_diffs)
            diffs_v = boot_diffs[valid]
            ci_lo, ci_hi = np.percentile(diffs_v, [2.5, 97.5])
            p_val = 2 * min(float((diffs_v <= 0).mean()), float((diffs_v >= 0).mean()))
            p_val = min(1.0, max(p_val, 1.0 / N_BOOT))
            p_raws.append(p_val)
            rows.append({
                'split': split_name,
                'type': qtype,
                'n': n,
                'comparison': f'{a_name} - {b_name}',
                'description': desc,
                'delta_recall_k4': obs_diff,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'p_value': p_val,
            })
        p_holms = holm(p_raws)
        for r, ph in zip(rows, p_holms):
            r['p_holm'] = ph
        all_sig_rows.extend(rows)

    df_points = pd.DataFrame(point_rows)
    df_points.to_csv(os.path.join(RESULTS_DIR, 'TABLE_matched_depth_k4_point_estimates.csv'), index=False)
    print("\nSaved TABLE_matched_depth_k4_point_estimates.csv")

    df_sig = pd.DataFrame(all_sig_rows)
    df_sig.to_csv(os.path.join(RESULTS_DIR, 'TABLE_matched_depth_k4_bootstrap.csv'), index=False)
    print("Saved TABLE_matched_depth_k4_bootstrap.csv")

def main():
    with open(os.path.join(RESULTS_DIR, 'precomputed_multi_depth_arrays.pkl'), 'rb') as f:
        data = pickle.load(f)

    run_multi_depth(data)
    run_matched_depth_k4(data)
    print("\nAll tables recomputed successfully.")

if __name__ == '__main__':
    main()

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.stats import holm

with open('results_full/precomputed_multi_depth_arrays.pkl', 'rb') as f:
    data = pickle.load(f)

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

def run_analysis_for_subset(arr, subset_name):
    n = arr['r_fixed'].shape[0]
    print(f"\n=======================================================")
    print(f"Running 8-Depth Analysis for {subset_name.upper()} (N={n})")
    print(f"=======================================================")
    
    # 1. Observed curves
    obs_k_fixed = arr['k_fixed']
    obs_r_fixed = arr['r_fixed'].mean(axis=0)
    
    obs_k_4b = arr['k_4b'].mean(axis=0)
    obs_r_4b = arr['r_4b'].mean(axis=0)
    
    obs_k_5 = arr['k_5'].mean(axis=0)
    obs_r_5 = arr['r_5'].mean(axis=0)
    
    obs_k_6 = arr['k_6'].mean(axis=0)
    obs_r_6 = arr['r_6'].mean(axis=0)
    
    print(f"Observed k-ranges:")
    print(f"  Fixed-k: [{obs_k_fixed.min():.2f}, {obs_k_fixed.max():.2f}]")
    print(f"  Arm 4b : [{obs_k_4b.min():.2f}, {obs_k_4b.max():.2f}]")
    print(f"  Arm 5  : [{obs_k_5.min():.2f}, {obs_k_5.max():.2f}]")
    print(f"  Arm 6  : [{obs_k_6.min():.2f}, {obs_k_6.max():.2f}]")
    
    # Allocate bootstrap matrices
    boot_r_fixed = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
    boot_r_4b = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
    boot_r_5 = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
    boot_r_6 = np.full((N_BOOT, len(TARGET_DEPTHS)), np.nan)
    
    # Dense grid for plotting bands
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
    print("\nPoint Estimates at Target Depths:")
    print(df_pts.round(4).to_string(index=False))
    
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
                
        # Holm adjustment across the 8-depth family
        covered_indices = [i for i, r in enumerate(depth_results) if r['covered']]
        p_raws = [depth_results[i]['p_raw'] for i in covered_indices]
        p_holms = holm(p_raws)
        
        for idx, orig_idx in enumerate(covered_indices):
            depth_results[orig_idx]['p_holm'] = p_holms[idx]
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

res_bridge = run_analysis_for_subset(data['bridge'], 'bridge')
res_comp = run_analysis_for_subset(data['comp'], 'comparison')

df_comp_all = pd.concat([res_bridge['df_comp'], res_comp['df_comp']], ignore_index=True)
df_comp_all.to_csv('results_full/TABLE_multi_depth_significance.csv', index=False)
res_bridge['df_comp'].to_csv('results_full/TABLE_multi_depth_bridge.csv', index=False)
res_comp['df_comp'].to_csv('results_full/TABLE_multi_depth_comparison.csv', index=False)

df_pts_all = pd.concat([
    res_bridge['df_pts'].assign(subset='bridge'),
    res_comp['df_pts'].assign(subset='comparison')
], ignore_index=True)
df_pts_all.to_csv('results_full/TABLE_multi_depth_point_estimates.csv', index=False)

print("\n" + "="*80)
print("UPDATED 8-DEPTH SIGNIFICANCE TABLE (BRIDGE):")
print("="*80)
print(res_bridge['df_comp'].to_string(index=False))

print("\n" + "="*80)
print("UPDATED 8-DEPTH SIGNIFICANCE TABLE (COMPARISON):")
print("="*80)
print(res_comp['df_comp'].to_string(index=False))

# =========================================================================
# FIGURE GENERATION
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for idx, (res, title) in enumerate([(res_bridge, 'Bridge Multi-Hop Questions (N=2077)'),
                                    (res_comp, 'Comparison Multi-Hop Questions (N=563)')]):
    ax = axes[idx]
    grid_k = res['grid_k']
    
    # 1. Fixed-k (interpolated line + points)
    lo_fix = np.nanpercentile(res['band_r_fixed'], 2.5, axis=0)
    hi_fix = np.nanpercentile(res['band_r_fixed'], 97.5, axis=0)
    ax.fill_between(grid_k, lo_fix, hi_fix, color='#1f77b4', alpha=0.15)
    fix_mask = (grid_k >= 1.0) & (grid_k <= 6.0)
    fix_interp_r = np.interp(grid_k[fix_mask], res['obs_k_fixed'], res['obs_r_fixed'])
    ax.plot(grid_k[fix_mask], fix_interp_r, '-', color='#1f77b4', lw=2.2, label='Fixed-k (Arm 2 interpolated)')
    ax.scatter(res['obs_k_fixed'], res['obs_r_fixed'], color='#1f77b4', s=45, zorder=4, edgecolor='black', lw=0.8)
    
    # 2. Arm 4b (Calibrated Single-Pass Gating)
    lo_4b = np.nanpercentile(res['band_r_4b'], 2.5, axis=0)
    hi_4b = np.nanpercentile(res['band_r_4b'], 97.5, axis=0)
    valid_4b = ~np.isnan(lo_4b)
    ax.fill_between(grid_k[valid_4b], lo_4b[valid_4b], hi_4b[valid_4b], color='#ff7f0e', alpha=0.15)
    ord_4b = np.argsort(res['obs_k_4b'])
    ax.plot(res['obs_k_4b'][ord_4b], np.maximum.accumulate(res['obs_r_4b'][ord_4b]), 's--', color='#ff7f0e', lw=1.8, ms=4, label='Arm 4b (Calib gating sweep)')
    
    # 3. Arm 5 (Calibrated Iterative Gating)
    lo_5 = np.nanpercentile(res['band_r_5'], 2.5, axis=0)
    hi_5 = np.nanpercentile(res['band_r_5'], 97.5, axis=0)
    valid_5 = ~np.isnan(lo_5)
    ax.fill_between(grid_k[valid_5], lo_5[valid_5], hi_5[valid_5], color='#2ca02c', alpha=0.15)
    ord_5 = np.argsort(res['obs_k_5'])
    ax.plot(res['obs_k_5'][ord_5], np.maximum.accumulate(res['obs_r_5'][ord_5]), '^-.', color='#2ca02c', lw=1.8, ms=4, label='Arm 5 (Iterative gated sweep)')
    
    # 4. Arm 6 (Sufficiency Stopper)
    lo_6 = np.nanpercentile(res['band_r_6'], 2.5, axis=0)
    hi_6 = np.nanpercentile(res['band_r_6'], 97.5, axis=0)
    valid_6 = ~np.isnan(lo_6)
    ax.fill_between(grid_k[valid_6], lo_6[valid_6], hi_6[valid_6], color='#d62728', alpha=0.15)
    ord_6 = np.argsort(res['obs_k_6'])
    ax.plot(res['obs_k_6'][ord_6], np.maximum.accumulate(res['obs_r_6'][ord_6]), 'd:', color='#d62728', lw=1.8, ms=4, label='Arm 6 (Sufficiency sweep)')
    
    # 5. Iterative ungated points at k=4 and k=5
    ax.scatter([4.0], [res['r_iter_k4']], marker='X', color='#9467bd', s=110, zorder=6, edgecolor='black', lw=1.0, label='Iterative ungated (k=4, 5)')
    ax.scatter([5.0], [res['r_iter_k5']], marker='X', color='#9467bd', s=110, zorder=6, edgecolor='black', lw=1.0)
    
    # Dotted k=4 line
    ax.axvline(x=4.0, color='gray', linestyle=':', lw=1.5, alpha=0.8, label='Reference Depth k=4.0')
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Mean Retrieval Depth ($k$ chunks)', fontsize=12)
    ax.set_ylabel('Supporting Paragraph Recall', fontsize=12)
    ax.set_xlim(1.5, 5.5)
    # y-axis lower limit 0.55 on both panels as requested
    ax.set_ylim(0.55, 0.92 if idx == 0 else 1.00)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92)

plt.tight_layout()
plt.savefig('figures/fig_pareto_recall_vs_k.pdf', bbox_inches='tight')
plt.savefig('figures/fig_pareto_recall_vs_k.png', bbox_inches='tight')
plt.savefig('results_full/fig_pareto_recall_vs_k.png', bbox_inches='tight')
plt.savefig('results_full/fig_pareto_recall_vs_k.pdf', bbox_inches='tight')
plt.close()
print("Saved updated figures.")

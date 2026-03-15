#!/usr/bin/env python3
"""
FFSI Publication Figures — Combined Script
==========================================
Generates two figures:

Figure 1 — fig_ffsi_rates_combined
    2x2 bar chart: FFSI rates across four biological datasets

Figure 2 — fig_histogram_pair_vs_full_k6
    2-panel histogram: pair-greedy vs full-data greedy entropy gap at k=6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import scipy.stats as stats
from pathlib import Path

# =============================================================================
# PUBLICATION STYLE — Bioinformatics Journal
# =============================================================================

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
})

# Shared color palette
COLORS = {
    'gene_expression':        '#1f77b4',
    'methylation_binary':     '#ff7f0e',
    'methylation_multiclass': '#2ca02c',
    'csf':                    '#d62728',
    'histogram':              '#9467bd',   # purple for both histogram panels
}

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("/Users/macbook/Documents/FFSI-paper/results")
OUTPUT_DIR  = Path("/Users/macbook/Documents/FFSI-paper/figures")

# --- Figure 1: FFSI rates ---
DATASET_NAMES = {
    'gene_expression':        'ALL/AML Microarray',
    'methylation_binary':     'Methylation (Binary)',
    'methylation_multiclass': 'Methylation (Multiclass)',
    'csf':                    'CSF scRNA-seq',
}
N_INSTANCES = {
    'gene_expression':        72,
    'methylation_binary':     233,
    'methylation_multiclass': 233,
    'csf':                    9893,
}

# --- Figure 2: Histogram ---
HISTOGRAM_DATASETS = [
    {
        'path':       RESULTS_DIR / "greedy_pairs/greedy_pairs_vs_singles_mirna_pg1k_k6.csv",
        'label':      'BRCA miRNA-Seq',
        'panel':      '(a)',
        'n_features': 1881,
        'n_sampled':  1000,
    },
    {
        'path':       RESULTS_DIR / "greedy_pairs/greedy_pairs_vs_singles_binary_pg1k_k6.csv",
        'label':      'Methylation (Binary)',
        'panel':      '(b)',
        'n_features': 5000,
        'n_sampled':  1000,
    },
]

N_BINS = 25


# =============================================================================
# SHARED HELPERS
# =============================================================================

def proportion_ci_wald(count, n, confidence=0.95):
    """Wald (Normal Approximation) interval for a binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = count / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / n)
    return p, max(0, p - margin), min(1, p + margin)


# =============================================================================
# FIGURE 1 — FFSI RATES
# =============================================================================

def load_trials_data(dataset_name: str) -> pd.DataFrame:
    """Load FFSI trial results for a dataset."""
    if dataset_name == 'gene_expression':
        df = pd.read_csv(RESULTS_DIR / "gene_expression" / "ffsi_trials_results.csv")
        return df[df['n_samples'] != 1000]
    else:
        folder = {
            'methylation_binary':     'methylation',
            'methylation_multiclass': 'methylation_multiclass',
            'csf':                    'csf',
        }[dataset_name]
        dfs = []
        for size in ['10K', '10k', '50K', '100K']:
            path = RESULTS_DIR / folder / f"ffsi_trials_results_{size}.csv"
            if path.exists():
                dfs.append(pd.read_csv(path))
        return pd.concat(dfs, ignore_index=True) if dfs else None


def create_rates_figure(all_data: dict, save_path=None):
    """2x2 bar chart of FFSI rates across four datasets."""
    datasets     = ['gene_expression', 'methylation_binary',
                    'methylation_multiclass', 'csf']
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    axes = axes.flatten()

    for dataset_name, ax, panel in zip(datasets, axes, panel_labels):
        color  = COLORS[dataset_name]
        title  = DATASET_NAMES[dataset_name]

        df_trials   = all_data[dataset_name]
        max_samples = df_trials['n_samples'].max()
        df_main     = df_trials[df_trials['n_samples'] == max_samples].sort_values('k')

        rates, ci_lower, ci_upper = [], [], []
        for _, row in df_main.iterrows():
            p, lo, hi = proportion_ci_wald(row['ffsi_count'], row['n_samples'])
            rates.append(p * 100)
            ci_lower.append(lo * 100)
            ci_upper.append(hi * 100)

        rates    = np.array(rates)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)
        x_pos    = np.arange(len(df_main))

        bars = ax.bar(x_pos, rates, width=0.6,
                      color=color, edgecolor='black', linewidth=0.8)
        ax.errorbar(x_pos, rates,
                    yerr=[rates - ci_lower, ci_upper - rates],
                    fmt='none', color='black',
                    capsize=5, capthick=1.2, linewidth=1.2)

        ax.set_xlabel('Number of selected features (k)')
        ax.set_ylabel('FFSI rate (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_main['k'].values)
        ax.set_title(f'{panel} {title}', fontweight='bold', fontsize=11, pad=18)
        ax.text(0.5, 1.02, f'R = {max_samples:,} random samples',
                transform=ax.transAxes, fontsize=8, ha='center', va='bottom')

        max_val = max(ci_upper) if len(ci_upper) > 0 else 1
        if max_val < 1:
            ax.set_ylim(0, max(0.2, max_val * 1.5))
        elif max_val < 5:
            ax.set_ylim(0, max_val * 1.5)
        else:
            ax.set_ylim(0, min(55, max_val * 1.2))

        for bar, rate, ci_hi in zip(bars, rates, ci_upper):
            if rate < 1:
                txt = f'{rate:.2f}%'
            elif rate < 10:
                txt = f'{rate:.1f}%'
            else:
                txt = f'{rate:.0f}%'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ci_hi + max_val * 0.03,
                    txt, ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    plt.tight_layout(h_pad=3.0, w_pad=2.0)

    if save_path:
        plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}.png', format='png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}.pdf / .png")

    return fig


# =============================================================================
# FIGURE 2 — HISTOGRAM: pg1k VS g-full
# =============================================================================

def load_gap(path: Path) -> np.ndarray:
    """
    gap = full_singles_cond_entropy - pairs_cond_entropy
    Positive => pg1k on 1K subsample beats g-full on full dataset.
    """
    df = pd.read_csv(path)
    df = df[df['trial'] != 'CONFIG'].copy()
    for col in ['pairs_cond_entropy', 'full_singles_cond_entropy']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return (df['full_singles_cond_entropy'] - df['pairs_cond_entropy']).dropna().values


def create_histogram_figure(save_path=None):
    """
    2-panel histogram of H(Y|g-full(k)) - H(Y|pg1k(k)) at k=6.
    Solid bars: positive gap (pg1k outperforms g-full).
    Hatched bars: negative gap (g-full outperforms pg1k).
    """
    color = COLORS['histogram']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    for ax, d in zip(axes, HISTOGRAM_DATASETS):
        if not d['path'].exists():
            raise FileNotFoundError(
                f"File not found: {d['path']}\n"
                "Update HISTOGRAM_DATASETS paths in the CONFIGURATION section."
            )

        gap   = load_gap(d['path'])
        n_pos = (gap > 0).sum()
        n_neg = (gap <= 0).sum()
        n     = len(gap)

        # Symmetric bins so zero falls on a bin boundary
        abs_max = max(abs(gap.min()), abs(gap.max())) * 1.05
        bins    = np.linspace(-abs_max, abs_max, N_BINS + 1)

        # Solid: pg1k outperforms g-full; hatched: g-full outperforms pg1k
        ax.hist(gap[gap > 0], bins=bins, color=color, alpha=0.75,
                edgecolor='black', linewidth=0.6)
        ax.hist(gap[gap <= 0], bins=bins, color=color, alpha=0.25,
                edgecolor='black', linewidth=0.6, hatch='///')

        # Reference lines
        ax.axvline(0, color='black', linewidth=1.1, linestyle='--', alpha=0.7)
        ax.axvline(gap.mean(), color=color, linewidth=1.4,
                   linestyle='-', alpha=0.9)

        ax.set_xlabel(
            r'$H(Y \mid G_{\mathrm{g\text{-}full}}(k)) - H(Y \mid G_{\mathrm{pg1k}}(k))$  (nats)'
        )
        ax.set_ylabel('Number of trials')
        ax.set_title(f"{d['panel']} {d['label']},  $k = 6$",
                     fontweight='bold', fontsize=11, pad=14)
        ax.text(0.5, 1.01, f'{n} random trials',
                transform=ax.transAxes, fontsize=8, ha='center', va='bottom')

        legend_elements = [
            mpatches.Patch(facecolor=color, alpha=0.75, edgecolor='black',
                           linewidth=0.6,
                           label=f'pg1k outperforms g-full  ({n_pos}/{n})'),
            mpatches.Patch(facecolor=color, alpha=0.25, edgecolor='black',
                           linewidth=0.6, hatch='///',
                           label=f'g-full outperforms pg1k  ({n_neg}/{n})'),
            Line2D([0], [0], color=color, linewidth=1.4,
                   label=f'Mean = {gap.mean():.4f}'),
        ]
        ax.legend(handles=legend_elements, frameon=False, fontsize=8.5,
                  loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  ncol=1, handlelength=1.5)

    plt.tight_layout(w_pad=3.5)
    plt.subplots_adjust(bottom=0.28)

    if save_path:
        plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}.png', format='png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}.pdf / .png")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: FFSI rates ---
    print("=" * 50)
    print("Figure 1 — FFSI Rates")
    print("=" * 50)
    print("\nLoading data...")
    all_data = {}
    for name in ['gene_expression', 'methylation_binary',
                 'methylation_multiclass', 'csf']:
        all_data[name] = load_trials_data(name)
        print(f"  {name}: loaded")

    print("\nGenerating figure...")
    create_rates_figure(
        all_data,
        save_path=OUTPUT_DIR / "fig_ffsi_rates_combined"
    )

    # --- Figure 2: Histogram ---
    print("\n" + "=" * 50)
    print("Figure 2 — Histogram: pg1k vs g-full")
    print("=" * 50)
    print("\nData summary:")
    for d in HISTOGRAM_DATASETS:
        gap   = load_gap(d['path'])
        n_pos = (gap > 0).sum()
        print(f"  {d['label']}: mean={gap.mean():.4f}, "
              f"std={gap.std():.4f}, pg1k outperforms g-full={n_pos}/{len(gap)}")

    print("\nGenerating figure...")
    create_histogram_figure(
        save_path=OUTPUT_DIR / "fig_histogram_pair_vs_full_k6"
    )

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
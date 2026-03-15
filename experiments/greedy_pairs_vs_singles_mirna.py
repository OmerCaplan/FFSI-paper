"""
Greedy Pairs vs Greedy Singles Feature Selection Experiment
TCGA-BRCA miRNA Expression Data (Cancer vs Normal)

Compares two feature selection strategies on the SAME sampled features:
1. Greedy-Pairs: Select best pair → best conditional pair → ... (k features)
2. Greedy-Singles: Select best feature → best conditional feature → ... (k features)

Reports conditional entropy H(Y|X) - lower is better (approaches 0).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import random
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime


# ——— CORE ENTROPY FUNCTIONS ———

def calculate_base_entropy(labels):
    """Calculate H(Y) - the base entropy of labels."""
    n_total = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_total
    return -np.sum(probs * np.log2(probs + 1e-10))


def calculate_conditional_entropy(features, labels):
    """
    Calculate conditional entropy H(Y|X).
    Lower is better - approaches 0 when X perfectly predicts Y.
    """
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    n_total = len(labels)
    n_cols = features.shape[1]

    # Create joint state from all feature columns
    multipliers = 2 ** np.arange(n_cols)
    row_ids = features @ multipliers

    unique_ids, inverse = np.unique(row_ids, return_inverse=True)

    weighted_entropy = 0.0
    for i in range(len(unique_ids)):
        mask = (inverse == i)
        group_labels = labels[mask]
        n_group = len(group_labels)

        _, counts = np.unique(group_labels, return_counts=True)
        probs = counts / n_group
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        weighted_entropy += (n_group / n_total) * ent

    return round(weighted_entropy, 4)


def calculate_conditional_entropy_diff(new_features, existing_features, labels):
    """
    Calculate reduction in conditional entropy from adding new features.
    Returns H(Y|X_existing) - H(Y|X_existing, X_new) (positive = improvement)
    """
    if existing_features is None:
        base_h = calculate_base_entropy(labels)
        new_h = calculate_conditional_entropy(new_features, labels)
        return round(base_h - new_h, 4)

    if len(new_features.shape) == 1:
        new_features = new_features.reshape(-1, 1)
    if len(existing_features.shape) == 1:
        existing_features = existing_features.reshape(-1, 1)

    h_existing = calculate_conditional_entropy(existing_features, labels)
    combined = np.hstack([existing_features, new_features])
    h_combined = calculate_conditional_entropy(combined, labels)

    return round(h_existing - h_combined, 4)


# ——— GREEDY SELECTION METHODS ———

def greedy_pairs_selection(X, feature_names, labels, k=6):
    """
    Greedy selection in pairs.
    """
    n_features = X.shape[1]
    n_pairs = k // 2
    base_entropy = calculate_base_entropy(labels)

    results = {
        'method': 'greedy_pairs',
        'steps': [],
        'selected_indices': [],
        'selected_feature_names': [],
        'cumulative_cond_entropy': [],
        'step_reductions': []
    }

    selected_indices = []
    X_selected = None

    for pair_num in range(n_pairs):
        available = [i for i in range(n_features) if i not in selected_indices]

        best_pair = None
        best_reduction = -np.inf

        for i, j in combinations(available, 2):
            x_pair = X[:, [i, j]]
            reduction = calculate_conditional_entropy_diff(x_pair, X_selected, labels)

            if reduction > best_reduction:
                best_reduction = reduction
                best_pair = (i, j)

        selected_indices.extend(best_pair)

        if X_selected is None:
            X_selected = X[:, list(best_pair)]
        else:
            X_selected = np.hstack([X_selected, X[:, list(best_pair)]])

        cond_entropy = calculate_conditional_entropy(X_selected, labels)

        results['steps'].append({
            'step': pair_num + 1,
            'features_added_names': [feature_names[best_pair[0]], feature_names[best_pair[1]]],
            'step_reduction': best_reduction,
            'cumulative_cond_entropy': cond_entropy
        })

        results['step_reductions'].append(best_reduction)
        results['cumulative_cond_entropy'].append(cond_entropy)

    results['selected_indices'] = selected_indices
    results['selected_feature_names'] = [feature_names[i] for i in selected_indices]
    results['final_cond_entropy'] = results['cumulative_cond_entropy'][-1] if results['cumulative_cond_entropy'] else base_entropy

    return results


def greedy_singles_selection(X, feature_names, labels, k=6):
    """
    Standard greedy selection one feature at a time.
    """
    n_features = X.shape[1]
    base_entropy = calculate_base_entropy(labels)

    results = {
        'method': 'greedy_singles',
        'steps': [],
        'selected_indices': [],
        'selected_feature_names': [],
        'cumulative_cond_entropy': [],
        'step_reductions': []
    }

    selected_indices = []
    X_selected = None

    for step in range(k):
        available = [i for i in range(n_features) if i not in selected_indices]

        best_feature = None
        best_reduction = -np.inf

        for i in available:
            x_feat = X[:, i]
            reduction = calculate_conditional_entropy_diff(x_feat, X_selected, labels)

            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = i

        selected_indices.append(best_feature)

        if X_selected is None:
            X_selected = X[:, [best_feature]]
        else:
            X_selected = np.hstack([X_selected, X[:, [best_feature]]])

        cond_entropy = calculate_conditional_entropy(X_selected, labels)

        results['steps'].append({
            'step': step + 1,
            'features_added_names': [feature_names[best_feature]],
            'step_reduction': best_reduction,
            'cumulative_cond_entropy': cond_entropy
        })

        results['step_reductions'].append(best_reduction)
        results['cumulative_cond_entropy'].append(cond_entropy)

    results['selected_indices'] = selected_indices
    results['selected_feature_names'] = [feature_names[i] for i in selected_indices]
    results['final_cond_entropy'] = results['cumulative_cond_entropy'][-1] if results['cumulative_cond_entropy'] else base_entropy

    return results


# ——— TRIAL RUNNER ———

def run_single_trial(args):
    """
    Worker function for a single trial.
    Runs BOTH greedy-pairs and greedy-singles on the SAME sampled features.
    """
    trial_id, X_full, labels, all_feature_names, n_sample_features, k, seed = args

    np.random.seed(seed)
    random.seed(seed)

    n_total_features = X_full.shape[1]

    # Sample features (same for both methods)
    sampled_indices = np.random.choice(n_total_features, size=n_sample_features, replace=False)
    sampled_indices = np.sort(sampled_indices)

    X_sampled = X_full[:, sampled_indices]
    sampled_feature_names = [all_feature_names[i] for i in sampled_indices]

    # Run greedy-pairs
    pairs_result = greedy_pairs_selection(X_sampled, sampled_feature_names, labels, k=k)

    # Run greedy-singles on SAME sampled features
    singles_result = greedy_singles_selection(X_sampled, sampled_feature_names, labels, k=k)

    print(f"  Trial {trial_id + 1} done | Pairs H(Y|X): {pairs_result['final_cond_entropy']:.4f} | Singles H(Y|X): {singles_result['final_cond_entropy']:.4f}")

    return {
        'trial_id': trial_id,
        'seed': seed,
        'greedy_pairs': {
            'selected_feature_names': pairs_result['selected_feature_names'],
            'step_reductions': pairs_result['step_reductions'],
            'cumulative_cond_entropy': pairs_result['cumulative_cond_entropy'],
            'final_cond_entropy': pairs_result['final_cond_entropy']
        },
        'greedy_singles': {
            'selected_feature_names': singles_result['selected_feature_names'],
            'step_reductions': singles_result['step_reductions'],
            'cumulative_cond_entropy': singles_result['cumulative_cond_entropy'],
            'final_cond_entropy': singles_result['final_cond_entropy']
        }
    }


def run_experiment(data_path, n_trials=100, n_sample_features=1881, k=8,
                   base_seed=42, n_workers=None, output_dir=None):
    """Run the full experiment."""

    if n_workers is None:
        n_workers = cpu_count()

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)

    # ── EDIT 1: label column for miRNA data ──────────────────────────────────
    label_col = 'label'
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values.astype(np.int8)

    # ── EDIT 2: labels already 0/1 integers ──────────────────────────────────
    labels = df[label_col].values.astype(np.int8)
    feature_names = feature_cols

    base_entropy = calculate_base_entropy(labels)

    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"Base entropy H(Y): {base_entropy:.4f}")
    print(f"Number of features: {len(feature_names)}")

    # ========================================
    # Run greedy-singles on FULL data (once)
    # ========================================
    print(f"\n--- Running greedy-singles on ALL {len(feature_names)} features (once) ---")
    full_singles_result = greedy_singles_selection(X, feature_names, labels, k=k)
    full_singles_h = full_singles_result['final_cond_entropy']
    print(f"Greedy-singles (full data) H(Y|X): {full_singles_h:.4f}")
    print(f"Selected features: {full_singles_result['selected_feature_names']}")

    # ========================================
    # Run trials (both methods on same sampled features)
    # ========================================
    print(f"\n--- Running {n_trials} trials ({n_sample_features} sampled features each) ---")
    print(f"Both greedy-pairs and greedy-singles run on SAME sampled features")
    print(f"Using {n_workers} workers...\n")

    args_list = [
        (i, X, labels, feature_names, n_sample_features, k, base_seed + i)
        for i in range(n_trials)
    ]

    all_results = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(run_single_trial, args_list),
                          total=n_trials,
                          desc="Running trials",
                          smoothing=0.1):
            all_results.append(result)

    # ========================================
    # Compare results
    # ========================================
    for r in all_results:
        pairs_h = r['greedy_pairs']['final_cond_entropy']
        singles_h = r['greedy_singles']['final_cond_entropy']
        r['comparison'] = {
            'cond_entropy_diff': pairs_h - singles_h,
            'pairs_better': bool(pairs_h < singles_h)
        }

    pairs_h_list = [r['greedy_pairs']['final_cond_entropy'] for r in all_results]
    singles_h_list = [r['greedy_singles']['final_cond_entropy'] for r in all_results]
    pairs_better_count = sum(1 for r in all_results if r['comparison']['pairs_better'])

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"\nTrials: {n_trials}")
    print(f"Sampled features: {n_sample_features}")
    print(f"k = {k}")
    print(f"Base entropy H(Y): {base_entropy:.4f}")

    print(f"\n--- Conditional Entropy H(Y|X) (lower is better) ---")
    print(f"Greedy-Singles (FULL {len(feature_names)} features): {full_singles_h:.4f}")
    print(f"Greedy-Pairs   (sampled): {np.mean(pairs_h_list):.4f} ± {np.std(pairs_h_list):.4f} (min: {np.min(pairs_h_list):.4f}, max: {np.max(pairs_h_list):.4f})")
    print(f"Greedy-Singles (sampled): {np.mean(singles_h_list):.4f} ± {np.std(singles_h_list):.4f} (min: {np.min(singles_h_list):.4f}, max: {np.max(singles_h_list):.4f})")
    print(f"\nPairs better than sampled singles: {pairs_better_count}/{n_trials} ({pairs_better_count/n_trials*100:.1f}%)")

    pairs_better_than_full = sum(1 for r in all_results if r['greedy_pairs']['final_cond_entropy'] < full_singles_h)
    print(f"Pairs better than FULL singles: {pairs_better_than_full}/{n_trials} ({pairs_better_than_full/n_trials*100:.1f}%)")

    # Save results as CSV
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        rows = []

        # First row: metadata/constants
        rows.append({
            'trial': 'CONFIG',
            'pairs_cond_entropy': f'k={k}',
            'singles_cond_entropy': f'n_samples={n_sample_features}',
            'full_singles_cond_entropy': full_singles_h,
            'difference': f'H(Y)={base_entropy:.4f}',
            'pairs_better_than_sampled': f'features={len(feature_names)}',
            'pairs_better_than_full': f'trials={n_trials}',
            'pairs_features': '|'.join(full_singles_result['selected_feature_names']),
            'singles_features': 'FULL_DATA_SINGLES_FEATURES'
        })

        # Trial rows
        for r in all_results:
            rows.append({
                'trial': r['trial_id'] + 1,
                'pairs_cond_entropy': r['greedy_pairs']['final_cond_entropy'],
                'singles_cond_entropy': r['greedy_singles']['final_cond_entropy'],
                'full_singles_cond_entropy': full_singles_h,
                'difference': r['comparison']['cond_entropy_diff'],
                'pairs_better_than_sampled': r['comparison']['pairs_better'],
                'pairs_better_than_full': r['greedy_pairs']['final_cond_entropy'] < full_singles_h,
                'pairs_features': '|'.join(r['greedy_pairs']['selected_feature_names']),
                'singles_features': '|'.join(r['greedy_singles']['selected_feature_names'])
            })

        results_df = pd.DataFrame(rows)
        results_path = Path(output_dir) / f"greedy_pairs_vs_singles_mirna_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

    return all_results


# ==================== CONFIGURATION ====================

DATA_PATH = Path("/Users/macbook/Documents/FFSI-paper/data/tcga_brca_mirna_binary.csv")
OUTPUT_DIR = Path("/Users/macbook/Documents/FFSI-paper/results")

N_TRIALS = 100
N_SAMPLE_FEATURES = 1000
K = 8
BASE_SEED = 42
N_WORKERS = None

# =======================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = run_experiment(
        data_path=DATA_PATH,
        n_trials=N_TRIALS,
        n_sample_features=N_SAMPLE_FEATURES,
        k=K,
        base_seed=BASE_SEED,
        n_workers=N_WORKERS,
        output_dir=OUTPUT_DIR
    )

    print("\nDone!")
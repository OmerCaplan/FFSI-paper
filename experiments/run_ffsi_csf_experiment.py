"""
FFSI Experiment for CSF Gene Expression Data

This script runs Forward Feature Selection Incompatibility (FFSI) experiments
on the CSF (Cerebrospinal Fluid) gene expression dataset (MS vs Control).

Dataset characteristics:
- Very sparse data (97.5% zeros)
- Uses 50 pre-selected uncorrelated features (|r| < 0.3)
- Binary label: MS (1) vs Control (0)

Usage:
    python experiments/run_ffsi_csf_experiment.py

Output:
    results/csf/ffsi_trials_results.csv
    results/csf/ffsi_raw_differences.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys
from itertools import combinations, chain, product
import random
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count


# ==================== CONFIGURATION ====================

# Project root (…/FFSI-paper)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to CSF data (with uncorrelated features)
DATA_PATH = PROJECT_ROOT / "data" / "CSF_uncorrelated_50_features.csv"

# Alternative: Use full pickle file and filter features
# DATA_PATH = PROJECT_ROOT / "data" / "CSF_final_gene_matrix_for_FFSI.pkl"

# Output paths for results
RESULTS_DIR = PROJECT_ROOT / "results" / "csf1"
OUTPUT_PATH = RESULTS_DIR / "ffsi_trials_results_100K.csv"
RAW_DIFF_PATH = RESULTS_DIR / "ffsi_raw_difference_100K.csv"
FFSI_DETAILS_PATH = RESULTS_DIR / "ffsi_detailed_cases_100K.csv"

# Experiment parameters
N_SAMPLES_LIST = [100000]  # Number of random subsets to sample
N_FEATURES_LIST = [7, 8, 9, 10]                 # Feature subset sizes to test

# Label column name
LABEL_COLUMN = 'label'

# Reproducibility
BASE_SEED = 123
SEED_OFFSET = 12  # To continue seed sequence from original experiment
SEED = BASE_SEED + SEED_OFFSET

# Parallel processing (None = use all CPU cores)
N_WORKERS = None

# ===========================================================


# ——— CORE FUNCTIONS ———

def powerset(n):
    """Generate indices for powerset of n elements (excluding empty set)"""
    return chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))


def calculate_entropy_gain_numpy(feature_subset, labels):
    """
    Calculate entropy gain for all subsets of features.

    Args:
        feature_subset: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)

    Returns:
        dict mapping subset size -> list of (subset_indices, entropy_gain)
    """
    n_total = len(labels)
    n_features = feature_subset.shape[1]
    entropy_results = {}

    for subset_idx in powerset(n_features):
        subset_cols = feature_subset[:, subset_idx]

        # Create unique row identifiers by treating binary columns as bits
        multipliers = 2 ** np.arange(len(subset_idx))
        row_ids = subset_cols @ multipliers

        unique_ids, inverse = np.unique(row_ids, return_inverse=True)

        # Calculate weighted entropy
        weighted_entropy = 0.0
        for i in range(len(unique_ids)):
            mask = (inverse == i)
            group_labels = labels[mask]
            n_group = len(group_labels)

            _, counts = np.unique(group_labels, return_counts=True)
            probs = counts / n_group
            ent = -np.sum(probs * np.log2(probs + 1e-10))
            weighted_entropy += (n_group / n_total) * ent

        # Entropy gain = 1 - conditional entropy (normalized)
        entropy_gain = round(1 - weighted_entropy, 4)
        r = len(subset_idx)
        entropy_results.setdefault(r, []).append((subset_idx, entropy_gain))

    # Sort by entropy gain (descending)
    for r in entropy_results:
        entropy_results[r].sort(key=lambda x: x[1], reverse=True)

    return entropy_results


def greedy_forward_selection(entropy_results, k):
    """
    Perform greedy forward feature selection.

    Args:
        entropy_results: dict from calculate_entropy_gain_numpy
        k: number of features to select

    Returns:
        tuple: (selected_indices, final_entropy_gain)
    """
    selected = []
    n_features = max(max(idx) for idx, _ in entropy_results[1]) + 1

    for step in range(k):
        best_gain = -1
        best_feature = None

        for f in range(n_features):
            if f in selected:
                continue

            candidate = tuple(sorted(selected + [f]))

            # Find this subset's entropy gain
            for subset_idx, gain in entropy_results[len(candidate)]:
                if subset_idx == candidate:
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = f
                    break

        if best_feature is not None:
            selected.append(best_feature)

    return tuple(sorted(selected)), best_gain


def find_best_subset(entropy_results, k):
    """
    Find the globally optimal subset of size k (exhaustive search).

    Args:
        entropy_results: dict from calculate_entropy_gain_numpy
        k: subset size

    Returns:
        tuple: (best_indices, best_entropy_gain)
    """
    best_subset, best_gain = entropy_results[k][0]
    return best_subset, best_gain


def check_ffsi_with_difference(args):
    """
    Worker function to check a single sample for FFSI.

    Args:
        args: tuple of (features_array, labels_array, n_features, sample_idx, seed, feature_names)

    Returns:
        dict with FFSI results including feature names and entropy values
    """
    features, labels, n_features, sample_idx, seed, feature_names = args

    # Set seed for reproducibility (unique per sample)
    rng = np.random.RandomState(seed + sample_idx)

    # Sample random feature indices
    n_available = features.shape[1]
    if n_available < n_features:
        return {'is_ffsi': False}

    feature_indices = rng.choice(n_available, size=n_features, replace=False)
    feature_subset = features[:, feature_indices]

    # Get the actual feature names for this sample
    selected_feature_names = [feature_names[i] for i in feature_indices]

    # Calculate entropy gains for all subsets
    entropy_results = calculate_entropy_gain_numpy(feature_subset, labels)

    # k = n_features - 1 (select all but one)
    k = n_features - 1

    # Greedy selection
    greedy_subset, greedy_gain = greedy_forward_selection(entropy_results, k)

    # Exhaustive best
    best_subset, best_gain = find_best_subset(entropy_results, k)

    # Check for FFSI
    is_ffsi = (greedy_subset != best_subset) and (best_gain > greedy_gain)
    difference = best_gain - greedy_gain if is_ffsi else 0.0

    # Build result
    result = {
        'is_ffsi': is_ffsi,
        'difference': difference,
        'greedy_gain': greedy_gain,
        'best_gain': best_gain,
    }

    # Add feature names for FFSI cases
    if is_ffsi:
        result['all_features'] = selected_feature_names

    return result


def run_trial(features, labels, n_features, n_samples, seed, n_workers=None, feature_names=None):
    """
    Run a single trial: check n_samples random feature subsets for FFSI.

    Args:
        features: numpy array of all features
        labels: numpy array of labels
        n_features: number of features per subset
        n_samples: number of random subsets to test
        seed: random seed
        n_workers: number of parallel workers
        feature_names: list of feature column names

    Returns:
        dict with trial results including detailed FFSI cases
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    # Prepare arguments for parallel processing
    args_list = [
        (features, labels, n_features, i, seed, feature_names)
        for i in range(n_samples)
    ]

    # Run in parallel
    results = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap(check_ffsi_with_difference, args_list),
                          total=n_samples,
                          desc=f"n_features={n_features}, n_samples={n_samples}"):
            results.append(result)

    # Separate FFSI cases
    ffsi_cases = [r for r in results if r['is_ffsi']]

    ffsi_count = len(ffsi_cases)
    ffsi_rate = ffsi_count / n_samples

    if ffsi_count > 0:
        differences = [r['difference'] for r in ffsi_cases]
        avg_diff = np.mean(differences)
        std_diff = np.std(differences)
        min_diff = np.min(differences)
        max_diff = np.max(differences)
    else:
        avg_diff = std_diff = min_diff = max_diff = 0.0

    return {
        'n_features': n_features,
        'k': n_features - 1,
        'n_samples': n_samples,
        'ffsi_count': ffsi_count,
        'ffsi_rate': ffsi_rate,
        'avg_difference': avg_diff,
        'std_difference': std_diff,
        'min_difference': min_diff,
        'max_difference': max_diff,
        'ffsi_details': ffsi_cases  # Full details for each FFSI case
    }


def run_all_trials(data_path, n_samples_list, n_features_list, label_column='label',
                   seed=None, n_workers=None, output_path=None, raw_differences_path=None,
                   ffsi_details_path=None):
    """
    Run all trial combinations and save results.
    """
    # Load data
    print(f"Loading data from {data_path}...")

    if str(data_path).endswith('.pkl'):
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)

    # Separate features and labels
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")

    labels = df[label_column].values
    feature_cols = [col for col in df.columns if col != label_column]
    features = df[feature_cols].values

    print(f"Data: {len(df)} instances, {len(feature_cols)} features")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Binarize features if not already binary
    unique_vals = np.unique(features)
    if len(unique_vals) > 2:
        print(f"Binarizing features (threshold at median per feature)...")
        features_binary = np.zeros_like(features, dtype=int)
        for i in range(features.shape[1]):
            median_val = np.median(features[:, i])
            features_binary[:, i] = (features[:, i] > median_val).astype(int)
        features = features_binary
        print(f"Binarization complete.")

    # Setup
    if seed is None:
        seed = 42

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"Using {n_workers} parallel workers")

    # Create output directory
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run trials
    all_results = []
    all_raw_diffs = []
    all_ffsi_details = []

    trial_combinations = list(product(n_features_list, n_samples_list))

    print(f"\n{'='*60}")
    print(f"Running {len(trial_combinations)} trial combinations:")
    print(f"  n_features: {n_features_list}")
    print(f"  n_samples: {n_samples_list}")
    print(f"{'='*60}\n")

    for trial_num, (n_features, n_samples) in enumerate(trial_combinations, 1):
        print(f"[Trial {trial_num}/{len(trial_combinations)}] n_features={n_features}, n_samples={n_samples:,}")
        print(f"  (selecting k={n_features-1} from {n_features} features)")

        # Use different seed offset for each trial
        trial_seed = seed + trial_num * 10000

        result = run_trial(
            features=features,
            labels=labels,
            n_features=n_features,
            n_samples=n_samples,
            seed=trial_seed,
            n_workers=n_workers,
            feature_names=feature_cols
        )

        # Process FFSI details
        for ffsi_case in result['ffsi_details']:
            # Raw differences (simple format)
            all_raw_diffs.append({
                'n_features': n_features,
                'n_samples': n_samples,
                'difference': ffsi_case['difference']
            })

            # Detailed FFSI info (minimal)
            detail_row = {
                'n_features': n_features,
                'n_samples': n_samples,
                'difference': ffsi_case['difference'],
                'greedy_gain': ffsi_case['greedy_gain'],
                'best_gain': ffsi_case['best_gain'],
                'all_features': '|'.join(ffsi_case['all_features']),
            }

            all_ffsi_details.append(detail_row)

        # Remove ffsi_details from summary (too large)
        del result['ffsi_details']
        all_results.append(result)

        # Print summary
        print(f"  FFSI count: {result['ffsi_count']} ({result['ffsi_rate']*100:.4f}%)")
        if result['ffsi_count'] > 0:
            print(f"  Avg difference: {result['avg_difference']:.4f}")
            print(f"  Max difference: {result['max_difference']:.4f}")
        print()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    if raw_differences_path and len(all_raw_diffs) > 0:
        raw_df = pd.DataFrame(all_raw_diffs)
        raw_df.to_csv(raw_differences_path, index=False)
        print(f"Raw differences saved to: {raw_differences_path}")

    # Save detailed FFSI info
    if ffsi_details_path and len(all_ffsi_details) > 0:
        details_df = pd.DataFrame(all_ffsi_details)
        details_df.to_csv(ffsi_details_path, index=False)
        print(f"FFSI details saved to: {ffsi_details_path}")

    # Print summary tables
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nFFSI Rate (%):")
    pivot_rate = results_df.pivot(index='n_features', columns='n_samples', values='ffsi_rate')
    print((pivot_rate * 100).round(4).to_string())

    print("\nFFSI Count:")
    pivot_count = results_df.pivot(index='n_features', columns='n_samples', values='ffsi_count')
    print(pivot_count.to_string())

    print("\nAverage Difference (when FFSI):")
    pivot_avg = results_df.pivot(index='n_features', columns='n_samples', values='avg_difference')
    print(pivot_avg.round(4).to_string())

    return results_df


# ==================== MAIN ====================

if __name__ == "__main__":

    # Run all trials
    results = run_all_trials(
        data_path=DATA_PATH,
        n_samples_list=N_SAMPLES_LIST,
        n_features_list=N_FEATURES_LIST,
        label_column=LABEL_COLUMN,
        seed=SEED,
        n_workers=N_WORKERS,
        output_path=OUTPUT_PATH,
        raw_differences_path=RAW_DIFF_PATH,
        ffsi_details_path=FFSI_DETAILS_PATH
    )

    print("\nDone!")
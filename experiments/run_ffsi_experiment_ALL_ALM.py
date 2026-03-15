from pathlib import Path

import numpy as np
import pandas as pd
import sys
from itertools import combinations, chain, product
import random
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count



# ——— CORE FUNCTIONS ———

def powerset(n):
    """Generate indices for powerset of n elements (excluding empty set)"""
    return chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))


def calculate_entropy_gain_numpy(feature_subset, labels):
    """
    Calculate entropy gain for all subsets of features.
    """
    n_total = len(labels)
    n_features = feature_subset.shape[1]
    entropy_results = {}

    for subset_idx in powerset(n_features):
        subset_cols = feature_subset[:, subset_idx]

        multipliers = 2 ** np.arange(len(subset_idx))
        row_ids = subset_cols @ multipliers

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

        entropy_gain = round(1 - weighted_entropy, 2)
        r = len(subset_idx)
        entropy_results.setdefault(r, []).append((subset_idx, entropy_gain))

    for r in entropy_results:
        entropy_results[r].sort(key=lambda x: x[1], reverse=True)

    return entropy_results


def check_ffsi_with_difference(args):
    """
    Worker function to check a single sample for FFSI and return the difference.
    """
    sample_id, sampled_indices, features, labels, k = args

    feature_subset = features[:, sampled_indices]
    entropy_results = calculate_entropy_gain_numpy(feature_subset, labels)

    # Greedy selection
    selected = []
    for i in range(1, k + 1):
        cands = entropy_results.get(i, [])
        if not cands:
            break
        top_subset = cands[0][0]
        for feat in top_subset:
            if feat not in selected:
                selected.append(feat)
            if len(selected) == k:
                break
        if len(selected) == k:
            break

    # Get best k-subset
    best_k = entropy_results.get(k, [(None, None)])[0]
    if best_k[0] is None:
        return {
            'sample_id': sample_id,
            'is_FFSI': False,
            'best_gain': None,
            'greedy_gain': None,
            'difference': None
        }

    best_gain = best_k[1]
    best_set = set(best_k[0])
    greedy_set = set(selected)

    # Check if greedy == optimal
    if greedy_set == best_set:
        return {
            'sample_id': sample_id,
            'is_FFSI': False,
            'best_gain': best_gain,
            'greedy_gain': best_gain,
            'difference': 0
        }

    # Check for ties
    if len(entropy_results[k]) > 1 and best_gain == entropy_results[k][1][1]:
        return {
            'sample_id': sample_id,
            'is_FFSI': False,
            'best_gain': best_gain,
            'greedy_gain': None,
            'difference': None
        }

    # It's FFSI - find greedy gain
    greedy_gain = None
    for subset, gain in entropy_results.get(k, []):
        if set(subset) == greedy_set:
            greedy_gain = gain
            break

    difference = best_gain - greedy_gain if greedy_gain is not None else None

    return {
        'sample_id': sample_id,
        'is_FFSI': True,
        'best_gain': best_gain,
        'greedy_gain': greedy_gain,
        'difference': difference
    }


def run_ffsi_trial(features, labels, n_samples, n_features, seed=None, n_workers=None):
    """
    Run a single FFSI trial and return statistics + raw differences.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if n_workers is None:
        n_workers = cpu_count()

    n_instances, n_total_features = features.shape
    k = n_features - 1

    # Pre-generate all random samples
    all_samples = []
    for i in range(n_samples):
        sampled_indices = random.sample(range(n_total_features), n_features)
        all_samples.append((i + 1, sampled_indices, features, labels, k))

    # Run parallel processing
    results = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap(check_ffsi_with_difference, all_samples, chunksize=100),
                           total=n_samples, 
                           desc=f"n_features={n_features}, n_samples={n_samples}"):
            results.append(result)

    # Collect FFSI results
    ffsi_results = [r for r in results if r['is_FFSI']]
    ffsi_count = len(ffsi_results)

    # Collect raw differences for box plots
    raw_differences = []
    for r in ffsi_results:
        if r['difference'] is not None:
            raw_differences.append({
                'n_features': n_features,
                'k': k,
                'n_samples': n_samples,
                'sample_id': r['sample_id'],
                'difference': r['difference']
            })

    # Calculate summary statistics
    if ffsi_count > 0:
        differences = [r['difference'] for r in ffsi_results if r['difference'] is not None]
        avg_difference = np.mean(differences) if differences else 0
        std_difference = np.std(differences, ddof=1) if len(differences) > 1 else 0
        max_difference = np.max(differences) if differences else 0
        min_difference = np.min(differences) if differences else 0
    else:
        avg_difference = 0
        std_difference = 0
        max_difference = 0
        min_difference = 0

    summary = {
        'n_features': n_features,
        'k': k,
        'n_samples': n_samples,
        'ffsi_count': ffsi_count,
        'ffsi_rate': ffsi_count / n_samples,
        'avg_difference': avg_difference,
        'std_difference': std_difference,
        'min_difference': min_difference,
        'max_difference': max_difference
    }

    return summary, raw_differences


def run_all_trials(data_path, n_samples_list, n_features_list, seed=None, n_workers=None, 
                   output_path=None, raw_differences_path=None):
    """
    Run FFSI trials for all combinations of n_samples and n_features.
    
    Args:
        data_path: Path to CSV data file
        n_samples_list: List of sample sizes to test
        n_features_list: List of feature counts to test
        seed: Random seed for reproducibility
        n_workers: Number of parallel workers
        output_path: Path to save summary results CSV
        raw_differences_path: Path to save raw differences CSV (for box plots)
    
    Returns:
        Tuple of (summary DataFrame, raw differences DataFrame)
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Load data once
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    labels = df['label'].values
    features = df.drop(columns=['label']).values.astype(np.int8)
    n_instances, n_total_features = features.shape
    
    print(f"Data: {n_instances} instances, {n_total_features} features")
    print(f"Using {n_workers} parallel workers")
    print()
    
    # Generate all combinations
    all_combinations = list(product(n_features_list, n_samples_list))
    total_trials = len(all_combinations)
    
    print(f"Running {total_trials} trial combinations:")
    print(f"  n_features: {n_features_list}")
    print(f"  n_samples: {n_samples_list}")
    print("=" * 60)
    print()
    
    # Run all trials
    all_summaries = []
    all_raw_differences = []
    
    for trial_num, (n_features, n_samples) in enumerate(all_combinations, 1):
        print(f"\n[Trial {trial_num}/{total_trials}] n_features={n_features}, n_samples={n_samples:,}")
        print("-" * 40)
        
        # Set seed for this specific combination (reproducible)
        trial_seed = seed + trial_num if seed is not None else None
        
        summary, raw_diffs = run_ffsi_trial(
            features=features,
            labels=labels,
            n_samples=n_samples,
            n_features=n_features,
            seed=trial_seed,
            n_workers=n_workers
        )
        
        all_summaries.append(summary)
        all_raw_differences.extend(raw_diffs)
        
        # Print trial summary
        print(f"  FFSI count: {summary['ffsi_count']} ({100*summary['ffsi_rate']:.4f}%)")
        print(f"  Avg difference: {summary['avg_difference']:.4f}")
        print(f"  Std difference: {summary['std_difference']:.4f}")
        print(f"  Max difference: {summary['max_difference']:.4f}")
    
    # Create DataFrames
    summary_df = pd.DataFrame(all_summaries)
    raw_diff_df = pd.DataFrame(all_raw_differences)
    
    # Save results
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"Summary results saved to: {output_path}")
    
    if raw_differences_path:
        raw_diff_df.to_csv(raw_differences_path, index=False)
        print(f"Raw differences saved to: {raw_differences_path}")
        print(f"  Total raw difference records: {len(raw_diff_df)}")
    
    # Print final summary table
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\nFFSI Rate (%):")
    pivot_rate = summary_df.pivot(index='n_features', columns='n_samples', values='ffsi_rate')
    print((pivot_rate * 100).round(4).to_string())
    
    print("\nFFSI Count:")
    pivot_count = summary_df.pivot(index='n_features', columns='n_samples', values='ffsi_count')
    print(pivot_count.to_string())
    
    print("\nAverage Difference:")
    pivot_avg = summary_df.pivot(index='n_features', columns='n_samples', values='avg_difference')
    print(pivot_avg.round(4).to_string())
    
    print("\nStd Difference:")
    pivot_std = summary_df.pivot(index='n_features', columns='n_samples', values='std_difference')
    print(pivot_std.round(4).to_string())
    
    print("\nMax Difference:")
    pivot_max = summary_df.pivot(index='n_features', columns='n_samples', values='max_difference')
    print(pivot_max.round(4).to_string())
    
    return summary_df, raw_diff_df

# ==================== CONFIGURATION ====================
# Project root (…/FFSI-paper)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to gene expression data
DATA_PATH = PROJECT_ROOT / "data" / "Gene_expression_FFSI.csv"

# Output paths for results
OUTPUT_PATH = PROJECT_ROOT / "results" / "ffsi_trials_results.csv"
RAW_DIFF_PATH = PROJECT_ROOT / "results" / "ffsi_raw_differences.csv"

# Experiment parameters
N_SAMPLES_LIST = [1000, 10000, 100000, 500000]  # Number of random subsets to sample
N_FEATURES_LIST = [7, 8, 9, 10]          # Feature subset sizes to test

# Reproducibility
SEED = 123

# Parallel processing (None = use all CPU cores)
N_WORKERS = 6

# ===========================================================


if __name__ == "__main__":
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Convert relative paths to absolute
    data_path = os.path.join(project_root, DATA_PATH)
    output_path = os.path.join(project_root, OUTPUT_PATH)
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Please download the data following instructions in data/README.md")
        sys.exit(1)
    
    # Run all trials
    summary_df, raw_diff_df = run_all_trials(
        data_path=DATA_PATH,
        n_samples_list=N_SAMPLES_LIST,
        n_features_list=N_FEATURES_LIST,
        seed=SEED,
        n_workers=N_WORKERS,
        output_path=OUTPUT_PATH,
        raw_differences_path=RAW_DIFF_PATH
    )
    
    print("\nDone!")

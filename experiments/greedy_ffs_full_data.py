# # #!/usr/bin/env python3
# # """
# # Greedy Forward Feature Selection on Full Raw Data (v2 - Corrected)
# # ===================================================================
# # This script runs greedy FFS on the complete datasets (not subsampled)
# # to measure the entropy gain at k=6,7,8,9 features.
# #
# # Key fix: Properly binarizes all data before entropy calculation,
# # matching the preprocessing in FFSI experiments.
# # """
# #
# # import pandas as pd
# # import numpy as np
# # from collections import Counter
# # import time
# # from tqdm import tqdm
# #
# # # =============================================================================
# # # CONFIGURATION - Edit these paths for your system
# # # =============================================================================
# #
# # DATASETS = {
# #     # "gene_expression": {
# #     #     "path": "/Users/macbook/Documents/FFSI-paper/data/Gene_expression_FFSI.csv",
# #     #     "label_column": "label",
# #     #     "drop_columns": [],
# #     #     "binarize": False,
# #     # },
# #     # "methylation_binary": {
# #     #     "path": "/Users/macbook/Documents/FFSI-paper/data/methylation_data.csv",
# #     #     "label_column": "Stage_Binarized",
# #     #     "drop_columns": ["Original_Stage", "Sample"],
# #     #     "binarize": False,
# #     # },
# #     # "methylation_multiclass": {
# #     #     "path": "/Users/macbook/Documents/FFSI-paper/data/methylation_data.csv",
# #     #     "label_column": "Original_Stage",
# #     #     "drop_columns": ["Stage_Binarized", "Sample"],
# #     #     "binarize": False,
# #     # },
# #     # "csf": {
# #     #     "path": "/Users/macbook/Documents/FFSI-paper/data/CSF_uncorrelated_50_features.csv",
# #     #     "label_column": "label",
# #     #     "drop_columns": [],
# #     #     "binarize": False,
# #     # },
# #     "ami_gene_expression": {
# #         "path": "/Users/macbook/Documents/FFSI-paper/data/GSE66360_expression.csv",
# #         "label_column": "label",
# #         "drop_columns": ["sample_id", "title", "condition", "cohort"],  # ← add "sample_id"
# #         "binarize": True,
# #     },
# # }
# #
# # # Feature subset sizes to evaluate
# # K_VALUES = [6, 7, 8, 9]
# #
# #
# # # =============================================================================
# # # ENTROPY FUNCTIONS (Optimized for binary features)
# # # =============================================================================
# #
# # def entropy(labels):
# #     """Calculate entropy of a label distribution."""
# #     n = len(labels)
# #     if n == 0:
# #         return 0.0
# #     counts = Counter(labels)
# #     probs = [count / n for count in counts.values()]
# #     return -sum(p * np.log2(p) for p in probs if p > 0)
# #
# #
# # def conditional_entropy_binary(X_selected, labels):
# #     """
# #     Calculate H(Y|X) for binary features using bit-packing.
# #
# #     X_selected: numpy array of shape (n_samples, k) with binary values
# #     labels: numpy array of shape (n_samples,)
# #
# #     Returns: conditional entropy H(Y|X)
# #     """
# #     n = len(labels)
# #     if n == 0:
# #         return 0.0
# #
# #     k = X_selected.shape[1]
# #
# #     # Pack binary features into integers for fast grouping
# #     # Each unique combination of features becomes a unique integer
# #     multipliers = 2 ** np.arange(k)
# #     group_ids = X_selected @ multipliers
# #
# #     # Group by feature combination
# #     unique_ids, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
# #
# #     # Calculate weighted entropy
# #     cond_ent = 0.0
# #     for i, group_id in enumerate(unique_ids):
# #         mask = (inverse == i)
# #         group_labels = labels[mask]
# #         n_group = len(group_labels)
# #
# #         # Entropy of this group
# #         label_counts = Counter(group_labels)
# #         probs = [c / n_group for c in label_counts.values()]
# #         group_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
# #
# #         # Weight by group size
# #         cond_ent += (n_group / n) * group_entropy
# #
# #     return cond_ent
# #
# #
# # def information_gain_binary(X_selected, labels, base_entropy):
# #     """Calculate information gain: H(Y) - H(Y|X)."""
# #     return base_entropy - conditional_entropy_binary(X_selected, labels)
# #
# #
# # # =============================================================================
# # # GREEDY FORWARD FEATURE SELECTION
# # # =============================================================================
# #
# # def greedy_ffs(X, y, max_k=9, show_progress=True):
# #     """
# #     Run greedy forward feature selection up to max_k features.
# #
# #     Args:
# #         X: pandas DataFrame with binary features
# #         y: pandas Series with labels
# #         max_k: maximum number of features to select
# #         show_progress: whether to show progress bar
# #
# #     Returns:
# #         dict with k -> {
# #             'selected_features': list of feature names,
# #             'cumulative_entropy_gain': total info gain with k features,
# #             'step_gains': info gain added at each step
# #         }
# #     """
# #     feature_names = list(X.columns)
# #     n_features = len(feature_names)
# #     n_samples = len(y)
# #
# #     # Convert to numpy arrays
# #     X_np = X.values.astype(np.int8)
# #     labels = y.values
# #
# #     base_entropy = entropy(labels)
# #
# #     print(f"  Base entropy H(Y): {base_entropy:.6f}")
# #     print(f"  Total features: {n_features}, Samples: {n_samples}")
# #
# #     selected_indices = []
# #     selected_names = []
# #     results = {}
# #
# #     prev_gain = 0.0
# #
# #     for k in range(1, max_k + 1):
# #         best_gain = -np.inf
# #         best_feature_idx = None
# #         best_feature_name = None
# #
# #         # Candidates: all features not yet selected
# #         candidates = [i for i in range(n_features) if i not in selected_indices]
# #
# #         if show_progress and k <= 5:
# #             # Show progress for first few steps (most expensive)
# #             iterator = tqdm(candidates, desc=f"  Step {k}", leave=False)
# #         else:
# #             iterator = candidates
# #
# #         for feat_idx in iterator:
# #             # Build candidate feature set
# #             candidate_indices = selected_indices + [feat_idx]
# #             X_candidate = X_np[:, candidate_indices]
# #
# #             # Calculate information gain
# #             gain = information_gain_binary(X_candidate, labels, base_entropy)
# #
# #             if gain > best_gain:
# #                 best_gain = gain
# #                 best_feature_idx = feat_idx
# #                 best_feature_name = feature_names[feat_idx]
# #
# #         # Add best feature
# #         selected_indices.append(best_feature_idx)
# #         selected_names.append(best_feature_name)
# #
# #         # Calculate step gain
# #         step_gain = best_gain - prev_gain
# #         prev_gain = best_gain
# #
# #         results[k] = {
# #             'selected_features': selected_names.copy(),
# #             'cumulative_entropy_gain': best_gain,
# #             'step_gain': step_gain,
# #             'last_feature_added': best_feature_name,
# #         }
# #
# #         if k >= 6:
# #             print(f"  k={k}: Gain = {best_gain:.6f} (step: {step_gain:.6f}), Added: {best_feature_name}")
# #
# #     return results, base_entropy
# #
# #
# # # =============================================================================
# # # DATA LOADING AND PREPROCESSING
# # # =============================================================================
# #
# # def binarize_features(X):
# #     """Binarize features using median threshold (same as FFSI experiments)."""
# #     X_binary = np.zeros_like(X.values, dtype=np.int8)
# #     for i in range(X.shape[1]):
# #         median_val = np.median(X.iloc[:, i])
# #         X_binary[:, i] = (X.iloc[:, i] > median_val).astype(np.int8)
# #     return pd.DataFrame(X_binary, index=X.index, columns=X.columns)
# #
# #
# # def load_dataset(config, dataset_name):
# #     """Load and prepare a dataset."""
# #     print(f"Loading from: {config['path']}")
# #     df = pd.read_csv(config['path'])
# #
# #     # Drop specified columns
# #     drop_cols = [c for c in config['drop_columns'] if c in df.columns]
# #     if drop_cols:
# #         df = df.drop(columns=drop_cols)
# #
# #     # Separate features and labels
# #     label_col = config['label_column']
# #     y = df[label_col]
# #     X = df.drop(columns=[label_col])
# #
# #     # Check if already binary
# #     unique_vals = np.unique(X.values)
# #     is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False})
# #
# #     if config.get('binarize', False) and not is_binary:
# #         print(f"  Binarizing features (threshold at median)...")
# #         X = binarize_features(X)
# #     elif is_binary:
# #         print(f"  Features already binary")
# #         X = X.astype(np.int8)
# #
# #     return X, y
# #
# #
# # # =============================================================================
# # # MAIN EXECUTION
# # # =============================================================================
# #
# # def main():
# #     print("=" * 70)
# #     print("GREEDY FORWARD FEATURE SELECTION ON FULL RAW DATA")
# #     print("=" * 70)
# #
# #     all_results = {}
# #
# #     for dataset_name, config in DATASETS.items():
# #         print(f"\n{'=' * 70}")
# #         print(f"DATASET: {dataset_name}")
# #         print(f"{'=' * 70}")
# #
# #         try:
# #             # Load data
# #             X, y = load_dataset(config, dataset_name)
# #             print(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features")
# #             print(f"  Labels: {dict(Counter(y))}")
# #
# #             # Run greedy FFS
# #             print(f"\nRunning greedy FFS (this may take a while for large datasets)...")
# #             start_time = time.time()
# #             results, base_entropy = greedy_ffs(X, y, max_k=max(K_VALUES))
# #             elapsed = time.time() - start_time
# #             print(f"  Completed in {elapsed:.1f} seconds")
# #
# #             all_results[dataset_name] = {
# #                 'results': results,
# #                 'base_entropy': base_entropy,
# #                 'n_samples': len(y),
# #                 'n_features': X.shape[1],
# #             }
# #
# #         except FileNotFoundError:
# #             print(f"  ERROR: File not found at {config['path']}")
# #         except Exception as e:
# #             import traceback
# #             print(f"  ERROR: {e}")
# #             traceback.print_exc()
# #
# #     # ==========================================================================
# #     # SUMMARY TABLE
# #     # ==========================================================================
# #     print("\n")
# #     print("=" * 70)
# #     print("SUMMARY: CUMULATIVE ENTROPY GAIN BY K")
# #     print("=" * 70)
# #     print(f"{'Dataset':<25} {'H(Y)':<10} {'k=6':<12} {'k=7':<12} {'k=8':<12} {'k=9':<12}")
# #     print("-" * 70)
# #
# #     for dataset_name, data in all_results.items():
# #         results = data['results']
# #         base_ent = data['base_entropy']
# #         gains = [results.get(k, {}).get('cumulative_entropy_gain', None) for k in K_VALUES]
# #         gains_str = [f"{g:.6f}" if g is not None else "N/A" for g in gains]
# #         print(
# #             f"{dataset_name:<25} {base_ent:<10.6f} {gains_str[0]:<12} {gains_str[1]:<12} {gains_str[2]:<12} {gains_str[3]:<12}")
# #
# #     # Normalized gains (as fraction of H(Y))
# #     print("\n")
# #     print("=" * 70)
# #     print("SUMMARY: ENTROPY REDUCTION (% of H(Y))")
# #     print("=" * 70)
# #     print(f"{'Dataset':<25} {'k=6':<12} {'k=7':<12} {'k=8':<12} {'k=9':<12}")
# #     print("-" * 70)
# #
# #     for dataset_name, data in all_results.items():
# #         results = data['results']
# #         base_ent = data['base_entropy']
# #         pcts = []
# #         for k in K_VALUES:
# #             gain = results.get(k, {}).get('cumulative_entropy_gain', None)
# #             if gain is not None and base_ent > 0:
# #                 pcts.append(f"{100 * gain / base_ent:.2f}%")
# #             else:
# #                 pcts.append("N/A")
# #         print(f"{dataset_name:<25} {pcts[0]:<12} {pcts[1]:<12} {pcts[2]:<12} {pcts[3]:<12}")
# #
# #     # ==========================================================================
# #     # SAVE RESULTS
# #     # ==========================================================================
# #     output_rows = []
# #     for dataset_name, data in all_results.items():
# #         results = data['results']
# #         for k in K_VALUES:
# #             if k in results:
# #                 output_rows.append({
# #                     'dataset': dataset_name,
# #                     'k': k,
# #                     'cumulative_entropy_gain': results[k]['cumulative_entropy_gain'],
# #                     'step_gain': results[k]['step_gain'],
# #                     'base_entropy': data['base_entropy'],
# #                     'pct_of_base': results[k]['cumulative_entropy_gain'] / data['base_entropy'] if data[
# #                                                                                                        'base_entropy'] > 0 else 0,
# #                     'n_samples': data['n_samples'],
# #                     'n_features': data['n_features'],
# #                     'selected_features': '|'.join(results[k]['selected_features']),
# #                 })
# #
# #     output_df = pd.DataFrame(output_rows)
# #     output_path = "greedy_ffs_full_data_results.csv"
# #     output_df.to_csv(output_path, index=False)
# #     print(f"\n\nResults saved to: {output_path}")
# #
# #
# # if __name__ == "__main__":
# #     main()



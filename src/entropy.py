"""
Core entropy and information gain calculations for FFSI analysis.

This module implements entropy-based feature evaluation using information gain
as the scoring criterion for forward feature selection.
"""

import numpy as np
from itertools import combinations, chain


def powerset(n):
    """
    Generate indices for powerset of n elements (excluding empty set).
    
    Args:
        n: Number of elements
        
    Yields:
        Tuples of indices representing each subset
    """
    return chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))


def compute_entropy(labels):
    """
    Compute Shannon entropy of a label array.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Entropy in bits
    """
    n = len(labels)
    if n == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_conditional_entropy(feature, labels):
    """
    Compute H(Y|X) - conditional entropy of labels given a feature.
    
    Args:
        feature: Array of feature values
        labels: Array of class labels
        
    Returns:
        Conditional entropy H(Y|X)
    """
    n = len(labels)
    weighted_entropy = 0.0
    
    for val in np.unique(feature):
        mask = (feature == val)
        group_labels = labels[mask]
        n_group = len(group_labels)
        
        ent = compute_entropy(group_labels)
        weighted_entropy += (n_group / n) * ent
    
    return weighted_entropy


def calculate_entropy_gain_numpy(feature_subset, labels):
    """
    Calculate entropy gain for all subsets of features.
    
    Uses information gain: IG(Y, X) = H(Y) - H(Y|X) = 1 - H(Y|X) for normalized labels.
    
    Args:
        feature_subset: 2D numpy array of shape (n_samples, n_features)
        labels: 1D numpy array of class labels
        
    Returns:
        Dictionary mapping subset size -> list of (subset_indices, gain) tuples,
        sorted by gain in descending order
    """
    n_total = len(labels)
    n_features = feature_subset.shape[1]
    entropy_results = {}

    for subset_idx in powerset(n_features):
        subset_cols = feature_subset[:, subset_idx]

        # Encode multi-column subsets as single integers for fast grouping
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

    # Sort by gain (descending) for each subset size
    for r in entropy_results:
        entropy_results[r].sort(key=lambda x: x[1], reverse=True)

    return entropy_results

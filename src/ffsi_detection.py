"""
Forward Feature Selection Incompatibility (FFSI) detection.

This module implements algorithms to detect when greedy forward feature selection
fails to find the optimal feature subset.
"""

import numpy as np
from .entropy import calculate_entropy_gain_numpy


def check_ffsi_with_difference(args):
    """
    Check a single feature sample for FFSI and return the accuracy difference.
    
    FFSI occurs when greedy forward selection produces a different (suboptimal)
    result compared to exhaustive search over all k-subsets.
    
    Args:
        args: Tuple of (sample_id, sampled_indices, features, labels, k)
            - sample_id: Identifier for this sample
            - sampled_indices: Indices of features to use
            - features: Full feature matrix
            - labels: Class labels
            - k: Target number of features to select
            
    Returns:
        Dictionary with:
            - sample_id: Sample identifier
            - is_FFSI: Boolean indicating FFSI incompatibility
            - best_gain: Information gain of optimal k-subset
            - greedy_gain: Information gain of greedy-selected k-subset
            - difference: best_gain - greedy_gain (accuracy loss due to greedy)
    """
    sample_id, sampled_indices, features, labels, k = args

    feature_subset = features[:, sampled_indices]
    entropy_results = calculate_entropy_gain_numpy(feature_subset, labels)

    # Greedy forward selection
    selected = []
    for i in range(1, k + 1):
        cands = entropy_results.get(i, [])
        if not cands:
            break
        top_subset = cands[0][0]  # Best subset of size i
        for feat in top_subset:
            if feat not in selected:
                selected.append(feat)
            if len(selected) == k:
                break
        if len(selected) == k:
            break

    # Get best k-subset from exhaustive search
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

    # Check if greedy found optimal
    if greedy_set == best_set:
        return {
            'sample_id': sample_id,
            'is_FFSI': False,
            'best_gain': best_gain,
            'greedy_gain': best_gain,
            'difference': 0
        }

    # Handle ties (multiple optimal solutions)
    if len(entropy_results[k]) > 1 and best_gain == entropy_results[k][1][1]:
        return {
            'sample_id': sample_id,
            'is_FFSI': False,
            'best_gain': best_gain,
            'greedy_gain': None,
            'difference': None
        }

    # FFSI detected - find greedy's actual gain
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

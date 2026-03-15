"""
FFSI Analysis - Forward Feature Selection Incompatibility

This package provides tools for analyzing when greedy forward feature selection
algorithms fail to find optimal solutions.
"""

from .entropy import (
    powerset,
    compute_entropy,
    compute_conditional_entropy,
    calculate_entropy_gain_numpy
)

from .ffsi_detection import check_ffsi_with_difference

__version__ = "1.0.0"
__all__ = [
    'powerset',
    'compute_entropy', 
    'compute_conditional_entropy',
    'calculate_entropy_gain_numpy',
    'check_ffsi_with_difference'
]

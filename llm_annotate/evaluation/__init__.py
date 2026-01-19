"""
Evaluation framework for comparing annotation strategies.
"""

from .metrics import (
    EvaluationMetrics,
    compute_metrics,
    compare_results,
)
from .benchmark import (
    Benchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    "EvaluationMetrics",
    "compute_metrics",
    "compare_results",
    "Benchmark",
    "BenchmarkResult",
    "run_benchmark",
]

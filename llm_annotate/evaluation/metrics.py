"""
Metrics for evaluating annotation strategies.
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics

from ..core import Document, AnnotationResult


@dataclass
class EvaluationMetrics:
    """
    Comprehensive metrics for evaluating an annotation result.
    """
    # Preservation metrics
    document_preserved: bool = True
    preservation_rate: float = 1.0  # 0-1, how much of original is preserved
    modification_count: int = 0

    # Annotation quality metrics
    annotation_count: int = 0
    annotations_with_valid_positions: int = 0
    annotations_with_anchors: int = 0
    position_accuracy: float = 1.0  # 0-1

    # Efficiency metrics
    llm_calls: int = 0
    total_tokens: int = 0
    tokens_per_annotation: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0

    # Coverage metrics
    document_coverage: float = 0.0  # Fraction of document with annotations nearby
    line_coverage: float = 0.0  # Fraction of lines with annotations

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "preservation": {
                "preserved": self.document_preserved,
                "rate": self.preservation_rate,
                "modifications": self.modification_count,
            },
            "quality": {
                "count": self.annotation_count,
                "valid_positions": self.annotations_with_valid_positions,
                "with_anchors": self.annotations_with_anchors,
                "position_accuracy": self.position_accuracy,
            },
            "efficiency": {
                "llm_calls": self.llm_calls,
                "total_tokens": self.total_tokens,
                "tokens_per_annotation": self.tokens_per_annotation,
                "latency_ms": self.latency_ms,
                "retries": self.retries,
            },
            "coverage": {
                "document_coverage": self.document_coverage,
                "line_coverage": self.line_coverage,
            },
        }


def compute_metrics(result: AnnotationResult) -> EvaluationMetrics:
    """
    Compute evaluation metrics for an annotation result.
    """
    doc = result.document
    annotations = result.annotations

    metrics = EvaluationMetrics()

    # Preservation metrics
    metrics.document_preserved = result.document_preserved
    metrics.modification_count = len(result.preservation_errors)

    if result.document_preserved:
        metrics.preservation_rate = 1.0
    else:
        # Estimate preservation rate from errors
        # This is a rough estimate - actual rate depends on error severity
        metrics.preservation_rate = max(0.0, 1.0 - (metrics.modification_count * 0.1))

    # Annotation quality metrics
    metrics.annotation_count = len(annotations)

    valid_positions = 0
    with_anchors = 0
    for ann in annotations:
        # Check if position is valid
        if 0 <= ann.position.offset <= len(doc.content):
            valid_positions += 1

        # Check if has anchor
        if ann.anchor_text:
            with_anchors += 1

    metrics.annotations_with_valid_positions = valid_positions
    metrics.annotations_with_anchors = with_anchors

    if metrics.annotation_count > 0:
        metrics.position_accuracy = valid_positions / metrics.annotation_count
    else:
        metrics.position_accuracy = 1.0

    # Efficiency metrics
    metrics.llm_calls = result.llm_calls
    metrics.total_tokens = result.total_tokens
    metrics.latency_ms = result.latency_ms
    metrics.retries = result.retries

    if metrics.annotation_count > 0:
        metrics.tokens_per_annotation = metrics.total_tokens / metrics.annotation_count
    else:
        metrics.tokens_per_annotation = float(metrics.total_tokens)

    # Coverage metrics
    if annotations and doc.content:
        # Calculate what fraction of document has annotations nearby
        annotated_regions = set()
        for ann in annotations:
            # Mark a region around each annotation
            start = max(0, ann.position.offset - 50)
            end = min(len(doc.content), ann.position.offset + 50)
            for i in range(start, end):
                annotated_regions.add(i)

        metrics.document_coverage = len(annotated_regions) / len(doc.content)

        # Calculate line coverage
        annotated_lines = set(ann.position.line for ann in annotations)
        metrics.line_coverage = len(annotated_lines) / doc.line_count if doc.line_count > 0 else 0.0
    else:
        metrics.document_coverage = 0.0
        metrics.line_coverage = 0.0

    return metrics


@dataclass
class ComparisonResult:
    """Result of comparing multiple annotation strategies."""
    strategy_metrics: dict[str, EvaluationMetrics]
    rankings: dict[str, dict[str, int]]  # metric -> strategy -> rank
    best_overall: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "strategies": {
                name: m.to_dict() for name, m in self.strategy_metrics.items()
            },
            "rankings": self.rankings,
            "best_overall": self.best_overall,
            "summary": self.summary,
        }


def compare_results(
    results: dict[str, AnnotationResult],
    weights: Optional[dict[str, float]] = None,
) -> ComparisonResult:
    """
    Compare annotation results from multiple strategies.

    Args:
        results: Dictionary mapping strategy names to results
        weights: Optional weights for different metrics in scoring

    Returns:
        ComparisonResult with rankings and analysis
    """
    default_weights = {
        "preservation": 0.4,  # Most important - document must be preserved
        "quality": 0.25,
        "efficiency": 0.2,
        "coverage": 0.15,
    }
    weights = weights or default_weights

    # Compute metrics for each strategy
    strategy_metrics = {}
    for name, result in results.items():
        strategy_metrics[name] = compute_metrics(result)

    # Rank strategies on each metric
    rankings = {}

    # Preservation ranking (higher is better)
    preservation_scores = {
        name: (1 if m.document_preserved else 0) + m.preservation_rate
        for name, m in strategy_metrics.items()
    }
    rankings["preservation"] = _rank_dict(preservation_scores, higher_is_better=True)

    # Quality ranking (more annotations with valid positions is better)
    quality_scores = {
        name: m.annotation_count * m.position_accuracy
        for name, m in strategy_metrics.items()
    }
    rankings["quality"] = _rank_dict(quality_scores, higher_is_better=True)

    # Efficiency ranking (fewer tokens is better)
    efficiency_scores = {
        name: m.total_tokens if m.total_tokens > 0 else float('inf')
        for name, m in strategy_metrics.items()
    }
    rankings["efficiency"] = _rank_dict(efficiency_scores, higher_is_better=False)

    # Coverage ranking (higher is better)
    coverage_scores = {
        name: m.document_coverage + m.line_coverage
        for name, m in strategy_metrics.items()
    }
    rankings["coverage"] = _rank_dict(coverage_scores, higher_is_better=True)

    # Compute overall scores
    overall_scores = {}
    for name in strategy_metrics.keys():
        score = 0.0
        for metric, weight in weights.items():
            if metric in rankings:
                # Convert rank to score (lower rank = higher score)
                num_strategies = len(strategy_metrics)
                rank = rankings[metric].get(name, num_strategies)
                metric_score = (num_strategies - rank + 1) / num_strategies
                score += weight * metric_score
        overall_scores[name] = score

    # Find best overall
    best_overall = max(overall_scores.items(), key=lambda x: x[1])[0]

    # Generate summary
    summary_lines = ["Strategy Comparison Summary:", ""]

    # Preservation results
    preserved_strategies = [
        name for name, m in strategy_metrics.items() if m.document_preserved
    ]
    summary_lines.append(f"Strategies that preserved document: {len(preserved_strategies)}/{len(strategy_metrics)}")
    summary_lines.append(f"  Preserved: {', '.join(preserved_strategies) or 'None'}")
    summary_lines.append("")

    # Best in each category
    for metric, ranks in rankings.items():
        best_in_category = min(ranks.items(), key=lambda x: x[1])[0]
        summary_lines.append(f"Best {metric}: {best_in_category}")

    summary_lines.append("")
    summary_lines.append(f"Best overall: {best_overall}")

    return ComparisonResult(
        strategy_metrics=strategy_metrics,
        rankings=rankings,
        best_overall=best_overall,
        summary='\n'.join(summary_lines),
    )


def _rank_dict(scores: dict[str, float], higher_is_better: bool) -> dict[str, int]:
    """Rank items in a dictionary by score."""
    sorted_items = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=higher_is_better,
    )
    return {name: rank + 1 for rank, (name, _) in enumerate(sorted_items)}

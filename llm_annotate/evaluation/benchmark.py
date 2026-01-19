"""
Benchmarking framework for annotation strategies.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..core import Document, AnnotationResult
from ..llm_client import LLMClient
from ..strategies import list_strategies, get_strategy
from .metrics import EvaluationMetrics, compute_metrics, compare_results, ComparisonResult


@dataclass
class BenchmarkDocument:
    """A document used in benchmarking."""
    name: str
    content: str
    category: str = "general"
    expected_annotations: int = 0  # Expected number of annotations (for quality check)
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    strategies: list[str] = field(default_factory=list)  # Empty = all strategies
    documents: list[BenchmarkDocument] = field(default_factory=list)
    instructions: str = "Add helpful annotations explaining the content."
    annotation_types: list[str] = field(default_factory=lambda: ["comment", "explanation"])
    runs_per_strategy: int = 1  # Multiple runs for consistency check
    save_results: bool = True
    output_dir: str = "benchmark_results"


@dataclass
class StrategyBenchmarkResult:
    """Results for a single strategy on a single document."""
    strategy_name: str
    document_name: str
    result: AnnotationResult
    metrics: EvaluationMetrics
    run_number: int = 1


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    config: BenchmarkConfig
    strategy_results: dict[str, dict[str, list[StrategyBenchmarkResult]]]  # strategy -> doc -> runs
    comparisons: dict[str, ComparisonResult]  # per-document comparisons
    overall_comparison: ComparisonResult
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "total_time_seconds": self.total_time_seconds,
            "config": {
                "strategies": self.config.strategies or list_strategies(),
                "documents": [d.name for d in self.config.documents],
                "instructions": self.config.instructions,
                "runs_per_strategy": self.config.runs_per_strategy,
            },
            "per_document_comparisons": {
                doc: comp.to_dict() for doc, comp in self.comparisons.items()
            },
            "overall_comparison": self.overall_comparison.to_dict(),
            "detailed_results": self._format_detailed_results(),
        }

    def _format_detailed_results(self) -> dict:
        """Format detailed results for each strategy/document."""
        detailed = {}
        for strategy, doc_results in self.strategy_results.items():
            detailed[strategy] = {}
            for doc, runs in doc_results.items():
                detailed[strategy][doc] = [
                    {
                        "run": r.run_number,
                        "preserved": r.result.document_preserved,
                        "annotations": r.result.annotation_count if hasattr(r.result, 'annotation_count') else len(r.result.annotations),
                        "tokens": r.result.total_tokens,
                        "retries": r.result.retries,
                        "latency_ms": r.result.latency_ms,
                    }
                    for r in runs
                ]
        return detailed

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total time: {self.total_time_seconds:.2f}s")
        print(f"Documents tested: {len(self.config.documents)}")
        print(f"Strategies tested: {len(self.config.strategies) or len(list_strategies())}")
        print()

        print(self.overall_comparison.summary)
        print()

        # Per-strategy summary
        print("\nPer-Strategy Results:")
        print("-" * 60)

        for strategy in self.strategy_results.keys():
            total_preserved = 0
            total_docs = 0
            total_annotations = 0
            total_tokens = 0

            for doc, runs in self.strategy_results[strategy].items():
                for run in runs:
                    total_docs += 1
                    if run.result.document_preserved:
                        total_preserved += 1
                    total_annotations += len(run.result.annotations)
                    total_tokens += run.result.total_tokens

            print(f"\n{strategy}:")
            print(f"  Preservation rate: {total_preserved}/{total_docs} ({100*total_preserved/total_docs:.1f}%)")
            print(f"  Total annotations: {total_annotations}")
            print(f"  Total tokens used: {total_tokens}")


class Benchmark:
    """
    Benchmark runner for comparing annotation strategies.
    """

    def __init__(self, client: LLMClient, config: Optional[BenchmarkConfig] = None):
        """
        Initialize the benchmark.

        Args:
            client: LLM client to use
            config: Benchmark configuration
        """
        self.client = client
        self.config = config or BenchmarkConfig()

    def add_document(
        self,
        content: str,
        name: str,
        category: str = "general",
        expected_annotations: int = 0,
    ):
        """Add a document to the benchmark."""
        self.config.documents.append(
            BenchmarkDocument(
                name=name,
                content=content,
                category=category,
                expected_annotations=expected_annotations,
            )
        )

    def add_document_from_file(self, path: str, category: str = "general"):
        """Add a document from a file."""
        file_path = Path(path)
        content = file_path.read_text()
        self.add_document(
            content=content,
            name=file_path.name,
            category=category,
        )

    def run(self) -> BenchmarkResult:
        """Run the full benchmark."""
        start_time = time.time()

        strategies = self.config.strategies or list_strategies()
        documents = self.config.documents

        if not documents:
            raise ValueError("No documents to benchmark. Add documents first.")

        strategy_results: dict[str, dict[str, list[StrategyBenchmarkResult]]] = {
            s: {} for s in strategies
        }

        # Run each strategy on each document
        for strategy_name in strategies:
            print(f"\nBenchmarking strategy: {strategy_name}")

            for doc in documents:
                print(f"  Document: {doc.name}")
                strategy_results[strategy_name][doc.name] = []

                for run_num in range(1, self.config.runs_per_strategy + 1):
                    try:
                        result = self._run_single(strategy_name, doc)
                        metrics = compute_metrics(result)

                        strategy_results[strategy_name][doc.name].append(
                            StrategyBenchmarkResult(
                                strategy_name=strategy_name,
                                document_name=doc.name,
                                result=result,
                                metrics=metrics,
                                run_number=run_num,
                            )
                        )

                        status = "✓" if result.document_preserved else "✗"
                        print(f"    Run {run_num}: {status} ({len(result.annotations)} annotations)")

                    except Exception as e:
                        print(f"    Run {run_num}: ERROR - {str(e)}")
                        # Create error result
                        error_result = AnnotationResult(
                            document=Document(content=doc.content, name=doc.name),
                            annotations=[],
                            strategy_name=strategy_name,
                            document_preserved=False,
                            preservation_errors=[str(e)],
                        )
                        strategy_results[strategy_name][doc.name].append(
                            StrategyBenchmarkResult(
                                strategy_name=strategy_name,
                                document_name=doc.name,
                                result=error_result,
                                metrics=compute_metrics(error_result),
                                run_number=run_num,
                            )
                        )

        # Compute per-document comparisons
        per_doc_comparisons = {}
        for doc in documents:
            doc_results = {}
            for strategy in strategies:
                runs = strategy_results[strategy][doc.name]
                if runs:
                    # Use first run for comparison
                    doc_results[strategy] = runs[0].result
            if doc_results:
                per_doc_comparisons[doc.name] = compare_results(doc_results)

        # Compute overall comparison (aggregate across documents)
        overall_results = {}
        for strategy in strategies:
            # Aggregate results across all documents
            all_annotations = []
            total_tokens = 0
            total_latency = 0.0
            total_retries = 0
            total_calls = 0
            all_preserved = True
            all_errors = []

            for doc in documents:
                runs = strategy_results[strategy][doc.name]
                for run in runs:
                    all_annotations.extend(run.result.annotations)
                    total_tokens += run.result.total_tokens
                    total_latency += run.result.latency_ms
                    total_retries += run.result.retries
                    total_calls += run.result.llm_calls
                    if not run.result.document_preserved:
                        all_preserved = False
                        all_errors.extend(run.result.preservation_errors)

            # Create aggregate result
            overall_results[strategy] = AnnotationResult(
                document=Document(content="", name="aggregate"),
                annotations=all_annotations,
                strategy_name=strategy,
                llm_calls=total_calls,
                total_tokens=total_tokens,
                latency_ms=total_latency,
                retries=total_retries,
                document_preserved=all_preserved,
                preservation_errors=all_errors[:10],  # Limit errors
            )

        overall_comparison = compare_results(overall_results)

        total_time = time.time() - start_time

        result = BenchmarkResult(
            config=self.config,
            strategy_results=strategy_results,
            comparisons=per_doc_comparisons,
            overall_comparison=overall_comparison,
            total_time_seconds=total_time,
        )

        # Save results if configured
        if self.config.save_results:
            self._save_results(result)

        return result

    def _run_single(self, strategy_name: str, doc: BenchmarkDocument) -> AnnotationResult:
        """Run a single strategy on a single document."""
        strategy = get_strategy(strategy_name)
        document = Document(content=doc.content, name=doc.name, metadata=doc.metadata)

        return strategy.annotate(
            document=document,
            client=self.client,
            instructions=self.config.instructions,
            annotation_types=self.config.annotation_types,
        )

    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\nResults saved to: {json_path}")


def run_benchmark(
    client: LLMClient,
    documents: list[tuple[str, str]],  # (name, content) pairs
    instructions: str = "Add helpful annotations explaining the content.",
    strategies: Optional[list[str]] = None,
    annotation_types: Optional[list[str]] = None,
) -> BenchmarkResult:
    """
    Convenience function to run a benchmark.

    Args:
        client: LLM client
        documents: List of (name, content) tuples
        instructions: Annotation instructions
        strategies: Strategies to test (None = all)
        annotation_types: Types of annotations

    Returns:
        BenchmarkResult
    """
    config = BenchmarkConfig(
        strategies=strategies or [],
        instructions=instructions,
        annotation_types=annotation_types or ["comment", "explanation"],
    )

    benchmark = Benchmark(client=client, config=config)

    for name, content in documents:
        benchmark.add_document(content=content, name=name)

    return benchmark.run()

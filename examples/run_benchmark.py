#!/usr/bin/env python3
"""
Run a benchmark comparing all annotation strategies.

This script loads test documents and runs each strategy on them,
producing a detailed comparison report.

Requirements:
    pip install llm-annotate[anthropic]
    export ANTHROPIC_API_KEY=your-key
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_annotate.llm_client import get_client
from llm_annotate.evaluation import Benchmark, BenchmarkConfig


# Sample documents for benchmarking
SAMPLE_DOCUMENTS = [
    {
        "name": "simple_prose.txt",
        "category": "prose",
        "content": """
The art of writing clean code is not about following rigid rules, but about
communicating clearly with other developers. When we write code, we are not
just telling the computer what to do - we are telling a story that other
humans need to understand.

Good variable names, consistent formatting, and logical organization all
contribute to readability. Comments should explain why, not what. The code
itself should be clear enough to explain what it does.
        """.strip(),
    },
    {
        "name": "python_function.py",
        "category": "code",
        "content": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
        """.strip(),
    },
    {
        "name": "api_docs.md",
        "category": "documentation",
        "content": """
# User API

## GET /users/{id}

Retrieves a user by their unique identifier.

### Parameters

| Name | Type | Description |
|------|------|-------------|
| id   | int  | User ID     |

### Response

```json
{
    "id": 123,
    "name": "John Doe",
    "email": "john@example.com"
}
```

### Errors

- 404: User not found
- 401: Unauthorized
        """.strip(),
    },
]


def run_full_benchmark():
    """Run benchmark on all strategies with sample documents."""
    print("=" * 70)
    print("LLM ANNOTATION STRATEGY BENCHMARK")
    print("=" * 70)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\nError: No API key found.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Determine provider
    provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    print(f"\nUsing provider: {provider}")

    # Create client
    client = get_client(provider)

    # Configure benchmark
    config = BenchmarkConfig(
        strategies=[],  # Empty = all strategies
        instructions="Add helpful annotations explaining the content. "
                    "For code, explain the algorithm. For prose, highlight key points. "
                    "For documentation, note any potential issues or improvements.",
        annotation_types=["comment", "explanation", "warning", "suggestion"],
        runs_per_strategy=1,
        save_results=True,
        output_dir="benchmark_results",
    )

    # Create benchmark
    benchmark = Benchmark(client=client, config=config)

    # Add documents
    for doc in SAMPLE_DOCUMENTS:
        benchmark.add_document(
            content=doc["content"],
            name=doc["name"],
            category=doc["category"],
        )

    print(f"\nDocuments: {len(SAMPLE_DOCUMENTS)}")
    print(f"Strategies: all ({len(benchmark.config.strategies) or 8})")
    print("\nStarting benchmark...")
    print("-" * 70)

    # Run benchmark
    result = benchmark.run()

    # Print detailed summary
    result.print_summary()

    # Print per-document breakdown
    print("\n" + "=" * 70)
    print("PER-DOCUMENT RESULTS")
    print("=" * 70)

    for doc_name, comparison in result.comparisons.items():
        print(f"\n{doc_name}:")
        print(f"  Best strategy: {comparison.best_overall}")
        print(f"  Strategies that preserved text: ", end="")

        preserved = [
            name for name, metrics in comparison.strategy_metrics.items()
            if metrics.document_preserved
        ]
        print(f"{len(preserved)}/{len(comparison.strategy_metrics)}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: benchmark_results/")


def run_quick_test():
    """Run a quick test with just a few strategies."""
    print("Running quick test with subset of strategies...")

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("No API key found. Cannot run test.")
        return

    provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    client = get_client(provider)

    config = BenchmarkConfig(
        strategies=["offset_based", "line_reference", "anchor_based"],
        instructions="Briefly annotate this text.",
        runs_per_strategy=1,
        save_results=False,
    )

    benchmark = Benchmark(client=client, config=config)
    benchmark.add_document(
        content="Hello, World! This is a test document.",
        name="quick_test.txt",
    )

    result = benchmark.run()
    result.print_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run annotation strategy benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with fewer strategies",
    )
    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_full_benchmark()

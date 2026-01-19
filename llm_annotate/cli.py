"""
Command-line interface for the LLM annotation library.
"""

import argparse
import json
import sys
from pathlib import Path

from .core import Document
from .annotator import Annotator
from .llm_client import get_client
from .strategies import list_strategies
from .evaluation import Benchmark, BenchmarkConfig, run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="LLM Document Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a file
  llm-annotate annotate document.txt --strategy offset_based

  # Run benchmark on all strategies
  llm-annotate benchmark --documents doc1.txt doc2.py

  # List available strategies
  llm-annotate list-strategies

  # Compare strategies on a document
  llm-annotate compare document.txt --strategies baseline_naive xml_inline offset_based
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Annotate command
    annotate_parser = subparsers.add_parser("annotate", help="Annotate a document")
    annotate_parser.add_argument("file", help="File to annotate")
    annotate_parser.add_argument(
        "--strategy", "-s",
        default="offset_based",
        help="Annotation strategy to use",
    )
    annotate_parser.add_argument(
        "--instructions", "-i",
        default="Add helpful annotations explaining the content.",
        help="Instructions for the annotator",
    )
    annotate_parser.add_argument(
        "--types", "-t",
        nargs="+",
        default=["comment", "explanation"],
        help="Types of annotations to generate",
    )
    annotate_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    annotate_parser.add_argument(
        "--format", "-f",
        choices=["inline", "margin", "json"],
        default="json",
        help="Output format",
    )
    annotate_parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider",
    )
    annotate_parser.add_argument(
        "--model",
        help="Model to use (provider-specific)",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    benchmark_parser.add_argument(
        "--documents", "-d",
        nargs="+",
        required=True,
        help="Documents to benchmark",
    )
    benchmark_parser.add_argument(
        "--strategies", "-s",
        nargs="+",
        help="Strategies to test (default: all)",
    )
    benchmark_parser.add_argument(
        "--instructions", "-i",
        default="Add helpful annotations explaining the content.",
        help="Instructions for annotation",
    )
    benchmark_parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of runs per strategy",
    )
    benchmark_parser.add_argument(
        "--output-dir", "-o",
        default="benchmark_results",
        help="Directory for results",
    )
    benchmark_parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare strategies")
    compare_parser.add_argument("file", help="File to annotate")
    compare_parser.add_argument(
        "--strategies", "-s",
        nargs="+",
        help="Strategies to compare (default: all)",
    )
    compare_parser.add_argument(
        "--instructions", "-i",
        default="Add helpful annotations explaining the content.",
        help="Instructions for annotation",
    )
    compare_parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider",
    )

    # List strategies command
    list_parser = subparsers.add_parser("list-strategies", help="List available strategies")

    args = parser.parse_args()

    if args.command == "annotate":
        run_annotate(args)
    elif args.command == "benchmark":
        run_benchmark_cmd(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "list-strategies":
        run_list_strategies()
    else:
        parser.print_help()


def run_annotate(args):
    """Run annotation on a single file."""
    # Read document
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text()
    document = Document(content=content, name=file_path.name)

    # Create client
    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model

    try:
        client = get_client(args.provider, **client_kwargs)
    except Exception as e:
        print(f"Error creating LLM client: {e}", file=sys.stderr)
        sys.exit(1)

    # Create annotator and run
    annotator = Annotator(client=client, default_strategy=args.strategy)

    print(f"Annotating {args.file} with strategy: {args.strategy}", file=sys.stderr)

    try:
        result = annotator.annotate(
            document=document,
            instructions=args.instructions,
            annotation_types=args.types,
        )
    except Exception as e:
        print(f"Error during annotation: {e}", file=sys.stderr)
        sys.exit(1)

    # Output result
    output = result.render_annotated(format=args.format)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    # Print summary
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Document preserved: {result.document_preserved}", file=sys.stderr)
    print(f"  Annotations: {len(result.annotations)}", file=sys.stderr)
    print(f"  LLM calls: {result.llm_calls}", file=sys.stderr)
    print(f"  Tokens: {result.total_tokens}", file=sys.stderr)


def run_benchmark_cmd(args):
    """Run benchmark on multiple documents."""
    # Create client
    try:
        client = get_client(args.provider)
    except Exception as e:
        print(f"Error creating LLM client: {e}", file=sys.stderr)
        sys.exit(1)

    # Create benchmark config
    config = BenchmarkConfig(
        strategies=args.strategies or [],
        instructions=args.instructions,
        runs_per_strategy=args.runs,
        output_dir=args.output_dir,
    )

    benchmark = Benchmark(client=client, config=config)

    # Add documents
    for doc_path in args.documents:
        path = Path(doc_path)
        if not path.exists():
            print(f"Warning: File not found: {doc_path}", file=sys.stderr)
            continue
        benchmark.add_document_from_file(str(path))

    if not benchmark.config.documents:
        print("Error: No valid documents to benchmark", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    print("Starting benchmark...", file=sys.stderr)
    result = benchmark.run()
    result.print_summary()


def run_compare(args):
    """Compare strategies on a single document."""
    # Read document
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text()
    document = Document(content=content, name=file_path.name)

    # Create client
    try:
        client = get_client(args.provider)
    except Exception as e:
        print(f"Error creating LLM client: {e}", file=sys.stderr)
        sys.exit(1)

    # Create annotator
    annotator = Annotator(client=client)

    strategies = args.strategies or list_strategies()

    print(f"Comparing {len(strategies)} strategies on: {args.file}", file=sys.stderr)

    # Run all strategies
    results = annotator.annotate_with_all_strategies(
        document=document,
        instructions=args.instructions,
        strategies=strategies,
    )

    # Compare and print results
    from .evaluation import compare_results

    comparison = compare_results(results)

    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(comparison.summary)

    print("\nDetailed Results:")
    print("-" * 60)

    for strategy, result in results.items():
        preserved_mark = "✓" if result.document_preserved else "✗"
        print(f"\n{strategy}:")
        print(f"  Preserved: {preserved_mark}")
        print(f"  Annotations: {len(result.annotations)}")
        print(f"  Tokens: {result.total_tokens}")
        print(f"  Retries: {result.retries}")
        if result.preservation_errors:
            print(f"  Errors: {result.preservation_errors[:2]}")


def run_list_strategies():
    """List available annotation strategies."""
    from .strategies import STRATEGIES

    print("Available Annotation Strategies:")
    print("-" * 40)

    for name, cls in STRATEGIES.items():
        print(f"\n{name}")
        print(f"  {cls.description}")


if __name__ == "__main__":
    main()

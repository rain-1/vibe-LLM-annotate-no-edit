#!/usr/bin/env python3
"""
Basic usage examples for the LLM Annotate library.

This script demonstrates how to use different annotation strategies
and compare their results.

Requirements:
    pip install llm-annotate[anthropic]  # or [openai] for OpenAI
    export ANTHROPIC_API_KEY=your-key   # or OPENAI_API_KEY
"""

import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_annotate import Document, Annotator
from llm_annotate.strategies import list_strategies


def example_single_strategy():
    """Example: Annotate a document with a single strategy."""
    print("=" * 60)
    print("Example 1: Single Strategy Annotation")
    print("=" * 60)

    # Create a document
    document = Document(
        content="""
def calculate_total(items, tax_rate=0.08):
    subtotal = sum(item.price * item.quantity for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax
        """.strip(),
        name="shopping_cart.py"
    )

    # Create annotator (uses Anthropic by default)
    annotator = Annotator(default_strategy="offset_based")

    # Annotate with specific instructions
    result = annotator.annotate(
        document=document,
        instructions="Explain what each part of this function does. Focus on the algorithm and edge cases.",
        annotation_types=["explanation", "warning"],
    )

    print(f"\nDocument preserved: {result.document_preserved}")
    print(f"Annotations found: {len(result.annotations)}")
    print(f"LLM calls made: {result.llm_calls}")
    print(f"Tokens used: {result.total_tokens}")

    print("\nAnnotations:")
    for i, ann in enumerate(result.annotations, 1):
        print(f"\n{i}. [{ann.annotation_type.value}] at line {ann.position.line + 1}")
        print(f"   {ann.content}")


def example_compare_strategies():
    """Example: Compare multiple strategies on the same document."""
    print("\n" + "=" * 60)
    print("Example 2: Compare Annotation Strategies")
    print("=" * 60)

    document = Document(
        content="""
The Internet of Things (IoT) refers to the network of physical devices,
vehicles, home appliances, and other items embedded with electronics,
software, sensors, and connectivity. These devices can collect and
exchange data over the internet without human intervention.

Key challenges in IoT include security vulnerabilities, privacy concerns,
and the need for standardized protocols. Despite these challenges, IoT
adoption continues to grow across industries.
        """.strip(),
        name="iot_overview.txt"
    )

    annotator = Annotator()

    # Compare a subset of strategies
    strategies_to_compare = [
        "offset_based",
        "line_reference",
        "anchor_based",
    ]

    print(f"\nComparing strategies: {', '.join(strategies_to_compare)}")

    results = annotator.annotate_with_all_strategies(
        document=document,
        instructions="Add annotations explaining technical terms and highlighting important points.",
        strategies=strategies_to_compare,
    )

    # Print comparison
    print("\nResults:")
    print("-" * 60)

    for strategy_name, result in results.items():
        preserved = "Yes" if result.document_preserved else "NO"
        print(f"\n{strategy_name}:")
        print(f"  Preserved document: {preserved}")
        print(f"  Annotations: {len(result.annotations)}")
        print(f"  Tokens: {result.total_tokens}")
        print(f"  Retries: {result.retries}")


def example_list_strategies():
    """Example: List all available strategies."""
    print("\n" + "=" * 60)
    print("Example 3: Available Strategies")
    print("=" * 60)

    strategies = list_strategies()
    print(f"\nFound {len(strategies)} strategies:")

    for name in strategies:
        print(f"  - {name}")


def example_render_formats():
    """Example: Different output formats for annotated documents."""
    print("\n" + "=" * 60)
    print("Example 4: Output Formats")
    print("=" * 60)

    # Using mock data for demonstration
    from llm_annotate.core import Annotation, Position, AnnotationType, AnnotationResult

    document = Document(
        content="Hello, World!\nThis is a test.",
        name="test.txt"
    )

    # Create mock annotations
    annotations = [
        Annotation(
            content="Greeting to the world",
            position=Position(offset=0, line=0, column=0),
            annotation_type=AnnotationType.COMMENT,
        ),
        Annotation(
            content="Test statement",
            position=Position(offset=14, line=1, column=0),
            annotation_type=AnnotationType.EXPLANATION,
        ),
    ]

    result = AnnotationResult(
        document=document,
        annotations=annotations,
        strategy_name="example",
    )

    print("\nInline format:")
    print("-" * 40)
    print(result.render_annotated(format="inline"))

    print("\nMargin format:")
    print("-" * 40)
    print(result.render_annotated(format="margin"))

    print("\nJSON format (truncated):")
    print("-" * 40)
    json_output = result.render_annotated(format="json")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)


def main():
    """Run all examples."""
    print("LLM Annotate Library - Usage Examples")
    print("=" * 60)

    # Check for API key
    has_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    has_openai = os.environ.get("OPENAI_API_KEY")

    if not has_anthropic and not has_openai:
        print("\nNote: No API key found. Running examples that don't require LLM calls.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run full examples.\n")

        example_list_strategies()
        example_render_formats()
        return

    try:
        example_single_strategy()
        example_compare_strategies()
        example_list_strategies()
        example_render_formats()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying examples that don't require API calls...")
        example_list_strategies()
        example_render_formats()


if __name__ == "__main__":
    main()

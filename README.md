# LLM Document Annotation (Insertion-Only)

![](/unnamed.jpg)

A library for using LLMs to annotate documents with **guaranteed preservation** of the original text. Annotations are insertion-only — the original document is never modified.

## The Problem

When you ask an LLM to "annotate" a document, it often:
- "Fixes" typos or grammar
- Rewrites sentences for clarity
- Changes formatting
- Modifies code while adding comments

This library solves that problem by implementing and benchmarking **8 different strategies** for getting annotations from LLMs while ensuring the original document remains untouched.

## Annotation Strategies

| Strategy | Description | Preserves? | Best For |
|----------|-------------|------------|----------|
| `baseline_naive` | Ask LLM to annotate inline | Often fails | Baseline comparison |
| `xml_inline` | Use XML tags with verification | Good | Structured docs |
| `marker_referenced` | Insert `[[1]]` markers, JSON annotations | Good | Minimal changes |
| `offset_based` | Annotations reference char offsets | Excellent | Precise positioning |
| `line_reference` | Annotations reference line:column | Excellent | Code annotation |
| `chunked_verified` | Split & verify each chunk | Good | Large documents |
| `diff_insertion_only` | Unified diff with only `+` lines | Good | Code-savvy users |
| `anchor_based` | Annotations attached to unique text | Excellent | Robust to changes |

## Installation

```bash
# Basic installation
pip install llm-annotate

# With Anthropic support
pip install llm-annotate[anthropic]

# With OpenAI support
pip install llm-annotate[openai]

# With both
pip install llm-annotate[all]
```

## Quick Start

```python
from llm_annotate import Document, Annotator

# Create a document
doc = Document(
    content="def hello(): print('Hello, World!')",
    name="example.py"
)

# Create annotator (uses Anthropic by default)
annotator = Annotator(default_strategy="offset_based")

# Annotate
result = annotator.annotate(
    document=doc,
    instructions="Explain what this code does",
)

# Check results
print(f"Document preserved: {result.document_preserved}")
print(f"Annotations: {len(result.annotations)}")

# Render annotated output
print(result.render_annotated(format="inline"))
```

## Strategies in Detail

### 1. Baseline Naive (`baseline_naive`)

Simply asks the LLM to insert annotations inline. Serves as a baseline — often fails to preserve the original text.

```python
annotator = Annotator(default_strategy="baseline_naive")
```

### 2. XML Inline (`xml_inline`)

Uses XML tags for annotations with structured verification.

```python
annotator = Annotator(default_strategy="xml_inline")
# Output: <ann type="comment">explanation</ann>
```

### 3. Marker Referenced (`marker_referenced`)

Inserts lightweight markers and provides annotations in JSON.

```python
annotator = Annotator(default_strategy="marker_referenced")
# Document: "The [[1]]quick brown fox"
# JSON: {"annotations": [{"id": 1, "content": "..."}]}
```

### 4. Offset Based (`offset_based`)

Annotations specify character offsets — document is never modified.

```python
annotator = Annotator(default_strategy="offset_based")
# {"offset": 10, "content": "This is at character 10"}
```

### 5. Line Reference (`line_reference`)

Annotations reference line numbers and columns.

```python
annotator = Annotator(default_strategy="line_reference")
# {"line": 5, "column": 0, "content": "Line 5 annotation"}
```

### 6. Chunked Verified (`chunked_verified`)

Splits document into chunks, annotates each with verification and retry.

```python
from llm_annotate.strategies import ChunkedVerifiedStrategy

strategy = ChunkedVerifiedStrategy(
    chunk_by="paragraph",  # or "sentence"
    max_chunk_size=500,
)
```

### 7. Diff Insertion Only (`diff_insertion_only`)

Produces unified diff with only additions — any deletions cause rejection.

```python
annotator = Annotator(default_strategy="diff_insertion_only")
# @@ -5,3 +5,4 @@
#  context
# +/* annotation */
#  context
```

### 8. Anchor Based (`anchor_based`)

Annotations attached to unique text snippets.

```python
annotator = Annotator(default_strategy="anchor_based")
# {"anchor": "quick brown fox", "content": "..."}
```

## Comparing Strategies

```python
from llm_annotate import Annotator, Document

annotator = Annotator()

# Compare all strategies on a document
results = annotator.annotate_with_all_strategies(
    document=Document(content="Your text here"),
    instructions="Add helpful annotations",
)

for strategy_name, result in results.items():
    print(f"{strategy_name}: preserved={result.document_preserved}")
```

## Benchmarking

Run comprehensive benchmarks to find the best strategy for your use case:

```python
from llm_annotate.llm_client import get_client
from llm_annotate.evaluation import Benchmark, BenchmarkConfig

client = get_client("anthropic")

config = BenchmarkConfig(
    strategies=[],  # Empty = all strategies
    instructions="Annotate the document",
    runs_per_strategy=3,  # Multiple runs for consistency
)

benchmark = Benchmark(client=client, config=config)
benchmark.add_document(content="...", name="doc1.txt")
benchmark.add_document_from_file("document.py")

result = benchmark.run()
result.print_summary()
```

## CLI Usage

```bash
# Annotate a file
llm-annotate annotate document.txt --strategy offset_based

# Compare strategies
llm-annotate compare document.txt --strategies offset_based anchor_based

# Run benchmark
llm-annotate benchmark --documents doc1.txt doc2.py doc3.md

# List strategies
llm-annotate list-strategies
```

## Evaluation Metrics

The library evaluates strategies on:

- **Preservation Rate**: % of original text preserved
- **Annotation Count**: Number of annotations generated
- **Position Accuracy**: Are annotations at valid positions?
- **Token Efficiency**: Tokens used per annotation
- **Retry Count**: How many retries needed
- **Latency**: Time to complete

## Output Formats

```python
# Inline comments
result.render_annotated(format="inline")
# Output: "text /* [comment] annotation */ more text"

# Margin notes
result.render_annotated(format="margin")
# Output: "text                    // [comment] annotation"

# JSON structure
result.render_annotated(format="json")
# Output: {"annotations": [...], "metrics": {...}}
```

## Configuration

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your-key
# or
export OPENAI_API_KEY=your-key
```

### Using Different Providers

```python
# Anthropic (default)
annotator = Annotator(provider="anthropic", model="claude-sonnet-4-20250514")

# OpenAI
annotator = Annotator(provider="openai", model="gpt-4o")
```

## Development

```bash
# Clone the repository
git clone https://github.com/rain-1/vibe-LLM-annotate-no-edit.git
cd vibe-LLM-annotate-no-edit

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run examples
python examples/basic_usage.py
python examples/run_benchmark.py --quick
```

## Project Structure

```
llm_annotate/
├── __init__.py          # Package exports
├── core.py              # Document, Annotation, Position classes
├── annotator.py         # Main Annotator class
├── llm_client.py        # LLM client interfaces
├── cli.py               # Command-line interface
├── strategies/          # Annotation strategies
│   ├── base.py
│   ├── baseline_naive.py
│   ├── xml_inline.py
│   ├── marker_referenced.py
│   ├── offset_based.py
│   ├── line_reference.py
│   ├── chunked_verified.py
│   ├── diff_insertion_only.py
│   └── anchor_based.py
└── evaluation/          # Benchmarking framework
    ├── metrics.py
    └── benchmark.py
```

## License

MIT License - see LICENSE file for details.

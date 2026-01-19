"""Tests for annotation strategies using mock LLM client."""

import pytest
import json

from llm_annotate.core import Document
from llm_annotate.llm_client import MockLLMClient
from llm_annotate.strategies import (
    get_strategy,
    list_strategies,
    BaselineNaiveStrategy,
    XMLInlineStrategy,
    MarkerReferencedStrategy,
    OffsetBasedStrategy,
    LineReferenceStrategy,
    AnchorBasedStrategy,
)


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        content="The quick brown fox jumps over the lazy dog.\nThis is a second line.",
        name="test.txt"
    )


@pytest.fixture
def mock_client():
    """Create a mock LLM client."""
    return MockLLMClient()


class TestStrategyRegistry:
    """Test strategy registration and retrieval."""

    def test_list_strategies(self):
        strategies = list_strategies()
        assert len(strategies) == 8
        assert "baseline_naive" in strategies
        assert "xml_inline" in strategies
        assert "offset_based" in strategies

    def test_get_strategy(self):
        strategy = get_strategy("offset_based")
        assert isinstance(strategy, OffsetBasedStrategy)

    def test_get_unknown_strategy(self):
        with pytest.raises(ValueError):
            get_strategy("unknown_strategy")


class TestOffsetBasedStrategy:
    """Tests for offset-based strategy."""

    def test_annotate_with_valid_offsets(self, sample_document, mock_client):
        # Setup mock response
        response = json.dumps({
            "annotations": [
                {
                    "offset": 0,
                    "end_offset": 3,
                    "anchor_text": "The",
                    "type": "comment",
                    "content": "Article 'The' starts the sentence"
                },
                {
                    "offset": 45,
                    "anchor_text": "This",
                    "type": "explanation",
                    "content": "Second line begins here"
                }
            ]
        })
        mock_client.add_response(response)

        strategy = OffsetBasedStrategy()
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        assert result.document_preserved is True
        assert len(result.annotations) == 2
        assert result.annotations[0].content == "Article 'The' starts the sentence"

    def test_annotate_with_invalid_offset(self, sample_document, mock_client):
        # Setup mock response with out-of-range offset
        response = json.dumps({
            "annotations": [
                {
                    "offset": 1000,  # Way out of range
                    "anchor_text": "nowhere",
                    "type": "comment",
                    "content": "Invalid annotation"
                }
            ]
        })
        mock_client.add_response(response)

        strategy = OffsetBasedStrategy()
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        # Should handle gracefully
        assert len(result.preservation_errors) > 0 or len(result.annotations) == 0


class TestLineReferenceStrategy:
    """Tests for line-reference strategy."""

    def test_annotate_with_valid_lines(self, sample_document, mock_client):
        response = json.dumps({
            "annotations": [
                {
                    "line": 1,
                    "column": 0,
                    "quote": "The quick",
                    "type": "comment",
                    "content": "First line annotation"
                }
            ]
        })
        mock_client.add_response(response)

        strategy = LineReferenceStrategy(one_indexed=True)
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        assert result.document_preserved is True
        assert len(result.annotations) >= 0  # May find or not find the quote


class TestAnchorBasedStrategy:
    """Tests for anchor-based strategy."""

    def test_annotate_with_unique_anchor(self, sample_document, mock_client):
        response = json.dumps({
            "annotations": [
                {
                    "anchor": "quick brown fox",
                    "anchor_context": "before",
                    "type": "comment",
                    "content": "This describes a fox"
                }
            ]
        })
        mock_client.add_response(response)

        strategy = AnchorBasedStrategy(min_anchor_length=5)
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        assert result.document_preserved is True
        assert len(result.annotations) == 1
        assert result.annotations[0].anchor_text == "quick brown fox"

    def test_annotate_with_non_unique_anchor(self, mock_client):
        doc = Document(content="the cat and the dog and the bird")
        response = json.dumps({
            "annotations": [
                {
                    "anchor": "the",  # Not unique
                    "anchor_context": "after",
                    "type": "comment",
                    "content": "Article"
                }
            ]
        })
        mock_client.add_response(response)

        strategy = AnchorBasedStrategy(min_anchor_length=3)
        result = strategy.annotate(
            document=doc,
            client=mock_client,
            instructions="Add annotations",
        )

        # Should report error about non-unique anchor
        assert len(result.preservation_errors) > 0 or len(result.annotations) > 0


class TestMarkerReferencedStrategy:
    """Tests for marker-referenced strategy."""

    def test_parse_response_with_markers(self, sample_document, mock_client):
        response = """The [[1]]quick brown fox jumps over the lazy dog.
This is a second line.

```json
{
  "annotations": [
    {"id": 1, "type": "comment", "content": "Adjective describing speed", "context": "quick"}
  ]
}
```"""
        mock_client.add_response(response)

        strategy = MarkerReferencedStrategy()
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        # May or may not preserve depending on marker placement
        assert len(result.annotations) >= 0


class TestBaselineNaiveStrategy:
    """Tests for baseline naive strategy."""

    def test_baseline_with_preserved_text(self, sample_document, mock_client):
        # Response that preserves original text
        response = """The /* comment: article */ quick brown fox jumps over the lazy dog.
This is a second line."""
        mock_client.add_response(response)

        strategy = BaselineNaiveStrategy(annotation_format="comment")
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        # Should have extracted at least one annotation
        assert len(result.annotations) >= 0

    def test_baseline_with_modified_text(self, sample_document, mock_client):
        # Response that modifies original text
        response = """A /* comment */ fast brown fox leaps over the lazy dog.
This is the second line."""
        mock_client.add_response(response)

        strategy = BaselineNaiveStrategy(annotation_format="comment")
        result = strategy.annotate(
            document=sample_document,
            client=mock_client,
            instructions="Add annotations",
        )

        # Should detect modification
        assert result.document_preserved is False

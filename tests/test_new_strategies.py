"""Tests for the new inline_diff_verify and index_labeling strategies."""

import pytest
import json

from llm_annotate.core import Document
from llm_annotate.llm_client import MockLLMClient
from llm_annotate.strategies import (
    InlineDiffVerifyStrategy,
    IndexLabelingStrategy,
)


@pytest.fixture
def dialogue_document():
    """Sample dialogue for speaker attribution testing."""
    return Document(
        content='"Hello," she said. "How are you?"',
        name="dialogue.txt"
    )


@pytest.fixture
def mock_client():
    return MockLLMClient()


class TestInlineDiffVerifyStrategy:
    """Tests for inline diff verification strategy."""

    def test_valid_tag_insertion(self, dialogue_document, mock_client):
        """Test that valid tag insertions are accepted."""
        # Mock response with only tag insertions
        response = '«Narrator»"Hello," «Narrator»she said. «Character»"How are you?"'
        mock_client.add_response(response)

        strategy = InlineDiffVerifyStrategy(tag_start="«", tag_end="»")
        result = strategy.annotate(
            document=dialogue_document,
            client=mock_client,
            instructions="Tag speakers",
        )

        assert result.document_preserved is True
        assert len(result.annotations) == 3

    def test_modification_detected(self, dialogue_document, mock_client):
        """Test that text modifications are caught by diff."""
        # Mock response that modifies original text
        response = '«Narrator»"Hi," «Narrator»she said. «Character»"How are you?"'  # "Hello" -> "Hi"
        mock_client.add_response(response)
        # Add a fallback response that also fails
        mock_client.add_response(response)
        mock_client.add_response(response)

        strategy = InlineDiffVerifyStrategy(
            tag_start="«",
            tag_end="»",
            fallback_on_failure=False,
        )
        result = strategy.annotate(
            document=dialogue_document,
            client=mock_client,
            instructions="Tag speakers",
        )

        assert result.document_preserved is False
        assert len(result.preservation_errors) > 0

    def test_deletion_detected(self, mock_client):
        """Test that deletions are caught by diff."""
        doc = Document(content="The quick brown fox jumps.")

        # Mock response that deletes "brown"
        response = '«Adj»The quick «Noun»fox jumps.'
        mock_client.add_response(response)
        mock_client.add_response(response)
        mock_client.add_response(response)

        strategy = InlineDiffVerifyStrategy(fallback_on_failure=False)
        result = strategy.annotate(
            document=doc,
            client=mock_client,
            instructions="Tag parts of speech",
        )

        assert result.document_preserved is False


class TestIndexLabelingStrategy:
    """Tests for index-based labeling strategy."""

    def test_basic_labeling(self, mock_client):
        """Test basic sentence labeling."""
        doc = Document(content="Hello world. How are you?")

        response = json.dumps({"1": "Greeting", "2": "Question"})
        mock_client.add_response(response)

        strategy = IndexLabelingStrategy(split_by="sentence")
        result = strategy.annotate(
            document=doc,
            client=mock_client,
            instructions="Label each sentence",
        )

        # Should always preserve - LLM never outputs text
        assert result.document_preserved is True
        assert len(result.annotations) == 2
        assert result.annotations[0].content == "Greeting"
        assert result.annotations[1].content == "Question"

    def test_preservation_guaranteed(self, mock_client):
        """Test that preservation is always guaranteed."""
        doc = Document(content="Original text that must be preserved.")

        # Even if LLM returns garbage, text is preserved
        response = json.dumps({"1": "SomeLabel"})
        mock_client.add_response(response)

        strategy = IndexLabelingStrategy()
        result = strategy.annotate(
            document=doc,
            client=mock_client,
            instructions="Label",
        )

        # Always preserved because LLM never outputs text
        assert result.document_preserved is True

    def test_missing_labels_get_default(self, mock_client):
        """Test that missing labels fall back to default."""
        doc = Document(content="First sentence. Second sentence. Third sentence.")

        # Only label some segments
        response = json.dumps({"1": "Intro"})  # Missing 2 and 3
        mock_client.add_response(response)

        strategy = IndexLabelingStrategy(
            split_by="sentence",
            default_label="Narrator",
        )
        result = strategy.annotate(
            document=doc,
            client=mock_client,
            instructions="Label",
        )

        assert result.document_preserved is True
        # Should have 3 annotations - unlabeled ones get default
        labels = [a.content for a in result.annotations]
        assert "Intro" in labels
        assert labels.count("Narrator") == 2

    def test_paragraph_split(self, mock_client):
        """Test splitting by paragraphs."""
        doc = Document(content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")

        response = json.dumps({"1": "A", "2": "B", "3": "C"})
        mock_client.add_response(response)

        strategy = IndexLabelingStrategy(split_by="paragraph")
        result = strategy.annotate(
            document=doc,
            client=mock_client,
            instructions="Label",
        )

        assert result.document_preserved is True
        assert len(result.annotations) == 3

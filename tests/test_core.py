"""Tests for core data structures."""

import pytest
from llm_annotate.core import Document, Annotation, Position, Span, AnnotationType


class TestDocument:
    """Tests for Document class."""

    def test_create_document(self):
        doc = Document(content="Hello, world!", name="test.txt")
        assert doc.content == "Hello, world!"
        assert doc.name == "test.txt"
        assert doc.char_count == 13

    def test_line_count(self):
        doc = Document(content="Line 1\nLine 2\nLine 3")
        assert doc.line_count == 3

    def test_get_line(self):
        doc = Document(content="Line 1\nLine 2\nLine 3")
        assert doc.get_line(0) == "Line 1"
        assert doc.get_line(1) == "Line 2"
        assert doc.get_line(2) == "Line 3"

    def test_offset_to_position(self):
        doc = Document(content="Hello\nWorld")

        pos = doc.offset_to_position(0)
        assert pos.line == 0
        assert pos.column == 0

        pos = doc.offset_to_position(6)  # 'W' in World
        assert pos.line == 1
        assert pos.column == 0

    def test_position_to_offset(self):
        doc = Document(content="Hello\nWorld")

        offset = doc.position_to_offset(0, 0)
        assert offset == 0

        offset = doc.position_to_offset(1, 0)
        assert offset == 6

    def test_find_all(self):
        doc = Document(content="the cat and the dog")
        offsets = doc.find_all("the")
        assert offsets == [0, 12]

    def test_is_unique(self):
        doc = Document(content="unique text and more unique")
        assert doc.is_unique("text and more") is True
        assert doc.is_unique("unique") is False


class TestAnnotation:
    """Tests for Annotation class."""

    def test_create_annotation(self):
        pos = Position(offset=10, line=1, column=5)
        ann = Annotation(
            content="This is a comment",
            position=pos,
            annotation_type=AnnotationType.COMMENT,
        )
        assert ann.content == "This is a comment"
        assert ann.position.offset == 10

    def test_annotation_with_span(self):
        pos = Position(offset=10)
        span = Span(start=10, end=20)
        ann = Annotation(
            content="Annotating this span",
            position=pos,
            span=span,
        )
        assert ann.span.length == 10

    def test_annotation_to_dict(self):
        pos = Position(offset=10, line=1, column=5)
        ann = Annotation(
            content="Test",
            position=pos,
            annotation_type=AnnotationType.WARNING,
        )
        d = ann.to_dict()
        assert d["content"] == "Test"
        assert d["type"] == "warning"
        assert d["position"]["offset"] == 10


class TestPosition:
    """Tests for Position class."""

    def test_create_position(self):
        pos = Position(offset=10, line=1, column=5)
        assert pos.offset == 10
        assert pos.line == 1
        assert pos.column == 5

    def test_invalid_offset(self):
        with pytest.raises(ValueError):
            Position(offset=-1)


class TestSpan:
    """Tests for Span class."""

    def test_create_span(self):
        span = Span(start=10, end=20)
        assert span.start == 10
        assert span.end == 20
        assert span.length == 10

    def test_invalid_span(self):
        with pytest.raises(ValueError):
            Span(start=20, end=10)  # start > end

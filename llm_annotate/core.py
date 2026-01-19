"""
Core data structures for the annotation library.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class AnnotationType(Enum):
    """Types of annotations that can be added."""
    COMMENT = "comment"
    EXPLANATION = "explanation"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    ERROR = "error"
    INFO = "info"
    CUSTOM = "custom"


@dataclass
class Position:
    """Represents a position in a document."""
    offset: int  # Character offset from start of document
    line: int = 0  # Line number (0-indexed)
    column: int = 0  # Column number (0-indexed)

    def __post_init__(self):
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")


@dataclass
class Span:
    """Represents a span of text in a document."""
    start: int  # Start character offset (inclusive)
    end: int  # End character offset (exclusive)

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError("Offsets must be non-negative")
        if self.start > self.end:
            raise ValueError("Start must be <= end")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Annotation:
    """
    Represents a single annotation attached to a document.

    Annotations are insertion-only - they reference positions in the
    original document but do not modify the document text.
    """
    content: str  # The annotation text/content
    position: Position  # Where in the document this annotation applies
    span: Optional[Span] = None  # Optional span of text this annotates
    annotation_type: AnnotationType = AnnotationType.COMMENT
    metadata: dict = field(default_factory=dict)
    anchor_text: Optional[str] = None  # The text this annotation refers to

    def to_dict(self) -> dict:
        """Convert annotation to dictionary representation."""
        return {
            "content": self.content,
            "position": {
                "offset": self.position.offset,
                "line": self.position.line,
                "column": self.position.column,
            },
            "span": {"start": self.span.start, "end": self.span.end} if self.span else None,
            "type": self.annotation_type.value,
            "metadata": self.metadata,
            "anchor_text": self.anchor_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        """Create annotation from dictionary."""
        pos_data = data["position"]
        position = Position(
            offset=pos_data["offset"],
            line=pos_data.get("line", 0),
            column=pos_data.get("column", 0),
        )
        span = None
        if data.get("span"):
            span = Span(start=data["span"]["start"], end=data["span"]["end"])
        return cls(
            content=data["content"],
            position=position,
            span=span,
            annotation_type=AnnotationType(data.get("type", "comment")),
            metadata=data.get("metadata", {}),
            anchor_text=data.get("anchor_text"),
        )


@dataclass
class Document:
    """
    Represents a document to be annotated.

    The document content is immutable - annotations are added
    as a separate layer that references positions in the original text.
    """
    content: str
    name: str = "untitled"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Compute line offsets for efficient line/column calculations
        self._line_offsets: list[int] = [0]
        for i, char in enumerate(self.content):
            if char == '\n':
                self._line_offsets.append(i + 1)

    @property
    def line_count(self) -> int:
        """Number of lines in the document."""
        return len(self._line_offsets)

    @property
    def char_count(self) -> int:
        """Number of characters in the document."""
        return len(self.content)

    def offset_to_position(self, offset: int) -> Position:
        """Convert a character offset to a Position with line/column."""
        if offset < 0 or offset > len(self.content):
            raise ValueError(f"Offset {offset} out of range [0, {len(self.content)}]")

        # Binary search for the line
        line = 0
        for i, line_offset in enumerate(self._line_offsets):
            if line_offset > offset:
                break
            line = i

        column = offset - self._line_offsets[line]
        return Position(offset=offset, line=line, column=column)

    def position_to_offset(self, line: int, column: int) -> int:
        """Convert line/column to character offset."""
        if line < 0 or line >= len(self._line_offsets):
            raise ValueError(f"Line {line} out of range [0, {len(self._line_offsets) - 1}]")
        return self._line_offsets[line] + column

    def get_line(self, line_num: int) -> str:
        """Get the content of a specific line (0-indexed)."""
        if line_num < 0 or line_num >= len(self._line_offsets):
            raise ValueError(f"Line {line_num} out of range")

        start = self._line_offsets[line_num]
        if line_num + 1 < len(self._line_offsets):
            end = self._line_offsets[line_num + 1] - 1  # Exclude newline
        else:
            end = len(self.content)
        return self.content[start:end]

    def get_text_at_span(self, span: Span) -> str:
        """Get the text at a given span."""
        return self.content[span.start:span.end]

    def get_lines(self) -> list[str]:
        """Get all lines as a list."""
        return self.content.split('\n')

    def find_all(self, text: str) -> list[int]:
        """Find all occurrences of text, return their offsets."""
        offsets = []
        start = 0
        while True:
            pos = self.content.find(text, start)
            if pos == -1:
                break
            offsets.append(pos)
            start = pos + 1
        return offsets

    def is_unique(self, text: str) -> bool:
        """Check if a text appears exactly once in the document."""
        return len(self.find_all(text)) == 1


@dataclass
class AnnotationResult:
    """
    Result of an annotation operation.

    Contains the annotations, metrics about the operation,
    and verification results.
    """
    document: Document
    annotations: list[Annotation]
    strategy_name: str

    # Metrics
    llm_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    latency_ms: float = 0.0

    # Verification
    document_preserved: bool = True
    preservation_errors: list[str] = field(default_factory=list)

    # Raw outputs for debugging
    raw_llm_responses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "document_name": self.document.name,
            "strategy": self.strategy_name,
            "annotations": [a.to_dict() for a in self.annotations],
            "metrics": {
                "llm_calls": self.llm_calls,
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "retries": self.retries,
                "latency_ms": self.latency_ms,
            },
            "verification": {
                "document_preserved": self.document_preserved,
                "preservation_errors": self.preservation_errors,
            },
        }

    def render_annotated(self, format: str = "inline") -> str:
        """
        Render the document with annotations.

        Formats:
        - inline: Insert annotations inline as comments
        - margin: Show annotations as margin notes
        - json: Return as JSON structure
        """
        if format == "json":
            import json
            return json.dumps(self.to_dict(), indent=2)

        if format == "inline":
            return self._render_inline()

        if format == "margin":
            return self._render_margin()

        raise ValueError(f"Unknown format: {format}")

    def _render_inline(self) -> str:
        """Render with inline annotations."""
        # Sort annotations by position (reverse order for insertion)
        sorted_annotations = sorted(
            self.annotations,
            key=lambda a: a.position.offset,
            reverse=True
        )

        result = self.document.content
        for ann in sorted_annotations:
            marker = f" /* [{ann.annotation_type.value}] {ann.content} */ "
            result = result[:ann.position.offset] + marker + result[ann.position.offset:]

        return result

    def _render_margin(self) -> str:
        """Render with margin annotations."""
        lines = self.document.get_lines()

        # Group annotations by line
        line_annotations: dict[int, list[Annotation]] = {}
        for ann in self.annotations:
            line = ann.position.line
            if line not in line_annotations:
                line_annotations[line] = []
            line_annotations[line].append(ann)

        # Build output with margin notes
        result_lines = []
        max_line_len = max(len(line) for line in lines) if lines else 0

        for i, line in enumerate(lines):
            if i in line_annotations:
                annotations = line_annotations[i]
                ann_text = "; ".join(f"[{a.annotation_type.value}] {a.content}" for a in annotations)
                result_lines.append(f"{line.ljust(max_line_len)}  // {ann_text}")
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)

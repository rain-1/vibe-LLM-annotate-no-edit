"""
Line Reference Strategy

Annotations reference line numbers and column positions.
The document is never modified inline - annotations are provided separately.
"""

import json
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class LineReferenceStrategy(BaseStrategy):
    """
    Line-reference annotation strategy.

    Annotations specify line numbers (and optionally columns) instead of
    character offsets. This is often more intuitive for LLMs as they
    naturally see documents as lines.
    """

    name = "line_reference"
    description = "Annotations reference line:column positions"

    def __init__(self, max_retries: int = 3, one_indexed: bool = True):
        """
        Args:
            max_retries: Maximum retry attempts
            one_indexed: Whether line numbers are 1-indexed (default) or 0-indexed
        """
        super().__init__(max_retries=max_retries)
        self.one_indexed = one_indexed

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation, warning, suggestion'

        # Generate numbered document view
        numbered_doc = self._generate_numbered_view(document)

        line_start = 1 if self.one_indexed else 0

        prompt = f"""Analyze this document and provide annotations referencing line numbers.

DOCUMENT WITH LINE NUMBERS:
{numbered_doc}

Provide annotations as JSON. Each annotation should reference:
- line: line number (starting from {line_start})
- column: (optional) column position within the line
- end_line/end_column: (optional) for multi-line annotations
- quote: exact text being annotated (for verification)
- type: one of [{types_str}]
- content: your annotation

RESPONSE FORMAT (JSON only):
{{
  "annotations": [
    {{
      "line": {line_start},
      "column": 0,
      "quote": "text at this position",
      "type": "comment",
      "content": "Your annotation"
    }}
  ]
}}

Instructions: {instructions}

Return ONLY valid JSON."""

        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0.0
        retries = 0
        raw_responses = []
        last_errors = []

        for attempt in range(self.max_retries):
            try:
                parsed, response = client.complete_json(
                    prompt, system_prompt=self._build_system_prompt()
                )
            except Exception as e:
                raw_responses.append(str(e))
                retries += 1
                continue

            total_tokens += response.total_tokens
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            latency_ms += response.latency_ms
            raw_responses.append(response.content)

            # Process annotations
            annotations, errors = self._process_annotations(document, parsed)
            last_errors = errors

            if annotations or not errors:
                return self._create_result(
                    document=document,
                    annotations=annotations,
                    llm_calls=attempt + 1,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retries=retries,
                    latency_ms=latency_ms,
                    preserved=True,
                    errors=errors,
                    raw_responses=raw_responses,
                )

            retries += 1

            if attempt < self.max_retries - 1:
                prompt = f"""Your annotations had line reference errors:
{chr(10).join(errors[:5])}

DOCUMENT WITH LINE NUMBERS:
{numbered_doc}

Instructions: {instructions}

Return corrected JSON annotations."""

        return self._create_result(
            document=document,
            annotations=annotations if 'annotations' in dir() else [],
            llm_calls=self.max_retries,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=retries,
            latency_ms=latency_ms,
            preserved=True,
            errors=last_errors,
            raw_responses=raw_responses,
        )

    def _generate_numbered_view(self, document: Document) -> str:
        """Generate document view with line numbers."""
        lines = document.get_lines()
        numbered_lines = []

        start = 1 if self.one_indexed else 0
        for i, line in enumerate(lines):
            line_num = i + start
            numbered_lines.append(f"{line_num:4d} | {line}")

        return '\n'.join(numbered_lines)

    def _process_annotations(
        self,
        document: Document,
        parsed: dict,
    ) -> tuple[list[Annotation], list[str]]:
        """Process parsed annotations, converting line refs to positions."""
        annotations = []
        errors = []

        lines = document.get_lines()
        line_offset = 1 if self.one_indexed else 0
        max_line = len(lines) - 1 + line_offset

        ann_list = parsed.get('annotations', [])

        for i, ann_data in enumerate(ann_list):
            line = ann_data.get('line', line_offset)
            column = ann_data.get('column', 0)
            quote = ann_data.get('quote', '')
            ann_type_str = ann_data.get('type', 'comment')
            content = ann_data.get('content', '')

            # Convert to 0-indexed
            line_idx = line - line_offset

            # Validate line number
            if line_idx < 0 or line_idx >= len(lines):
                errors.append(f"Annotation {i}: line {line} out of range [{line_offset}, {max_line}]")
                continue

            # Validate column
            line_content = lines[line_idx]
            if column < 0 or column > len(line_content):
                errors.append(f"Annotation {i}: column {column} out of range for line {line}")
                column = 0  # Default to start of line

            # Verify quote if provided
            if quote:
                actual_text = line_content[column:column + len(quote)]
                if actual_text != quote:
                    # Try to find the quote in the line
                    quote_pos = line_content.find(quote)
                    if quote_pos >= 0:
                        column = quote_pos
                    else:
                        errors.append(
                            f"Annotation {i}: quote '{quote}' not found at line {line}, column {column}"
                        )

            # Convert to offset-based position
            try:
                offset = document.position_to_offset(line_idx, column)
                position = Position(offset=offset, line=line_idx, column=column)
            except ValueError:
                position = Position(offset=0, line=0, column=0)

            # Handle end position for spans
            span = None
            end_line = ann_data.get('end_line')
            end_column = ann_data.get('end_column')

            if end_line is not None:
                end_line_idx = end_line - line_offset
                if 0 <= end_line_idx < len(lines):
                    end_col = end_column if end_column is not None else len(lines[end_line_idx])
                    try:
                        end_offset = document.position_to_offset(end_line_idx, end_col)
                        span = Span(start=position.offset, end=end_offset)
                    except ValueError:
                        pass

            # Parse annotation type
            try:
                ann_type = AnnotationType(ann_type_str)
            except ValueError:
                ann_type = AnnotationType.COMMENT

            annotations.append(
                Annotation(
                    content=content,
                    position=position,
                    span=span,
                    annotation_type=ann_type,
                    anchor_text=quote,
                )
            )

        return annotations, errors

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """Document is never modified in this strategy."""
        return True, []

    def _build_system_prompt(self) -> str:
        idx_type = "1-indexed" if self.one_indexed else "0-indexed"
        return (
            f"You are a document analyzer that provides annotations with line numbers. "
            f"Line numbers are {idx_type}. You never modify the document - "
            f"you only reference positions in it. Always include a quote to verify "
            f"your line references are accurate."
        )

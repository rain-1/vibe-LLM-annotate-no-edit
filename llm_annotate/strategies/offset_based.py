"""
Offset-Based Strategy

LLM provides annotations with character offsets - document is never modified.
Annotations reference specific positions using character offsets.
"""

import json
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class OffsetBasedStrategy(BaseStrategy):
    """
    Offset-based annotation strategy.

    The LLM provides annotations with explicit character offsets.
    The document is NEVER modified - annotations are purely positional references.
    This is the safest approach for preservation but requires the LLM to
    accurately count characters.
    """

    name = "offset_based"
    description = "Annotations reference character offsets, document unchanged"

    def __init__(self, max_retries: int = 3, verify_anchors: bool = True):
        super().__init__(max_retries=max_retries)
        self.verify_anchors = verify_anchors

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation, warning, suggestion'

        # Add character position hints
        char_hints = self._generate_position_hints(document)

        prompt = f"""Analyze the following document and provide annotations with character offsets.

CHARACTER POSITION REFERENCE:
{char_hints}

DOCUMENT (total {len(document.content)} characters):
```
{document.content}
```

Provide annotations as JSON. For each annotation, specify:
- offset: character position where the annotation applies (0-indexed)
- end_offset: (optional) end position if annotating a span
- anchor_text: the exact text at that position (for verification)
- type: one of [{types_str}]
- content: your annotation

RESPONSE FORMAT (JSON only):
{{
  "annotations": [
    {{
      "offset": 0,
      "end_offset": 10,
      "anchor_text": "exact text",
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

            # Verify and create annotations
            annotations, errors = self._process_annotations(document, parsed)
            last_errors = errors

            # If we have some valid annotations, accept the result
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
                    preserved=True,  # Document is never modified in this strategy
                    errors=errors,
                    raw_responses=raw_responses,
                )

            retries += 1

            if attempt < self.max_retries - 1:
                # Provide feedback for retry
                prompt = f"""Your previous annotations had offset errors:
{chr(10).join(errors[:5])}

Please provide corrected annotations. Use the position reference below.

CHARACTER POSITION REFERENCE:
{char_hints}

DOCUMENT:
```
{document.content}
```

Instructions: {instructions}

Return ONLY valid JSON with corrected offsets."""

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

    def _generate_position_hints(self, document: Document) -> str:
        """Generate position hints to help LLM with offsets."""
        hints = []
        lines = document.get_lines()

        current_offset = 0
        for i, line in enumerate(lines[:20]):  # First 20 lines
            preview = line[:50] + "..." if len(line) > 50 else line
            hints.append(f"Line {i} starts at offset {current_offset}: \"{preview}\"")
            current_offset += len(line) + 1  # +1 for newline

        if len(lines) > 20:
            hints.append(f"... ({len(lines) - 20} more lines)")

        return '\n'.join(hints)

    def _process_annotations(
        self,
        document: Document,
        parsed: dict,
    ) -> tuple[list[Annotation], list[str]]:
        """Process parsed JSON into annotations, verifying offsets."""
        annotations = []
        errors = []

        ann_list = parsed.get('annotations', [])

        for i, ann_data in enumerate(ann_list):
            offset = ann_data.get('offset', 0)
            end_offset = ann_data.get('end_offset')
            anchor_text = ann_data.get('anchor_text', '')
            ann_type_str = ann_data.get('type', 'comment')
            content = ann_data.get('content', '')

            # Validate offset
            if offset < 0 or offset > len(document.content):
                errors.append(f"Annotation {i}: offset {offset} out of range [0, {len(document.content)}]")
                continue

            # Verify anchor text if provided
            if self.verify_anchors and anchor_text:
                if end_offset:
                    actual_text = document.content[offset:end_offset]
                else:
                    actual_text = document.content[offset:offset + len(anchor_text)]

                if actual_text != anchor_text:
                    errors.append(
                        f"Annotation {i}: anchor mismatch at offset {offset}. "
                        f"Expected '{anchor_text}', found '{actual_text}'"
                    )
                    # Try to find the correct offset
                    correct_offset = document.content.find(anchor_text)
                    if correct_offset >= 0:
                        offset = correct_offset
                        if end_offset:
                            end_offset = offset + len(anchor_text)

            # Create position
            try:
                position = document.offset_to_position(offset)
            except ValueError:
                position = Position(offset=0, line=0, column=0)

            # Create span if end_offset provided
            span = None
            if end_offset and end_offset > offset:
                span = Span(start=offset, end=min(end_offset, len(document.content)))

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
                    anchor_text=anchor_text,
                )
            )

        return annotations, errors

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """In offset-based strategy, document is never modified."""
        # This strategy never modifies the document
        return True, []

    def _build_system_prompt(self) -> str:
        return (
            "You are a precise document analyzer that provides annotations with exact "
            "character positions. You never modify the document - you only reference "
            "positions in it. Be very careful with offset calculations. "
            "Always include anchor_text to verify your offsets are correct."
        )

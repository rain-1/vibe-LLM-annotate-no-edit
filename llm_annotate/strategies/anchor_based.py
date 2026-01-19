"""
Anchor-Based Strategy

Annotations are attached to unique text anchors (snippets) in the document.
The LLM identifies distinctive text that serves as an anchor for each annotation.
Verification checks that anchors exist uniquely in the document.
"""

import json
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class AnchorBasedStrategy(BaseStrategy):
    """
    Anchor-based annotation strategy.

    Instead of using positions or modifying text, annotations are attached
    to unique text anchors (snippets that appear exactly once in the document).

    Benefits:
    - Document is never modified
    - Anchors are human-readable
    - Robust to small document changes
    - Easy to verify (anchor must exist uniquely)
    """

    name = "anchor_based"
    description = "Annotations attached to unique text anchors"

    def __init__(
        self,
        max_retries: int = 3,
        min_anchor_length: int = 10,
        max_anchor_length: int = 100,
    ):
        super().__init__(max_retries=max_retries)
        self.min_anchor_length = min_anchor_length
        self.max_anchor_length = max_anchor_length

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation, warning, suggestion'

        prompt = f"""Analyze this document and provide annotations with text anchors.

DOCUMENT:
```
{document.content}
```

For each annotation, specify:
- anchor: A unique text snippet (10-100 chars) from the document that identifies the annotation location
- anchor_context: "before" (annotation goes before anchor) or "after" (annotation goes after anchor)
- type: one of [{types_str}]
- content: Your annotation text

IMPORTANT: Each anchor must:
1. Be copied EXACTLY from the document (character-for-character)
2. Appear EXACTLY ONCE in the document (be unique)
3. Be {self.min_anchor_length}-{self.max_anchor_length} characters long

RESPONSE FORMAT (JSON only):
{{
  "annotations": [
    {{
      "anchor": "exact text from document",
      "anchor_context": "before",
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

            # Validate anchors and create annotations
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
                # Provide feedback on anchor issues
                prompt = f"""Your anchors had issues:
{chr(10).join(errors[:5])}

Anchors must be EXACT text from the document and appear ONLY ONCE.

DOCUMENT:
```
{document.content}
```

Instructions: {instructions}

Provide corrected annotations with valid unique anchors (JSON only)."""

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

    def _process_annotations(
        self,
        document: Document,
        parsed: dict,
    ) -> tuple[list[Annotation], list[str]]:
        """Validate anchors and create annotations."""
        annotations = []
        errors = []

        ann_list = parsed.get('annotations', [])

        for i, ann_data in enumerate(ann_list):
            anchor = ann_data.get('anchor', '')
            context = ann_data.get('anchor_context', 'after')
            ann_type_str = ann_data.get('type', 'comment')
            content = ann_data.get('content', '')

            # Validate anchor length
            if len(anchor) < self.min_anchor_length:
                errors.append(f"Annotation {i}: anchor too short ({len(anchor)} chars)")
                continue
            if len(anchor) > self.max_anchor_length:
                errors.append(f"Annotation {i}: anchor too long ({len(anchor)} chars)")
                # Try to use it anyway, just warn
                anchor = anchor[:self.max_anchor_length]

            # Find anchor in document
            occurrences = document.find_all(anchor)

            if len(occurrences) == 0:
                # Try fuzzy matching - maybe whitespace issue
                normalized_anchor = ' '.join(anchor.split())
                for offset in range(len(document.content) - len(normalized_anchor)):
                    candidate = document.content[offset:offset + len(anchor) + 20]
                    normalized_candidate = ' '.join(candidate.split())
                    if normalized_anchor in normalized_candidate:
                        occurrences = [offset]
                        anchor = candidate[:len(anchor)]
                        break

                if len(occurrences) == 0:
                    errors.append(f"Annotation {i}: anchor not found: '{anchor[:40]}...'")
                    continue

            if len(occurrences) > 1:
                errors.append(
                    f"Annotation {i}: anchor not unique, found {len(occurrences)} times: '{anchor[:40]}...'"
                )
                # Use first occurrence anyway
                occurrences = occurrences[:1]

            # Calculate position
            anchor_offset = occurrences[0]
            if context == 'before':
                offset = anchor_offset
            else:
                offset = anchor_offset + len(anchor)

            offset = min(offset, len(document.content))

            try:
                position = document.offset_to_position(offset)
            except ValueError:
                position = Position(offset=0, line=0, column=0)

            # Create span for the anchor text
            span = Span(start=anchor_offset, end=anchor_offset + len(anchor))

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
                    anchor_text=anchor,
                    metadata={'anchor_context': context},
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
        return (
            "You are a document analyzer that provides annotations with text anchors. "
            "Each anchor must be an EXACT substring from the document that appears "
            "ONLY ONCE. You never modify the document - you only identify unique "
            "text snippets to anchor your annotations to."
        )

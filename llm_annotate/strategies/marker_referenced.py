"""
Marker Referenced Strategy

Inserts lightweight markers in the text (e.g., [[1]], [[2]]) and provides
annotations separately in a JSON structure. This minimizes inline changes.
"""

import re
import json
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class MarkerReferencedStrategy(BaseStrategy):
    """
    Marker-referenced annotation strategy.

    Inserts small markers like [[1]], [[2]] in the document and provides
    full annotations in a separate JSON structure. This approach:
    - Minimizes inline text changes
    - Keeps annotations structured and parseable
    - Makes verification easier (just check marker format)
    """

    name = "marker_referenced"
    description = "Lightweight markers with separate JSON annotations"

    def __init__(
        self,
        max_retries: int = 3,
        marker_format: str = "[[{id}]]",
    ):
        super().__init__(max_retries=max_retries)
        self.marker_format = marker_format
        # Create regex pattern from marker format
        self.marker_pattern = re.escape(marker_format).replace(r'\{id\}', r'(\d+)')

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation, warning, suggestion'

        prompt = f"""Annotate the following document by inserting markers and providing annotations separately.

RULES:
1. Insert markers in the format {self.marker_format.format(id='N')} where N is a number starting from 1
2. Do NOT modify any text - only INSERT markers
3. Provide annotations as JSON after the document

DOCUMENT:
---
{document.content}
---

RESPONSE FORMAT:
First, output the document with markers inserted.
Then output a JSON block with your annotations:

```json
{{
  "annotations": [
    {{"id": 1, "type": "comment", "content": "Your annotation text", "context": "nearby text for reference"}},
    ...
  ]
}}
```

Valid annotation types: {types_str}

Instructions: {instructions}"""

        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0.0
        retries = 0
        raw_responses = []

        for attempt in range(self.max_retries):
            response = client.complete(prompt, system_prompt=self._build_system_prompt())

            total_tokens += response.total_tokens
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            latency_ms += response.latency_ms
            raw_responses.append(response.content)

            # Parse response
            marked_doc, annotations_json = self._parse_response(response.content)

            # Verify preservation
            preserved, errors = self.verify_preservation(document, marked_doc)

            if preserved:
                # Extract and create annotations
                annotations = self._create_annotations(document, marked_doc, annotations_json)

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
                    errors=[],
                    raw_responses=raw_responses,
                )

            retries += 1

            if attempt < self.max_retries - 1:
                prompt = f"""Your previous response MODIFIED the original text. This is not allowed.

Errors: {'; '.join(errors[:3])}

ONLY insert markers like {self.marker_format.format(id='1')}, {self.marker_format.format(id='2')}, etc.
Do NOT change ANY other text.

ORIGINAL DOCUMENT:
---
{document.content}
---

Try again. Insert only markers, then provide JSON annotations."""

        # Return with errors if all retries failed
        annotations = self._create_annotations(document, marked_doc, annotations_json)
        return self._create_result(
            document=document,
            annotations=annotations,
            llm_calls=self.max_retries,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=retries,
            latency_ms=latency_ms,
            preserved=False,
            errors=errors,
            raw_responses=raw_responses,
        )

    def _parse_response(self, response: str) -> tuple[str, list[dict]]:
        """Parse the response into marked document and annotations JSON."""
        # Find JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if not json_match:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*"annotations"[\s\S]*\}', response)

        annotations_json = []
        if json_match:
            try:
                json_str = json_match.group(1) if '```' in response else json_match.group(0)
                parsed = json.loads(json_str)
                annotations_json = parsed.get('annotations', [])
            except json.JSONDecodeError:
                pass

        # Extract document (everything before JSON or code block)
        doc_content = response

        # Remove JSON block
        if '```json' in doc_content:
            doc_content = re.sub(r'```json[\s\S]*?```', '', doc_content)
        elif json_match:
            doc_content = doc_content[:json_match.start()]

        # Clean up document markers
        doc_content = doc_content.strip()

        # Remove --- delimiters if present
        doc_content = re.sub(r'^---\s*', '', doc_content)
        doc_content = re.sub(r'\s*---$', '', doc_content)

        # Remove markdown code blocks
        if doc_content.startswith('```'):
            lines = doc_content.split('\n')
            doc_content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        return doc_content.strip(), annotations_json

    def verify_preservation(
        self,
        original: Document,
        marked_doc: str,
    ) -> tuple[bool, list[str]]:
        """Verify that only markers were added."""
        errors = []

        # Remove all markers
        cleaned = re.sub(self.marker_pattern, '', marked_doc)

        # Compare with original
        if cleaned != original.content:
            # Find first difference
            min_len = min(len(cleaned), len(original.content))
            for i in range(min_len):
                if cleaned[i] != original.content[i]:
                    context_start = max(0, i - 15)
                    context_end = min(min_len, i + 15)
                    errors.append(
                        f"Mismatch at pos {i}: expected '{original.content[context_start:context_end]}', "
                        f"got '{cleaned[context_start:context_end]}'"
                    )
                    break

            if len(cleaned) != len(original.content):
                errors.append(f"Length mismatch: {len(original.content)} vs {len(cleaned)}")

        return len(errors) == 0, errors

    def _create_annotations(
        self,
        document: Document,
        marked_doc: str,
        annotations_json: list[dict],
    ) -> list[Annotation]:
        """Create Annotation objects from markers and JSON."""
        annotations = []

        # Build mapping of marker IDs to positions
        marker_positions = {}
        for match in re.finditer(self.marker_pattern, marked_doc):
            marker_id = int(match.group(1))

            # Calculate position in original document
            preceding = marked_doc[:match.start()]
            preceding_clean = re.sub(self.marker_pattern, '', preceding)
            offset = len(preceding_clean)
            offset = min(offset, len(document.content))

            marker_positions[marker_id] = offset

        # Create annotations
        for ann_data in annotations_json:
            ann_id = ann_data.get('id')
            if ann_id not in marker_positions:
                continue

            offset = marker_positions[ann_id]

            try:
                position = document.offset_to_position(offset)
            except ValueError:
                position = Position(offset=0, line=0, column=0)

            try:
                ann_type = AnnotationType(ann_data.get('type', 'comment'))
            except ValueError:
                ann_type = AnnotationType.COMMENT

            annotations.append(
                Annotation(
                    content=ann_data.get('content', ''),
                    position=position,
                    annotation_type=ann_type,
                    anchor_text=ann_data.get('context'),
                    metadata={'marker_id': ann_id},
                )
            )

        return annotations

    def _build_system_prompt(self) -> str:
        return (
            "You are a document annotator that uses reference markers. "
            "Insert small markers like [[1]], [[2]] at annotation points, "
            "then provide full annotations in JSON. "
            "NEVER modify the original text - only insert markers."
        )

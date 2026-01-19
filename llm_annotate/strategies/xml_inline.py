"""
XML Inline Strategy

Uses XML tags for annotations with structured verification.
The document is wrapped in tags, and annotations are inserted as XML elements.
"""

import re
import xml.etree.ElementTree as ET
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class XMLInlineStrategy(BaseStrategy):
    """
    XML-based inline annotation strategy.

    Uses XML structure to clearly delineate annotations from content.
    Annotations are inserted as <ann> tags within the document.
    Verification checks that text outside annotations matches original.
    """

    name = "xml_inline"
    description = "XML tags for structured inline annotations with verification"

    def __init__(self, max_retries: int = 3, strict_verify: bool = True):
        super().__init__(max_retries=max_retries)
        self.strict_verify = strict_verify

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation, warning, suggestion'

        prompt = f"""Annotate the following document using XML annotation tags.

RULES:
1. The original text MUST remain EXACTLY as-is - do not modify any character
2. Insert annotations using: <ann type="TYPE">annotation text</ann>
3. To annotate a span of text, wrap it: <span><ann type="TYPE">note</ann>original text here</span>
4. Valid annotation types: {types_str}

DOCUMENT TO ANNOTATE:
<document>
{document.content}
</document>

Return the document with your XML annotations inserted. The text between annotations must be IDENTICAL to the original.
Return ONLY the annotated document wrapped in <document> tags."""

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

            # Extract document content from response
            annotated = self._extract_document_content(response.content)

            # Verify preservation
            preserved, errors = self.verify_preservation(document, annotated)

            if preserved:
                # Extract annotations
                annotations = self._extract_annotations(document, annotated)

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
                # Add correction to prompt for retry
                prompt = f"""Your previous annotation MODIFIED the original text, which is not allowed.

Errors found:
{chr(10).join(errors[:5])}

Try again. Insert ONLY <ann> tags. Do NOT change ANY text.

ORIGINAL DOCUMENT:
<document>
{document.content}
</document>

Return the document with annotations. The text MUST be identical to the original."""

        # Final attempt failed
        annotations = self._extract_annotations(document, annotated)
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

    def _extract_document_content(self, response: str) -> str:
        """Extract content from <document> tags."""
        # Try to find document tags
        match = re.search(r'<document>(.*?)</document>', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: remove markdown code blocks
        content = response.strip()
        if content.startswith("```"):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        return content

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """Verify original text is preserved outside of annotations."""
        errors = []

        # Remove all annotation tags
        cleaned = annotated_text

        # Remove <ann>...</ann> tags (annotations inline)
        cleaned = re.sub(r'<ann[^>]*>.*?</ann>', '', cleaned, flags=re.DOTALL)

        # Handle <span><ann>...</ann>text</span> - keep the text
        cleaned = re.sub(r'<span>\s*<ann[^>]*>.*?</ann>\s*(.*?)</span>', r'\1', cleaned, flags=re.DOTALL)

        # Remove any remaining span tags
        cleaned = re.sub(r'</?span>', '', cleaned)

        # Compare
        if self.strict_verify:
            # Exact character-by-character comparison
            if cleaned != original.content:
                # Find first difference
                for i, (o, c) in enumerate(zip(original.content, cleaned)):
                    if o != c:
                        context_start = max(0, i - 20)
                        context_end = min(len(original.content), i + 20)
                        errors.append(
                            f"Difference at position {i}: "
                            f"expected '{original.content[context_start:context_end]}', "
                            f"got '{cleaned[context_start:min(len(cleaned), context_end)]}'"
                        )
                        break

                if len(original.content) != len(cleaned):
                    errors.append(
                        f"Length mismatch: original={len(original.content)}, "
                        f"annotated={len(cleaned)}"
                    )
        else:
            # Whitespace-normalized comparison
            orig_normalized = ' '.join(original.content.split())
            cleaned_normalized = ' '.join(cleaned.split())

            if orig_normalized != cleaned_normalized:
                errors.append("Content mismatch after normalization")

        return len(errors) == 0, errors

    def _extract_annotations(
        self,
        document: Document,
        annotated_text: str,
    ) -> list[Annotation]:
        """Extract annotations from XML-annotated text."""
        annotations = []

        # Track position offset as we process
        current_pos = 0
        cleaned_pos = 0

        # Find all annotation patterns
        # Pattern 1: <ann type="...">content</ann> (standalone annotation)
        # Pattern 2: <span><ann type="...">content</ann>text</span> (annotation on text)

        # Process standalone annotations
        for match in re.finditer(r'<ann\s+type=["\'](\w+)["\']>(.*?)</ann>', annotated_text, re.DOTALL):
            ann_type = match.group(1)
            content = match.group(2).strip()

            # Calculate position in original document
            preceding = annotated_text[:match.start()]
            # Remove annotations from preceding text
            preceding_clean = re.sub(r'<ann[^>]*>.*?</ann>', '', preceding, flags=re.DOTALL)
            preceding_clean = re.sub(r'<span>\s*<ann[^>]*>.*?</ann>\s*(.*?)</span>', r'\1', preceding_clean, flags=re.DOTALL)
            preceding_clean = re.sub(r'</?span>', '', preceding_clean)

            offset = len(preceding_clean)
            offset = min(offset, len(document.content))

            try:
                position = document.offset_to_position(offset)
            except ValueError:
                position = Position(offset=0, line=0, column=0)

            try:
                annotation_type = AnnotationType(ann_type)
            except ValueError:
                annotation_type = AnnotationType.COMMENT

            annotations.append(
                Annotation(
                    content=content,
                    position=position,
                    annotation_type=annotation_type,
                )
            )

        # Process span annotations (annotations attached to specific text)
        for match in re.finditer(
            r'<span>\s*<ann\s+type=["\'](\w+)["\']>(.*?)</ann>\s*(.*?)</span>',
            annotated_text,
            re.DOTALL
        ):
            ann_type = match.group(1)
            content = match.group(2).strip()
            anchor_text = match.group(3).strip()

            # Find where this anchor text is in original
            preceding = annotated_text[:match.start()]
            preceding_clean = re.sub(r'<ann[^>]*>.*?</ann>', '', preceding, flags=re.DOTALL)
            preceding_clean = re.sub(r'<span>\s*<ann[^>]*>.*?</ann>\s*(.*?)</span>', r'\1', preceding_clean, flags=re.DOTALL)
            preceding_clean = re.sub(r'</?span>', '', preceding_clean)

            offset = len(preceding_clean)
            offset = min(offset, len(document.content))

            try:
                position = document.offset_to_position(offset)
            except ValueError:
                position = Position(offset=0, line=0, column=0)

            try:
                annotation_type = AnnotationType(ann_type)
            except ValueError:
                annotation_type = AnnotationType.COMMENT

            span = Span(start=offset, end=min(offset + len(anchor_text), len(document.content)))

            annotations.append(
                Annotation(
                    content=content,
                    position=position,
                    span=span,
                    annotation_type=annotation_type,
                    anchor_text=anchor_text,
                )
            )

        return annotations

    def _build_system_prompt(self) -> str:
        return (
            "You are a precise XML document annotator. You add annotations using XML tags. "
            "CRITICAL: You must NEVER modify the original text - only insert <ann> tags. "
            "Even if you see errors, typos, or issues in the text - leave them exactly as-is."
        )

"""
Baseline Naive Strategy

Simply asks the LLM to annotate the document inline.
This serves as a baseline - it often results in modifications to the original text.
"""

import re
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class BaselineNaiveStrategy(BaseStrategy):
    """
    Baseline strategy: Ask LLM to annotate inline, no verification.

    This is the simplest approach - just ask the LLM to insert annotations.
    It typically has the highest modification rate as LLMs often
    "improve" the text while annotating.
    """

    name = "baseline_naive"
    description = "Simple inline annotation with no verification"

    def __init__(self, annotation_format: str = "comment", max_retries: int = 3):
        """
        Args:
            annotation_format: Format for annotations - "comment" (/* */),
                             "xml" (<annotation>), or "bracket" ([[ ]])
        """
        super().__init__(max_retries=max_retries)
        self.annotation_format = annotation_format

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        # Build prompt
        format_examples = {
            "comment": "/* annotation text */",
            "xml": "<annotation type='comment'>text</annotation>",
            "bracket": "[[annotation: text]]",
        }
        format_example = format_examples.get(self.annotation_format, format_examples["comment"])

        prompt = f"""Annotate the following document by inserting annotations inline.

CRITICAL: Do NOT modify the original text in any way. Only INSERT annotations.
Do NOT fix typos, change wording, or alter any part of the original text.

Use this format for annotations: {format_example}

Annotation instructions: {instructions}

Types of annotations to add: {', '.join(annotation_types) if annotation_types else 'any relevant comments'}

DOCUMENT:
```
{document.content}
```

Return ONLY the annotated document with your annotations inserted. Do not include any other text or explanation."""

        # Call LLM
        response = client.complete(prompt, system_prompt=self._build_system_prompt())

        # Extract annotated content
        annotated = response.content.strip()

        # Remove markdown code blocks if present
        if annotated.startswith("```"):
            lines = annotated.split('\n')
            annotated = '\n'.join(lines[1:-1])

        # Verify preservation
        preserved, errors = self.verify_preservation(document, annotated)

        # Extract annotations from the result
        annotations = self._extract_annotations(document, annotated)

        return self._create_result(
            document=document,
            annotations=annotations,
            llm_calls=1,
            total_tokens=response.total_tokens,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            retries=0,
            latency_ms=response.latency_ms,
            preserved=preserved,
            errors=errors,
            raw_responses=[response.content],
        )

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """Check if original text is preserved by removing annotations."""
        errors = []

        # Remove annotations based on format
        cleaned = annotated_text
        if self.annotation_format == "comment":
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        elif self.annotation_format == "xml":
            cleaned = re.sub(r'<annotation[^>]*>.*?</annotation>', '', cleaned, flags=re.DOTALL)
        elif self.annotation_format == "bracket":
            cleaned = re.sub(r'\[\[annotation:.*?\]\]', '', cleaned, flags=re.DOTALL)

        # Normalize whitespace for comparison
        original_normalized = ' '.join(original.content.split())
        cleaned_normalized = ' '.join(cleaned.split())

        if original_normalized != cleaned_normalized:
            # Find differences
            original_words = original.content.split()
            cleaned_words = cleaned.split()

            # Simple diff to find changes
            for i, (o, c) in enumerate(zip(original_words, cleaned_words)):
                if o != c:
                    errors.append(f"Word {i} changed: '{o}' -> '{c}'")
                    if len(errors) >= 10:
                        errors.append("... (more differences)")
                        break

            if len(original_words) != len(cleaned_words):
                errors.append(
                    f"Word count changed: {len(original_words)} -> {len(cleaned_words)}"
                )

        return len(errors) == 0, errors

    def _extract_annotations(
        self,
        document: Document,
        annotated_text: str,
    ) -> list[Annotation]:
        """Extract annotations from the annotated text."""
        annotations = []

        # Patterns for different formats
        patterns = {
            "comment": r'/\*(.*?)\*/',
            "xml": r'<annotation[^>]*type=[\'"](\w+)[\'"][^>]*>(.*?)</annotation>',
            "bracket": r'\[\[annotation:\s*(.*?)\]\]',
        }

        pattern = patterns.get(self.annotation_format, patterns["comment"])

        # Find all annotations
        for match in re.finditer(pattern, annotated_text, re.DOTALL):
            # Estimate position by finding where in the cleaned text this would be
            position_in_annotated = match.start()

            # Count how much annotation text comes before this
            preceding_text = annotated_text[:position_in_annotated]
            if self.annotation_format == "comment":
                preceding_clean = re.sub(r'/\*.*?\*/', '', preceding_text, flags=re.DOTALL)
            elif self.annotation_format == "xml":
                preceding_clean = re.sub(
                    r'<annotation[^>]*>.*?</annotation>', '', preceding_text, flags=re.DOTALL
                )
            else:
                preceding_clean = re.sub(
                    r'\[\[annotation:.*?\]\]', '', preceding_text, flags=re.DOTALL
                )

            # The position in original document
            offset = len(preceding_clean)
            offset = min(offset, len(document.content))

            position = document.offset_to_position(offset)

            # Extract annotation content
            if self.annotation_format == "xml":
                ann_type = match.group(1)
                content = match.group(2).strip()
            else:
                content = match.group(1).strip()
                ann_type = "comment"

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

        return annotations

    def _build_system_prompt(self) -> str:
        return (
            "You are a document annotator. Your ONLY job is to INSERT annotations "
            "into documents. You must NEVER modify, fix, or change the original text. "
            "Even if you see typos, grammatical errors, or other issues - leave them. "
            "Only add annotations."
        )

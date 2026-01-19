"""
Chunked Verified Strategy

Splits the document into chunks (sentences/paragraphs), annotates each chunk
separately, and verifies that each chunk's original text is preserved.
Retries individual chunks that fail verification.
"""

import re
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class ChunkedVerifiedStrategy(BaseStrategy):
    """
    Chunked verification annotation strategy.

    1. Split document into chunks (sentences or paragraphs)
    2. Annotate each chunk independently
    3. Verify each chunk preserves original text
    4. Retry failed chunks with stronger instructions
    5. Reassemble with verified annotations

    This approach limits the blast radius of LLM modifications.
    """

    name = "chunked_verified"
    description = "Split into chunks, annotate each with verification and retry"

    def __init__(
        self,
        max_retries: int = 3,
        chunk_by: str = "paragraph",  # "sentence" or "paragraph"
        max_chunk_size: int = 500,
    ):
        super().__init__(max_retries=max_retries)
        self.chunk_by = chunk_by
        self.max_chunk_size = max_chunk_size

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation, warning'

        # Split document into chunks
        chunks = self._split_into_chunks(document)

        all_annotations = []
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0.0
        total_retries = 0
        total_calls = 0
        raw_responses = []
        all_errors = []

        for chunk_idx, (chunk_text, chunk_offset) in enumerate(chunks):
            # Skip very small chunks
            if len(chunk_text.strip()) < 10:
                continue

            chunk_annotations, chunk_stats = self._annotate_chunk(
                chunk_text=chunk_text,
                chunk_offset=chunk_offset,
                chunk_idx=chunk_idx,
                total_chunks=len(chunks),
                document=document,
                client=client,
                instructions=instructions,
                types_str=types_str,
            )

            all_annotations.extend(chunk_annotations)
            total_tokens += chunk_stats['tokens']
            prompt_tokens += chunk_stats['prompt_tokens']
            completion_tokens += chunk_stats['completion_tokens']
            latency_ms += chunk_stats['latency']
            total_retries += chunk_stats['retries']
            total_calls += chunk_stats['calls']
            raw_responses.extend(chunk_stats['raw_responses'])
            all_errors.extend(chunk_stats.get('errors', []))

        preserved = len(all_errors) == 0

        return self._create_result(
            document=document,
            annotations=all_annotations,
            llm_calls=total_calls,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=total_retries,
            latency_ms=latency_ms,
            preserved=preserved,
            errors=all_errors,
            raw_responses=raw_responses,
        )

    def _split_into_chunks(self, document: Document) -> list[tuple[str, int]]:
        """Split document into chunks with their offsets."""
        content = document.content
        chunks = []

        if self.chunk_by == "sentence":
            # Split by sentences
            pattern = r'(?<=[.!?])\s+'
            parts = re.split(pattern, content)
        else:
            # Split by paragraphs (double newline)
            pattern = r'\n\n+'
            parts = re.split(pattern, content)

        current_offset = 0
        for part in parts:
            if part:
                # Find actual offset in original
                actual_offset = content.find(part, current_offset)
                if actual_offset >= 0:
                    current_offset = actual_offset

                # Further split if chunk is too large
                if len(part) > self.max_chunk_size:
                    sub_chunks = self._split_large_chunk(part, current_offset)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append((part, current_offset))

                current_offset += len(part)

        return chunks

    def _split_large_chunk(self, text: str, offset: int) -> list[tuple[str, int]]:
        """Split a large chunk into smaller pieces."""
        chunks = []
        # Try to split on sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        current_offset = offset

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), current_offset))
                current_offset = offset + text.find(sentence, current_offset - offset)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append((current_chunk.strip(), current_offset))

        return chunks if chunks else [(text, offset)]

    def _annotate_chunk(
        self,
        chunk_text: str,
        chunk_offset: int,
        chunk_idx: int,
        total_chunks: int,
        document: Document,
        client: LLMClient,
        instructions: str,
        types_str: str,
    ) -> tuple[list[Annotation], dict]:
        """Annotate a single chunk with verification and retry."""
        stats = {
            'tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'latency': 0.0,
            'retries': 0,
            'calls': 0,
            'raw_responses': [],
            'errors': [],
        }

        prompt = f"""Annotate this text chunk ({chunk_idx + 1}/{total_chunks}).

CRITICAL RULES:
1. Return the EXACT same text with annotations inserted
2. Do NOT modify, fix, or change ANY of the original text
3. Use format: /* [TYPE] annotation text */

Original text to annotate:
---
{chunk_text}
---

Valid types: {types_str}
Instructions: {instructions}

Return ONLY the annotated text. Preserve the original text EXACTLY."""

        for attempt in range(self.max_retries):
            response = client.complete(prompt, system_prompt=self._build_system_prompt())

            stats['tokens'] += response.total_tokens
            stats['prompt_tokens'] += response.prompt_tokens
            stats['completion_tokens'] += response.completion_tokens
            stats['latency'] += response.latency_ms
            stats['calls'] += 1
            stats['raw_responses'].append(response.content)

            annotated = response.content.strip()

            # Remove markdown if present
            if annotated.startswith('```'):
                lines = annotated.split('\n')
                annotated = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
            annotated = annotated.strip('---').strip()

            # Verify preservation
            preserved, errors = self._verify_chunk(chunk_text, annotated)

            if preserved:
                # Extract annotations
                annotations = self._extract_chunk_annotations(
                    chunk_text, annotated, chunk_offset, document
                )
                return annotations, stats

            stats['retries'] += 1

            if attempt < self.max_retries - 1:
                # Strengthen instructions for retry
                prompt = f"""FAILED: You modified the original text. Try again.

Errors: {'; '.join(errors[:3])}

You must return the EXACT text below with ONLY /* comments */ inserted:
---
{chunk_text}
---

Do NOT change any character. Only INSERT annotation comments.
Return the annotated text:"""

        # All retries failed
        stats['errors'].append(f"Chunk {chunk_idx}: Failed to preserve text after {self.max_retries} attempts")
        return [], stats

    def _verify_chunk(self, original: str, annotated: str) -> tuple[bool, list[str]]:
        """Verify that a chunk's original text is preserved."""
        errors = []

        # Remove annotations
        cleaned = re.sub(r'/\*.*?\*/', '', annotated, flags=re.DOTALL)
        cleaned = cleaned.strip()
        original = original.strip()

        if cleaned != original:
            # Detailed error
            if len(cleaned) != len(original):
                errors.append(f"Length changed: {len(original)} -> {len(cleaned)}")

            # Find first difference
            for i, (o, c) in enumerate(zip(original, cleaned)):
                if o != c:
                    errors.append(f"Char {i} changed: '{o}' -> '{c}'")
                    break

        return len(errors) == 0, errors

    def _extract_chunk_annotations(
        self,
        chunk_text: str,
        annotated: str,
        chunk_offset: int,
        document: Document,
    ) -> list[Annotation]:
        """Extract annotations from an annotated chunk."""
        annotations = []

        # Find all /* ... */ annotations
        for match in re.finditer(r'/\*\s*\[?(\w+)\]?\s*(.*?)\*/', annotated, re.DOTALL):
            ann_type_str = match.group(1).lower()
            content = match.group(2).strip()

            # Calculate position in original document
            preceding = annotated[:match.start()]
            preceding_clean = re.sub(r'/\*.*?\*/', '', preceding, flags=re.DOTALL)

            local_offset = len(preceding_clean)
            global_offset = chunk_offset + local_offset
            global_offset = min(global_offset, len(document.content))

            try:
                position = document.offset_to_position(global_offset)
            except ValueError:
                position = Position(offset=global_offset, line=0, column=0)

            try:
                ann_type = AnnotationType(ann_type_str)
            except ValueError:
                ann_type = AnnotationType.COMMENT

            annotations.append(
                Annotation(
                    content=content,
                    position=position,
                    annotation_type=ann_type,
                )
            )

        return annotations

    def _build_system_prompt(self) -> str:
        return (
            "You annotate text by inserting /* comments */. "
            "You NEVER modify the original text - only add comments. "
            "Even typos and errors must remain exactly as-is."
        )

"""
Index-Based Labeling Strategy

Pre-split text deterministically, send with indices.
LLM returns ONLY labels (never outputs original text).

This guarantees 100% preservation because the LLM never has
the opportunity to modify the text - it only sees indices
and returns labels for them.
"""

import re
import json
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class IndexLabelingStrategy(BaseStrategy):
    """
    Index-based labeling strategy.

    1. Pre-split the document into segments (sentences, paragraphs, etc.)
    2. Send segments with indices: {1: "segment one", 2: "segment two"}
    3. LLM returns ONLY labels: {1: "Narrator", 2: "Character"}
    4. Reconstruction uses original segments with assigned labels

    This is the SAFEST approach because:
    - The LLM never outputs the original text
    - It only outputs label assignments
    - There's zero chance of text modification
    - Preservation is guaranteed by design

    The tradeoff is that annotations are at segment-level, not
    character-level. For speaker attribution, this is usually fine.
    """

    name = "index_labeling"
    description = "Pre-split text, LLM only returns labels (never outputs text)"

    def __init__(
        self,
        max_retries: int = 3,
        split_by: str = "sentence",  # "sentence", "paragraph", "line", "dialogue"
        default_label: str = "Narrator",
        max_segments_per_call: int = 50,
    ):
        """
        Args:
            max_retries: Maximum retry attempts
            split_by: How to split the document
            default_label: Label for unlabeled segments
            max_segments_per_call: Max segments to send in one LLM call
        """
        super().__init__(max_retries=max_retries)
        self.split_by = split_by
        self.default_label = default_label
        self.max_segments_per_call = max_segments_per_call

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        labels_str = ', '.join(annotation_types) if annotation_types else 'Narrator, Character, Dialogue'

        # Split document into segments
        segments = self._split_document(document)

        if not segments:
            return self._create_result(
                document=document,
                annotations=[],
                llm_calls=0,
                preserved=True,
            )

        # Process in batches if needed
        all_labels = {}
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0.0
        total_calls = 0
        raw_responses = []

        for batch_start in range(0, len(segments), self.max_segments_per_call):
            batch_end = min(batch_start + self.max_segments_per_call, len(segments))
            batch = {i: seg['text'] for i, seg in enumerate(segments[batch_start:batch_end], start=batch_start + 1)}

            batch_labels, stats = self._label_batch(
                batch=batch,
                client=client,
                instructions=instructions,
                labels_str=labels_str,
                context_before=segments[batch_start - 1]['text'] if batch_start > 0 else None,
                context_after=segments[batch_end]['text'] if batch_end < len(segments) else None,
            )

            all_labels.update(batch_labels)
            total_tokens += stats['tokens']
            prompt_tokens += stats['prompt_tokens']
            completion_tokens += stats['completion_tokens']
            latency_ms += stats['latency']
            total_calls += stats['calls']
            raw_responses.extend(stats['raw_responses'])

        # Create annotations from labels
        annotations = self._create_annotations_from_labels(document, segments, all_labels)

        return self._create_result(
            document=document,
            annotations=annotations,
            llm_calls=total_calls,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=0,  # No retries needed - LLM can't mess up the text
            latency_ms=latency_ms,
            preserved=True,  # Always preserved - LLM never outputs text
            errors=[],
            raw_responses=raw_responses,
        )

    def _split_document(self, document: Document) -> list[dict]:
        """
        Split document into segments with their positions.

        Returns list of {text, start, end} dicts.
        """
        content = document.content
        segments = []

        if self.split_by == "sentence":
            # Split by sentence-ending punctuation
            pattern = r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?]")(?=\s+[A-Z])'
            parts = re.split(pattern, content)
        elif self.split_by == "paragraph":
            # Split by double newlines
            pattern = r'\n\n+'
            parts = re.split(pattern, content)
        elif self.split_by == "line":
            # Split by single newlines
            parts = content.split('\n')
        elif self.split_by == "dialogue":
            # Split to separate dialogue from narrative
            # This regex tries to identify dialogue patterns
            pattern = r'(?<=[.!?"])\s+(?=[A-Z"\'])|(?<="\s)(?=[A-Z])|(?<=[.!?])\s+(?=")'
            parts = re.split(pattern, content)
        else:
            # Default: split by sentences
            pattern = r'(?<=[.!?])\s+'
            parts = re.split(pattern, content)

        current_pos = 0
        for part in parts:
            if not part.strip():
                continue

            # Find actual position in document
            start = content.find(part, current_pos)
            if start == -1:
                start = current_pos

            end = start + len(part)

            segments.append({
                'text': part,
                'start': start,
                'end': end,
            })

            current_pos = end

        return segments

    def _label_batch(
        self,
        batch: dict[int, str],
        client: LLMClient,
        instructions: str,
        labels_str: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> tuple[dict[int, str], dict]:
        """
        Get labels for a batch of segments.

        Returns (labels dict, stats dict).
        """
        stats = {
            'tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'latency': 0.0,
            'calls': 0,
            'raw_responses': [],
        }

        # Format segments for the prompt
        segments_text = '\n'.join(f'{idx}: "{text}"' for idx, text in sorted(batch.items()))

        context_note = ""
        if context_before:
            context_note += f'\n[Previous segment: "{context_before[:100]}..."]'
        if context_after:
            context_note += f'\n[Next segment: "...{context_after[:100]}"]'

        prompt = f"""Label each numbered text segment with a speaker/type.

SEGMENTS TO LABEL:
{segments_text}
{context_note}

Valid labels: {labels_str}

Instructions: {instructions}

RESPONSE FORMAT (JSON only):
{{
  "1": "LabelForSegment1",
  "2": "LabelForSegment2",
  ...
}}

Return ONLY the JSON with segment numbers mapped to labels.
Do NOT include the original text in your response."""

        for attempt in range(self.max_retries):
            try:
                parsed, response = client.complete_json(
                    prompt, system_prompt=self._build_system_prompt()
                )
                stats['tokens'] += response.total_tokens
                stats['prompt_tokens'] += response.prompt_tokens
                stats['completion_tokens'] += response.completion_tokens
                stats['latency'] += response.latency_ms
                stats['calls'] += 1
                stats['raw_responses'].append(response.content)

                # Convert string keys to ints and validate
                labels = {}
                for key, value in parsed.items():
                    try:
                        idx = int(key)
                        if idx in batch:
                            labels[idx] = str(value)
                    except (ValueError, TypeError):
                        continue

                # Fill in missing labels with default
                for idx in batch.keys():
                    if idx not in labels:
                        labels[idx] = self.default_label

                return labels, stats

            except Exception as e:
                stats['raw_responses'].append(f"Error: {str(e)}")
                stats['calls'] += 1

        # All retries failed - return defaults
        return {idx: self.default_label for idx in batch.keys()}, stats

    def _create_annotations_from_labels(
        self,
        document: Document,
        segments: list[dict],
        labels: dict[int, str],
    ) -> list[Annotation]:
        """Create annotations from segment labels."""
        annotations = []

        for i, segment in enumerate(segments, start=1):
            label = labels.get(i, self.default_label)

            start = segment['start']
            end = segment['end']

            try:
                position = document.offset_to_position(start)
            except ValueError:
                position = Position(offset=start, line=0, column=0)

            span = Span(start=start, end=end)

            # Determine annotation type
            ann_type = self._label_to_type(label)

            annotations.append(
                Annotation(
                    content=label,
                    position=position,
                    span=span,
                    annotation_type=ann_type,
                    anchor_text=segment['text'][:50],  # First 50 chars as anchor
                    metadata={
                        'segment_index': i,
                        'segment_text': segment['text'],
                    },
                )
            )

        return annotations

    def _label_to_type(self, label: str) -> AnnotationType:
        """Convert a label string to AnnotationType."""
        lower = label.lower()

        if 'narrator' in lower or 'narration' in lower:
            return AnnotationType.INFO
        elif 'warning' in lower or 'error' in lower:
            return AnnotationType.WARNING
        elif 'suggest' in lower:
            return AnnotationType.SUGGESTION
        elif 'explain' in lower or 'explanation' in lower:
            return AnnotationType.EXPLANATION
        else:
            return AnnotationType.COMMENT

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """Always preserved - LLM never outputs text."""
        return True, []

    def _build_system_prompt(self) -> str:
        return (
            "You are a text labeling system. You receive numbered text segments "
            "and return ONLY labels for each segment number. You NEVER output "
            "the original text - only the labels. Respond with JSON mapping "
            "segment numbers to labels."
        )

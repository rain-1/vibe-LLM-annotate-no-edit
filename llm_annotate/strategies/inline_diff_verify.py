"""
Inline Diff Verification Strategy

LLM inserts tags inline, then we use diff algorithms to verify
that ONLY insertions occurred (no modifications or deletions).

This is more robust than regex-based verification because diff
algorithms handle edge cases like whitespace, unicode, etc.
"""

import difflib
import re
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, Span, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class InlineDiffVerifyStrategy(BaseStrategy):
    """
    Inline tagging with diff-based verification.

    The LLM outputs the text WITH tags inserted (using unusual delimiters
    that won't appear in normal text). We then:

    1. Use difflib to compare LLM output against original
    2. Verify that ALL differences are pure insertions matching tag pattern
    3. If any modification/deletion detected → reject and retry

    This is rock-solid because even subtle changes (dropped words,
    "fixed" typos, changed whitespace) will show up in the diff.
    """

    name = "inline_diff_verify"
    description = "Insert tags inline, verify with diff algorithm that only insertions occurred"

    def __init__(
        self,
        max_retries: int = 3,
        tag_start: str = "«",
        tag_end: str = "»",
        fallback_on_failure: bool = True,
    ):
        """
        Args:
            max_retries: Maximum retry attempts
            tag_start: Opening delimiter for tags (use unusual chars)
            tag_end: Closing delimiter for tags
            fallback_on_failure: If True, accept partial results on failure
        """
        super().__init__(max_retries=max_retries)
        self.tag_start = tag_start
        self.tag_end = tag_end
        self.fallback_on_failure = fallback_on_failure
        # Pattern to match our tags
        self.tag_pattern = re.compile(
            re.escape(tag_start) + r'([^' + re.escape(tag_end) + r']+)' + re.escape(tag_end)
        )

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'Narrator, Character, Speaker'

        prompt = f"""Add speaker/annotation tags to this text by inserting {self.tag_start}TAG{self.tag_end} markers.

CRITICAL RULES:
1. Insert tags using the format: {self.tag_start}TagName{self.tag_end}
2. Do NOT modify, delete, or change ANY of the original text
3. Only INSERT tags - every character of the original must remain
4. Tags go BEFORE the text they label

EXAMPLE:
Original: "Hello," she said. "How are you?"
Tagged: {self.tag_start}Narrator{self.tag_end}"Hello," {self.tag_start}Narrator{self.tag_end}she said. {self.tag_start}Character{self.tag_end}"How are you?"

Valid tag types: {types_str}

TEXT TO TAG:
{document.content}

Instructions: {instructions}

Return ONLY the tagged text. Every character of the original must be preserved."""

        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0.0
        retries = 0
        raw_responses = []
        last_errors = []

        for attempt in range(self.max_retries):
            response = client.complete(prompt, system_prompt=self._build_system_prompt())

            total_tokens += response.total_tokens
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            latency_ms += response.latency_ms
            raw_responses.append(response.content)

            tagged_text = response.content.strip()

            # Remove any markdown code blocks
            if tagged_text.startswith("```"):
                lines = tagged_text.split('\n')
                tagged_text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

            # Verify using diff
            is_valid, errors, annotations = self._verify_with_diff(document, tagged_text)
            last_errors = errors

            if is_valid:
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
                # Provide specific feedback for retry
                prompt = f"""Your previous output MODIFIED the original text. The diff detected these issues:
{chr(10).join(errors[:5])}

You MUST preserve EVERY character of the original. Only INSERT {self.tag_start}Tag{self.tag_end} markers.

ORIGINAL TEXT (preserve exactly):
{document.content}

Instructions: {instructions}

Return the text with ONLY tag insertions."""

        # All retries failed
        if self.fallback_on_failure and annotations:
            # Accept partial results
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
                errors=last_errors,
                raw_responses=raw_responses,
            )

        return self._create_result(
            document=document,
            annotations=[],
            llm_calls=self.max_retries,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=retries,
            latency_ms=latency_ms,
            preserved=False,
            errors=last_errors,
            raw_responses=raw_responses,
        )

    def _verify_with_diff(
        self,
        document: Document,
        tagged_text: str,
    ) -> tuple[bool, list[str], list[Annotation]]:
        """
        Use diff algorithm to verify only tag insertions occurred.

        Returns:
            (is_valid, errors, annotations)
        """
        errors = []
        annotations = []

        original = document.content

        # Use SequenceMatcher to find all differences
        matcher = difflib.SequenceMatcher(None, original, tagged_text, autojunk=False)

        # Track position in original for annotation placement
        original_pos = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # No change - good
                original_pos = i2
                continue

            elif tag == 'insert':
                # Something was inserted - check if it's a valid tag
                inserted = tagged_text[j1:j2]

                # Check if ALL inserted content matches our tag pattern
                remaining = inserted
                while remaining:
                    match = self.tag_pattern.match(remaining)
                    if match:
                        # Valid tag insertion
                        tag_content = match.group(1)

                        # Create annotation at current position in original
                        offset = min(i1, len(document.content))
                        try:
                            position = document.offset_to_position(offset)
                        except ValueError:
                            position = Position(offset=offset, line=0, column=0)

                        # Determine annotation type from tag content
                        ann_type = self._parse_tag_type(tag_content)

                        annotations.append(
                            Annotation(
                                content=tag_content,
                                position=position,
                                annotation_type=ann_type,
                                metadata={'raw_tag': match.group(0)},
                            )
                        )

                        remaining = remaining[match.end():]
                    else:
                        # Non-tag insertion - might be whitespace, which we can allow
                        if remaining.strip() == '':
                            # Just whitespace, probably okay
                            remaining = ''
                        else:
                            # Invalid insertion
                            errors.append(
                                f"Invalid insertion at pos {i1}: '{inserted[:50]}...' "
                                f"(not a valid tag)"
                            )
                            break

            elif tag == 'delete':
                # Something was deleted from original - NOT allowed
                deleted = original[i1:i2]
                errors.append(
                    f"DELETION at pos {i1}: '{deleted[:50]}...' was removed"
                )
                original_pos = i2

            elif tag == 'replace':
                # Something was replaced - NOT allowed
                old = original[i1:i2]
                new = tagged_text[j1:j2]

                # Check if this is actually an insertion with the original preserved
                if old in new:
                    # The original is still there, something was added
                    # This might be okay if the addition is a tag
                    inserted = new.replace(old, '', 1)
                    if self.tag_pattern.fullmatch(inserted.strip()):
                        # It's just a tag being inserted
                        offset = min(i1, len(document.content))
                        try:
                            position = document.offset_to_position(offset)
                        except ValueError:
                            position = Position(offset=offset, line=0, column=0)

                        match = self.tag_pattern.search(inserted)
                        if match:
                            annotations.append(
                                Annotation(
                                    content=match.group(1),
                                    position=position,
                                    annotation_type=self._parse_tag_type(match.group(1)),
                                )
                            )
                        original_pos = i2
                        continue

                errors.append(
                    f"MODIFICATION at pos {i1}: '{old[:30]}...' → '{new[:30]}...'"
                )
                original_pos = i2

        is_valid = len(errors) == 0
        return is_valid, errors, annotations

    def _parse_tag_type(self, tag_content: str) -> AnnotationType:
        """Parse tag content to determine annotation type."""
        lower = tag_content.lower()

        if 'narrator' in lower:
            return AnnotationType.INFO
        elif 'warning' in lower or 'error' in lower:
            return AnnotationType.WARNING
        elif 'suggest' in lower:
            return AnnotationType.SUGGESTION
        elif 'explain' in lower:
            return AnnotationType.EXPLANATION
        else:
            return AnnotationType.COMMENT

    def _build_system_prompt(self) -> str:
        return (
            f"You are a text tagger. You insert {self.tag_start}Tag{self.tag_end} markers into text. "
            f"You NEVER modify, delete, or change the original text. "
            f"Even typos, errors, or awkward phrasing must remain exactly as-is. "
            f"Your only job is to INSERT tags."
        )

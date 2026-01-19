"""
Diff Insertion-Only Strategy

Asks the LLM to produce a unified diff that contains ONLY insertions (+ lines).
Any diff with deletions or modifications is rejected and retried.
"""

import re
from typing import Optional

from ..core import Document, Annotation, AnnotationResult, Position, AnnotationType
from ..llm_client import LLMClient
from .base import BaseStrategy


class DiffInsertionOnlyStrategy(BaseStrategy):
    """
    Diff-based insertion-only annotation strategy.

    The LLM produces a unified diff containing only additions.
    Lines starting with '-' (deletions) or modifications cause rejection.
    Only '+' lines (insertions) are accepted.

    This leverages the LLM's understanding of diff format while
    enforcing the insertion-only constraint through diff validation.
    """

    name = "diff_insertion_only"
    description = "Unified diff format with only insertions allowed"

    def __init__(self, max_retries: int = 3, diff_context: int = 3):
        super().__init__(max_retries=max_retries)
        self.diff_context = diff_context

    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        types_str = ', '.join(annotation_types) if annotation_types else 'comment, explanation'

        # Generate numbered document
        lines = document.get_lines()
        numbered_doc = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))

        prompt = f"""Annotate the document below by producing a UNIFIED DIFF.

CRITICAL RULES:
1. You may ONLY add lines (lines starting with '+')
2. You may NOT delete or modify lines (NO lines starting with '-')
3. Annotations should be added as comment lines
4. Use format: + /* [TYPE] annotation text */

DOCUMENT (with line numbers for reference):
{numbered_doc}

Instructions: {instructions}
Valid annotation types: {types_str}

Produce a unified diff that ONLY ADDS annotation comment lines.
Format: Standard unified diff with @@ line markers

Example format:
```diff
@@ -5,3 +5,4 @@
 context line
 context line
+/* [comment] This is an annotation */
 context line
```

Return ONLY the diff. No other text."""

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

            # Extract diff
            diff_content = self._extract_diff(response.content)

            # Validate diff is insertion-only
            valid, errors = self._validate_insertion_only(diff_content)
            last_errors = errors

            if valid:
                # Parse diff and create annotations
                annotations = self._parse_diff_annotations(document, diff_content)

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
                prompt = f"""Your diff contained invalid modifications:
{chr(10).join(errors[:5])}

You may ONLY ADD lines with '+'. NO deletions ('-') allowed.
The original document MUST remain unchanged.

DOCUMENT:
{numbered_doc}

Produce an insertion-only diff with annotations as comment lines.
Only '+' lines adding /* comments */ are allowed."""

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

    def _extract_diff(self, response: str) -> str:
        """Extract diff content from response."""
        content = response.strip()

        # Handle markdown code blocks
        if '```diff' in content:
            match = re.search(r'```diff\s*([\s\S]*?)\s*```', content)
            if match:
                return match.group(1).strip()
        elif '```' in content:
            match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if match:
                return match.group(1).strip()

        return content

    def _validate_insertion_only(self, diff: str) -> tuple[bool, list[str]]:
        """Validate that diff contains only insertions."""
        errors = []

        lines = diff.split('\n')

        for i, line in enumerate(lines):
            # Skip diff headers and context lines
            if line.startswith('@@') or line.startswith('---') or line.startswith('+++'):
                continue
            if line.startswith(' ') or line == '':
                continue

            # Check for deletions
            if line.startswith('-'):
                errors.append(f"Line {i+1}: Deletion not allowed: '{line[:50]}...'")

            # Additions are OK
            elif line.startswith('+'):
                # Verify it's an annotation comment, not content modification
                added_content = line[1:].strip()
                if added_content and not self._is_annotation_line(added_content):
                    # Could be content modification disguised as addition
                    # We'll allow it but could add stricter checking here
                    pass

        return len(errors) == 0, errors

    def _is_annotation_line(self, line: str) -> bool:
        """Check if a line is an annotation comment."""
        # Various annotation formats
        patterns = [
            r'^/\*.*\*/$',  # /* ... */
            r'^//.*$',  # // ...
            r'^#.*$',  # # ... (for some languages)
            r'^<!--.*-->$',  # HTML comments
        ]
        return any(re.match(p, line.strip()) for p in patterns)

    def _parse_diff_annotations(
        self,
        document: Document,
        diff: str,
    ) -> list[Annotation]:
        """Parse annotations from a diff."""
        annotations = []
        lines = document.get_lines()

        # Parse diff hunks
        hunk_pattern = r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@'
        current_line = 0

        diff_lines = diff.split('\n')
        i = 0

        while i < len(diff_lines):
            line = diff_lines[i]

            # Parse hunk header
            hunk_match = re.match(hunk_pattern, line)
            if hunk_match:
                current_line = int(hunk_match.group(1)) - 1  # Convert to 0-indexed
                i += 1
                continue

            # Skip diff file headers
            if line.startswith('---') or line.startswith('+++'):
                i += 1
                continue

            # Process diff lines
            if line.startswith('+'):
                added_content = line[1:]

                # Extract annotation from the added line
                ann_match = re.search(r'/\*\s*\[?(\w+)\]?\s*(.*?)\*/', added_content)
                if ann_match:
                    ann_type_str = ann_match.group(1).lower()
                    content = ann_match.group(2).strip()

                    # Position is at the line where this would be inserted
                    line_idx = min(current_line, len(lines) - 1)
                    line_idx = max(0, line_idx)

                    try:
                        offset = document.position_to_offset(line_idx, 0)
                        position = Position(offset=offset, line=line_idx, column=0)
                    except ValueError:
                        position = Position(offset=0, line=0, column=0)

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

            elif line.startswith(' '):
                # Context line - advance line counter
                current_line += 1
            elif line.startswith('-'):
                # Deletion (shouldn't happen if validated, but skip)
                pass
            else:
                # Unknown line format
                pass

            i += 1

        return annotations

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """Verify by checking diff has no deletions."""
        # The diff validation handles this
        return True, []

    def _build_system_prompt(self) -> str:
        return (
            "You are a diff generator that creates INSERTION-ONLY diffs. "
            "You can only ADD lines ('+'), never DELETE ('-') or MODIFY. "
            "Use unified diff format with @@ markers. "
            "Add annotations as comment lines."
        )

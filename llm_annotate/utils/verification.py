"""
Verification utilities for checking document preservation.

The core technique: strip all annotations from the output,
then diff the stripped version against the original.
Any diff = modification detected.
"""

import difflib
import re
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class VerificationResult:
    """Result of verifying document preservation."""
    preserved: bool
    errors: list[str]
    diff_lines: list[str]  # Unified diff output for debugging
    stripped_output: str  # The output after stripping annotations


def strip_and_diff(
    original: str,
    annotated: str,
    strip_pattern: Optional[str] = None,
    strip_func: Optional[Callable[[str], str]] = None,
) -> VerificationResult:
    """
    Strip annotations from output, then diff against original.

    This is the most reliable verification approach:
    1. Remove all annotations from the LLM output
    2. Diff the stripped result against the original
    3. Any differences = document was modified

    Args:
        original: The original document text
        annotated: The LLM's annotated output
        strip_pattern: Regex pattern to remove annotations (e.g., r'/\\*.*?\\*/')
        strip_func: Custom function to strip annotations (alternative to pattern)

    Returns:
        VerificationResult with preservation status and any errors
    """
    # Strip annotations
    if strip_func:
        stripped = strip_func(annotated)
    elif strip_pattern:
        stripped = re.sub(strip_pattern, '', annotated, flags=re.DOTALL)
    else:
        # Default: try common annotation patterns
        stripped = annotated
        patterns = [
            r'/\*.*?\*/',  # /* ... */
            r'//[^\n]*',  # // ...
            r'#[^\n]*',  # # ...
            r'<!--.*?-->',  # <!-- ... -->
            r'«[^»]*»',  # « ... »
            r'\[\[[^\]]*\]\]',  # [[ ... ]]
            r'<ann[^>]*>.*?</ann>',  # <ann>...</ann>
        ]
        for pattern in patterns:
            stripped = re.sub(pattern, '', stripped, flags=re.DOTALL)

    # Generate unified diff
    original_lines = original.splitlines(keepends=True)
    stripped_lines = stripped.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        original_lines,
        stripped_lines,
        fromfile='original',
        tofile='stripped',
        lineterm='',
    ))

    # Analyze diff for errors
    errors = []

    if diff:
        # There are differences
        for line in diff:
            if line.startswith('-') and not line.startswith('---'):
                # Something was removed from original
                errors.append(f"DELETED: {line[1:].strip()[:50]}...")
            elif line.startswith('+') and not line.startswith('+++'):
                # Something was added (not stripped properly?)
                errors.append(f"RESIDUAL: {line[1:].strip()[:50]}...")

    # Also do a direct comparison for subtle issues
    if stripped != original:
        if not errors:
            # Diff didn't catch it - must be whitespace or encoding
            if stripped.replace(' ', '').replace('\n', '') != original.replace(' ', '').replace('\n', ''):
                errors.append("Content differs after normalization")
            else:
                errors.append("Whitespace differences detected")

    return VerificationResult(
        preserved=len(errors) == 0,
        errors=errors,
        diff_lines=diff,
        stripped_output=stripped,
    )


def verify_insertion_only(
    original: str,
    modified: str,
    allowed_insertions: Optional[Callable[[str], bool]] = None,
) -> VerificationResult:
    """
    Verify that only insertions occurred (no deletions or modifications).

    Uses difflib.SequenceMatcher to analyze the exact changes.

    Args:
        original: Original document text
        modified: Modified text (with annotations inserted)
        allowed_insertions: Optional function to validate each insertion

    Returns:
        VerificationResult
    """
    errors = []
    insertions = []

    matcher = difflib.SequenceMatcher(None, original, modified, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        elif tag == 'insert':
            inserted = modified[j1:j2]
            insertions.append(inserted)

            # Validate insertion if checker provided
            if allowed_insertions and not allowed_insertions(inserted):
                errors.append(f"Invalid insertion at {i1}: '{inserted[:50]}...'")

        elif tag == 'delete':
            deleted = original[i1:i2]
            errors.append(f"DELETION at {i1}: '{deleted[:50]}...'")

        elif tag == 'replace':
            old = original[i1:i2]
            new = modified[j1:j2]
            errors.append(f"REPLACEMENT at {i1}: '{old[:30]}...' → '{new[:30]}...'")

    # Generate diff for debugging
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile='original',
        tofile='modified',
    ))

    return VerificationResult(
        preserved=len(errors) == 0,
        errors=errors,
        diff_lines=diff,
        stripped_output=modified,  # Not stripped in this mode
    )


def make_tag_stripper(tag_start: str, tag_end: str) -> Callable[[str], str]:
    """
    Create a stripper function for custom tag delimiters.

    Example:
        stripper = make_tag_stripper("«", "»")
        clean = stripper("«Speaker»Hello world")  # "Hello world"
    """
    pattern = re.escape(tag_start) + r'[^' + re.escape(tag_end) + r']*' + re.escape(tag_end)

    def stripper(text: str) -> str:
        return re.sub(pattern, '', text)

    return stripper


def make_pattern_validator(pattern: str) -> Callable[[str], bool]:
    """
    Create a validator that checks if insertions match a pattern.

    Example:
        validator = make_pattern_validator(r'«\w+»')
        validator("«Speaker»")  # True
        validator("random text")  # False
    """
    compiled = re.compile(pattern)

    def validator(text: str) -> bool:
        # Allow whitespace-only insertions
        if not text.strip():
            return True
        return bool(compiled.fullmatch(text.strip()))

    return validator

"""Utility functions for the LLM annotation library."""

from .verification import (
    VerificationResult,
    strip_and_diff,
    verify_insertion_only,
    make_tag_stripper,
    make_pattern_validator,
)

__all__ = [
    "VerificationResult",
    "strip_and_diff",
    "verify_insertion_only",
    "make_tag_stripper",
    "make_pattern_validator",
]

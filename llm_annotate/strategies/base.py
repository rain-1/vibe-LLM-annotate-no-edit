"""
Base class for annotation strategies.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

from ..core import Document, Annotation, AnnotationResult
from ..llm_client import LLMClient


class BaseStrategy(ABC):
    """
    Abstract base class for annotation strategies.

    Each strategy must implement the annotate() method which takes
    a document and annotation instructions, and returns annotations
    without modifying the original document.
    """

    name: str = "base"
    description: str = "Base strategy"

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    @abstractmethod
    def annotate(
        self,
        document: Document,
        client: LLMClient,
        instructions: str,
        annotation_types: Optional[list[str]] = None,
    ) -> AnnotationResult:
        """
        Annotate a document using this strategy.

        Args:
            document: The document to annotate
            client: LLM client to use for generation
            instructions: Instructions for what to annotate
            annotation_types: Types of annotations to generate

        Returns:
            AnnotationResult containing annotations and metrics
        """
        pass

    def verify_preservation(
        self,
        original: Document,
        annotated_text: str,
    ) -> tuple[bool, list[str]]:
        """
        Verify that the original document content is preserved.

        This base implementation checks if the original text can be
        recovered from the annotated text by removing annotations.

        Returns:
            Tuple of (is_preserved, list of error messages)
        """
        # Default implementation - subclasses should override
        return True, []

    def _create_result(
        self,
        document: Document,
        annotations: list[Annotation],
        llm_calls: int = 0,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        retries: int = 0,
        latency_ms: float = 0.0,
        preserved: bool = True,
        errors: Optional[list[str]] = None,
        raw_responses: Optional[list[str]] = None,
    ) -> AnnotationResult:
        """Helper to create an AnnotationResult."""
        return AnnotationResult(
            document=document,
            annotations=annotations,
            strategy_name=self.name,
            llm_calls=llm_calls,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=retries,
            latency_ms=latency_ms,
            document_preserved=preserved,
            preservation_errors=errors or [],
            raw_llm_responses=raw_responses or [],
        )

    def _build_system_prompt(self) -> str:
        """Build a system prompt for the LLM. Override in subclasses."""
        return (
            "You are a precise document annotator. Your task is to add "
            "annotations to documents without modifying the original content. "
            "Follow the instructions exactly."
        )

"""
Main Annotator class that orchestrates annotation using different strategies.
"""

from typing import Optional, Union

from .core import Document, AnnotationResult
from .llm_client import LLMClient, get_client
from .strategies import BaseStrategy, get_strategy, list_strategies


class Annotator:
    """
    High-level interface for annotating documents.

    Supports multiple annotation strategies and provides a unified
    interface for annotation operations.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        provider: str = "anthropic",
        default_strategy: str = "offset_based",
        **client_kwargs,
    ):
        """
        Initialize the Annotator.

        Args:
            client: Pre-configured LLM client (optional)
            provider: LLM provider if client not provided ("anthropic", "openai", "mock")
            default_strategy: Default annotation strategy to use
            **client_kwargs: Additional arguments for the LLM client
        """
        self.client = client or get_client(provider, **client_kwargs)
        self.default_strategy = default_strategy

    def annotate(
        self,
        document: Union[Document, str],
        instructions: str,
        strategy: Optional[Union[str, BaseStrategy]] = None,
        annotation_types: Optional[list[str]] = None,
        **strategy_kwargs,
    ) -> AnnotationResult:
        """
        Annotate a document.

        Args:
            document: Document object or raw text string to annotate
            instructions: Instructions for what to annotate
            strategy: Strategy name or instance (uses default if not specified)
            annotation_types: Types of annotations to generate
            **strategy_kwargs: Additional arguments for the strategy

        Returns:
            AnnotationResult with annotations and metrics
        """
        # Convert string to Document if needed
        if isinstance(document, str):
            document = Document(content=document)

        # Get strategy instance
        if strategy is None:
            strategy = get_strategy(self.default_strategy, **strategy_kwargs)
        elif isinstance(strategy, str):
            strategy = get_strategy(strategy, **strategy_kwargs)

        # Run annotation
        return strategy.annotate(
            document=document,
            client=self.client,
            instructions=instructions,
            annotation_types=annotation_types,
        )

    def annotate_with_all_strategies(
        self,
        document: Union[Document, str],
        instructions: str,
        annotation_types: Optional[list[str]] = None,
        strategies: Optional[list[str]] = None,
    ) -> dict[str, AnnotationResult]:
        """
        Annotate a document using multiple strategies for comparison.

        Args:
            document: Document to annotate
            instructions: Annotation instructions
            annotation_types: Types of annotations
            strategies: List of strategy names (all if not specified)

        Returns:
            Dictionary mapping strategy name to AnnotationResult
        """
        if isinstance(document, str):
            document = Document(content=document)

        strategy_names = strategies or list_strategies()
        results = {}

        for name in strategy_names:
            try:
                strategy = get_strategy(name)
                result = strategy.annotate(
                    document=document,
                    client=self.client,
                    instructions=instructions,
                    annotation_types=annotation_types,
                )
                results[name] = result
            except Exception as e:
                # Create error result
                results[name] = AnnotationResult(
                    document=document,
                    annotations=[],
                    strategy_name=name,
                    document_preserved=False,
                    preservation_errors=[f"Strategy failed: {str(e)}"],
                )

        return results

    @staticmethod
    def available_strategies() -> list[str]:
        """List available strategy names."""
        return list_strategies()

    @staticmethod
    def create_document(
        content: str,
        name: str = "untitled",
        metadata: Optional[dict] = None,
    ) -> Document:
        """Create a Document object."""
        return Document(
            content=content,
            name=name,
            metadata=metadata or {},
        )

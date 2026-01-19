"""
Annotation strategies for LLM-based document annotation.

Each strategy implements a different approach to getting annotations
from an LLM while preserving the original document.
"""

from .base import BaseStrategy
from .baseline_naive import BaselineNaiveStrategy
from .xml_inline import XMLInlineStrategy
from .marker_referenced import MarkerReferencedStrategy
from .offset_based import OffsetBasedStrategy
from .line_reference import LineReferenceStrategy
from .chunked_verified import ChunkedVerifiedStrategy
from .diff_insertion_only import DiffInsertionOnlyStrategy
from .anchor_based import AnchorBasedStrategy

__all__ = [
    "BaseStrategy",
    "BaselineNaiveStrategy",
    "XMLInlineStrategy",
    "MarkerReferencedStrategy",
    "OffsetBasedStrategy",
    "LineReferenceStrategy",
    "ChunkedVerifiedStrategy",
    "DiffInsertionOnlyStrategy",
    "AnchorBasedStrategy",
]

# Registry of all available strategies
STRATEGIES = {
    "baseline_naive": BaselineNaiveStrategy,
    "xml_inline": XMLInlineStrategy,
    "marker_referenced": MarkerReferencedStrategy,
    "offset_based": OffsetBasedStrategy,
    "line_reference": LineReferenceStrategy,
    "chunked_verified": ChunkedVerifiedStrategy,
    "diff_insertion_only": DiffInsertionOnlyStrategy,
    "anchor_based": AnchorBasedStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Get a strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


def list_strategies() -> list[str]:
    """List all available strategy names."""
    return list(STRATEGIES.keys())

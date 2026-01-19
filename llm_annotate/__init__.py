"""
LLM Document Annotation Library

A library for annotating documents using LLMs with guarantees
that the original document content is preserved (insertion-only annotations).
"""

from .core import Document, Annotation, AnnotationResult
from .annotator import Annotator
from .strategies import (
    BaseStrategy,
    BaselineNaiveStrategy,
    XMLInlineStrategy,
    MarkerReferencedStrategy,
    OffsetBasedStrategy,
    LineReferenceStrategy,
    ChunkedVerifiedStrategy,
    DiffInsertionOnlyStrategy,
    AnchorBasedStrategy,
    InlineDiffVerifyStrategy,
    IndexLabelingStrategy,
)

__version__ = "0.1.0"

__all__ = [
    "Document",
    "Annotation",
    "AnnotationResult",
    "Annotator",
    "BaseStrategy",
    "BaselineNaiveStrategy",
    "XMLInlineStrategy",
    "MarkerReferencedStrategy",
    "OffsetBasedStrategy",
    "LineReferenceStrategy",
    "ChunkedVerifiedStrategy",
    "DiffInsertionOnlyStrategy",
    "AnchorBasedStrategy",
    "InlineDiffVerifyStrategy",
    "IndexLabelingStrategy",
]

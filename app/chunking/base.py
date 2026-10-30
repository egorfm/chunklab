from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from app.ingestion.pdf_loader import ExtractedDocument, ExtractedBlock, TextSpan


@dataclass
class Chunk:
    chunk_index: int
    text: str
    page_number: int  # 1-based page number
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    char_length: int
    token_count: int
    strategy: str
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "char_length": self.char_length,
            "token_count": self.token_count,
            "strategy": self.strategy,
            "strategy_params": self.strategy_params,
        }


def compute_union_bbox(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Computes enclosing bounding box for a list of bounding boxes."""
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))


def map_text_to_spans(
    text: str,
    spans: List[TextSpan],
    fallback_bbox: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Finds spans whose text appears in the chunk text to compute an accurate union bbox."""
    if not spans:
        return fallback_bbox

    matched_bboxes = []
    chunk_lower = text.lower()
    for span in spans:
        # Check if word/span text is in chunk text
        clean_span = span.text.strip().lower()
        if clean_span and clean_span in chunk_lower:
            matched_bboxes.append(span.bbox)

    if matched_bboxes:
        return compute_union_bbox(matched_bboxes)
    return fallback_bbox


class BaseChunker(ABC):
    """Abstract base class for all coordinate-preserving chunkers."""

    def __init__(self, name: str, params: Dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def chunk_document(self, document: ExtractedDocument) -> List[Chunk]:
        """Split extracted document into chunks with preserved page and bbox coordinates."""
        pass

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional


@dataclass
class TextSpan:
    """Represents a fine-grained word or span with bounding box."""
    text: str
    page_number: int  # 1-based
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class ExtractedBlock:
    """Represents a structural block of text with coordinates from a PDF."""
    page_number: int  # 1-based
    block_index: int
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    spans: List[TextSpan] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    filename: str
    num_pages: int
    blocks: List[ExtractedBlock]
    page_dimensions: Dict[int, Tuple[float, float]]  # page_num -> (width, height)
    total_characters: int


class PDFLoader:
    """Loads PDF files and extracts text while preserving page numbers and 2D bounding box coordinates."""

    @staticmethod
    def load_pdf(file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")

        doc = fitz.open(str(path))
        num_pages = len(doc)
        blocks_list: List[ExtractedBlock] = []
        page_dimensions: Dict[int, Tuple[float, float]] = {}
        total_chars = 0

        for page_idx in range(num_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            page_dimensions[page_num] = (float(rect.width), float(rect.height))

            # Extract raw blocks: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type == 0 is text, 1 is image
            text_blocks = page.get_text("blocks")
            
            # Also extract words for fine-grained spans if needed:
            # (x0, y0, x1, y1, word, block_no, line_no, word_no)
            words = page.get_text("words")
            words_by_block: Dict[int, List[TextSpan]] = {}
            for w in words:
                bx = int(w[5])
                span = TextSpan(
                    text=str(w[4]),
                    page_number=page_num,
                    bbox=(round(float(w[0]), 2), round(float(w[1]), 2), round(float(w[2]), 2), round(float(w[3]), 2))
                )
                words_by_block.setdefault(bx, []).append(span)

            for b in text_blocks:
                if b[6] != 0:  # Skip image blocks
                    continue
                block_text = b[4].strip()
                if not block_text:
                    continue

                bbox = (
                    round(float(b[0]), 2),
                    round(float(b[1]), 2),
                    round(float(b[2]), 2),
                    round(float(b[3]), 2),
                )
                block_no = int(b[5])
                spans = words_by_block.get(block_no, [])

                block = ExtractedBlock(
                    page_number=page_num,
                    block_index=block_no,
                    text=block_text,
                    bbox=bbox,
                    spans=spans
                )
                blocks_list.append(block)
                total_chars += len(block_text)

        doc.close()

        return ExtractedDocument(
            filename=path.name,
            num_pages=num_pages,
            blocks=blocks_list,
            page_dimensions=page_dimensions,
            total_characters=total_chars
        )

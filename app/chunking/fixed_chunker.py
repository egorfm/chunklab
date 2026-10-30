import re
from typing import List, Dict, Any, Tuple
from app.ingestion.pdf_loader import ExtractedDocument, ExtractedBlock, TextSpan
from app.chunking.base import BaseChunker, Chunk, compute_union_bbox, map_text_to_spans


class FixedChunker(BaseChunker):
    """Fixed-size chunker with character window and overlap, tracking PDF bounding boxes."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        **kwargs: Any
    ):
        super().__init__(name="fixed", params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, **kwargs})
        self.chunk_size = max(50, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 10))

    def chunk_document(self, document: ExtractedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        global_chunk_idx = 0

        # Group blocks by page
        pages: Dict[int, List[ExtractedBlock]] = {}
        for block in document.blocks:
            pages.setdefault(block.page_number, []).append(block)

        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            all_page_spans: List[TextSpan] = []
            for b in page_blocks:
                all_page_spans.extend(b.spans)

            # Build full page text while tracking block origins
            page_text_pieces = []
            block_spans_map = []
            for b in page_blocks:
                page_text_pieces.append(b.text)
                block_spans_map.append(b)

            full_page_text = "\n\n".join(page_text_pieces)
            if not full_page_text.strip():
                continue

            step = self.chunk_size - self.chunk_overlap
            pos = 0
            text_len = len(full_page_text)

            while pos < text_len:
                end_pos = min(pos + self.chunk_size, text_len)
                chunk_text = full_page_text[pos:end_pos].strip()

                if chunk_text:
                    # Find coordinates by checking which spans overlap with this chunk
                    chunk_bbox = map_text_to_spans(
                        text=chunk_text,
                        spans=all_page_spans,
                        fallback_bbox=compute_union_bbox([b.bbox for b in page_blocks])
                    )

                    chunks.append(
                        Chunk(
                            chunk_index=global_chunk_idx,
                            text=chunk_text,
                            page_number=page_num,
                            bbox=chunk_bbox,
                            char_length=len(chunk_text),
                            token_count=max(1, len(chunk_text.split())),
                            strategy="fixed",
                            strategy_params=self.params
                        )
                    )
                    global_chunk_idx += 1

                if end_pos >= text_len:
                    break
                pos += step

        return chunks

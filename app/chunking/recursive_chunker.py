from typing import List, Dict, Any, Tuple
from app.ingestion.pdf_loader import ExtractedDocument, ExtractedBlock, TextSpan
from app.chunking.base import BaseChunker, Chunk, compute_union_bbox, map_text_to_spans

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Graceful fallback recursive splitter if package is still loading
    RecursiveCharacterTextSplitter = None


class FallbackRecursiveSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, separators: List[str] | None = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        next_seps = separators[1:] if len(separators) > 1 else []

        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)

        chunks = []
        current = ""

        for part in splits:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > self.chunk_size and next_seps:
                    sub_chunks = self._split(part, next_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current and current.strip():
            chunks.append(current.strip())

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive character chunker that splits text using a hierarchy of separators and tracks bounding boxes."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] | None = None,
        **kwargs: Any
    ):
        super().__init__(
            name="recursive",
            params={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "separators": separators or ["\n\n", "\n", ". ", " ", ""],
                **kwargs
            }
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        if RecursiveCharacterTextSplitter is not None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators
            )
        else:
            self._splitter = FallbackRecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators
            )

    def chunk_document(self, document: ExtractedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        global_chunk_idx = 0

        pages: Dict[int, List[ExtractedBlock]] = {}
        for block in document.blocks:
            pages.setdefault(block.page_number, []).append(block)

        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            all_page_spans: List[TextSpan] = []
            for b in page_blocks:
                all_page_spans.extend(b.spans)

            full_page_text = "\n\n".join(b.text for b in page_blocks)
            if not full_page_text.strip():
                continue

            raw_chunks = self._splitter.split_text(full_page_text)

            for chunk_text in raw_chunks:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

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
                        strategy="recursive",
                        strategy_params=self.params
                    )
                )
                global_chunk_idx += 1

        return chunks

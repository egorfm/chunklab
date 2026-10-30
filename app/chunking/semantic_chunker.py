import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from app.ingestion.pdf_loader import ExtractedDocument, ExtractedBlock, TextSpan
from app.chunking.base import BaseChunker, Chunk, compute_union_bbox, map_text_to_spans


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries."""
    # Split on periods, exclamation marks, question marks, or double newlines followed by space or uppercase
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class SemanticChunker(BaseChunker):
    """Chunks text semantically by splitting into sentences and clustering consecutive

    sentences whose semantic embedding similarity exceeds a threshold.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        min_chunk_size: int = 150,
        max_chunk_size: int = 1200,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        **kwargs: Any
    ):
        super().__init__(
            name="semantic",
            params={
                "similarity_threshold": similarity_threshold,
                "min_chunk_size": min_chunk_size,
                "max_chunk_size": max_chunk_size,
                **kwargs
            }
        )
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.embedding_fn = embedding_fn

    def set_embedding_fn(self, embedding_fn: Callable[[List[str]], List[List[float]]]) -> None:
        self.embedding_fn = embedding_fn

    def _fallback_similarity(self, s1: str, s2: str) -> float:
        """Jaccard / character ngram similarity fallback when no embedding model is loaded."""
        w1 = set(re.findall(r'\w+', s1.lower()))
        w2 = set(re.findall(r'\w+', s2.lower()))
        if not w1 or not w2:
            return 0.5
        intersection = len(w1.intersection(w2))
        union = len(w1.union(w2))
        return intersection / union if union > 0 else 0.0

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # Compute embeddings for sentences if embedding_fn is available
        embeddings = None
        if self.embedding_fn is not None:
            try:
                embeddings = np.array(self.embedding_fn(sentences))
            except Exception:
                embeddings = None

        chunks: List[str] = []
        current_sentences: List[str] = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            curr_sent = sentences[i]
            prev_sent = sentences[i - 1]

            if embeddings is not None and len(embeddings) > i:
                sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            else:
                sim = self._fallback_similarity(prev_sent, curr_sent)

            # Check boundary conditions:
            # 1. If adding this sentence exceeds max_chunk_size -> split
            # 2. If similarity < threshold AND current chunk >= min_chunk_size -> split
            should_split = False
            if current_length + len(curr_sent) > self.max_chunk_size:
                should_split = True
            elif sim < self.similarity_threshold and current_length >= self.min_chunk_size:
                should_split = True

            if should_split:
                chunks.append(" ".join(current_sentences))
                current_sentences = [curr_sent]
                current_length = len(curr_sent)
            else:
                current_sentences.append(curr_sent)
                current_length += len(curr_sent) + 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

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

            sentences = split_into_sentences(full_page_text)
            grouped_texts = self._group_sentences(sentences)

            for chunk_text in grouped_texts:
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
                        strategy="semantic",
                        strategy_params=self.params
                    )
                )
                global_chunk_idx += 1

        return chunks

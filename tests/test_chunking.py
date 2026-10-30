from app.ingestion.pdf_loader import PDFLoader
from app.chunking.fixed_chunker import FixedChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.semantic_chunker import SemanticChunker
from app.chunking.factory import ChunkerFactory


def test_fixed_chunker(sample_pdf_path):
    doc = PDFLoader.load_pdf(sample_pdf_path)
    chunker = FixedChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.page_number in [1, 2]
        assert len(c.text) > 0
        assert c.char_length == len(c.text)
        assert c.token_count > 0
        assert c.strategy == "fixed"
        # Bbox check
        x0, y0, x1, y1 = c.bbox
        assert x1 >= x0
        assert y1 >= y0


def test_recursive_chunker(sample_pdf_path):
    doc = PDFLoader.load_pdf(sample_pdf_path)
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.page_number in [1, 2]
        assert len(c.text) > 0
        assert c.strategy == "recursive"
        x0, y0, x1, y1 = c.bbox
        assert x1 >= x0
        assert y1 >= y0


def test_semantic_chunker(sample_pdf_path):
    doc = PDFLoader.load_pdf(sample_pdf_path)
    chunker = SemanticChunker(similarity_threshold=0.6, min_chunk_size=50, max_chunk_size=300)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.page_number in [1, 2]
        assert len(c.text) > 0
        assert c.strategy == "semantic"
        x0, y0, x1, y1 = c.bbox
        assert x1 >= x0
        assert y1 >= y0


def test_chunker_factory():
    c_fixed = ChunkerFactory.create("fixed", {"chunk_size": 200, "chunk_overlap": 50})
    assert isinstance(c_fixed, FixedChunker)
    assert c_fixed.chunk_size == 200

    c_rec = ChunkerFactory.create("recursive", {"chunk_size": 250, "chunk_overlap": 40})
    assert isinstance(c_rec, RecursiveChunker)
    assert c_rec.chunk_size == 250

    c_sem = ChunkerFactory.create("semantic", {"similarity_threshold": 0.75})
    assert isinstance(c_sem, SemanticChunker)
    assert c_sem.similarity_threshold == 0.75

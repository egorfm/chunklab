import pytest
import pymupdf as fitz
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database.database import Base, get_db
from app.config import settings
from app.api.main import app

# Test database
TEST_DB_PATH = settings.DATA_DIR / "test_chunklab.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH.resolve()}"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    settings.ensure_directories()
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture
def db_session():
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_path(tmp_path) -> Path:
    """Generates a real 2-page test PDF with distinct text blocks and coordinates."""
    pdf_path = tmp_path / "sample_benchmark_doc.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=600, height=800)
    page1.insert_text((50, 80), "ChunkLab Benchmark Engine Introduction", fontsize=18)
    page1.insert_text(
        (50, 130),
        "Chunking strategies are critical for Retrieval-Augmented Generation (RAG) systems. "
        "Different chunking techniques impact both recall and precision of retrieved passages.",
        fontsize=12
    )
    page1.insert_text(
        (50, 220),
        "Fixed-size chunking divides text into equal windows with arbitrary overlap boundaries. "
        "Recursive chunking respects natural paragraph and sentence delimiters.",
        fontsize=12
    )

    # Page 2
    page2 = doc.new_page(width=600, height=800)
    page2.insert_text((50, 80), "Semantic Chunking and Evaluation Metrics", fontsize=18)
    page2.insert_text(
        (50, 130),
        "Semantic chunking determines split boundaries based on semantic embedding distance between sentences. "
        "This ensures topical coherence within individual chunks.",
        fontsize=12
    )
    page2.insert_text(
        (50, 240),
        "Evaluation metrics include Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG. "
        "Relevance is grounded in exact bounding box coordinates.",
        fontsize=12
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

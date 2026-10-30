import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pymupdf as fitz
from app.config import settings
from app.database import init_db, SessionLocal, crud
from app.ingestion.pdf_loader import PDFLoader
from app.evaluation.evaluator import BenchmarkEngine
from app.embeddings.service import EmbeddingService
from app.retrieval.qdrant_retriever import QdrantRetriever


def generate_sample_pdf(dest_path: Path):
    doc = fitz.open()

    # --- Page 1: RAG Architecture & Vector Search ---
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((54, 72), "Modern RAG Architectures: An Empirical Analysis", fontsize=18)
    p1.insert_text((54, 110), "1. Introduction to Retrieval-Augmented Generation", fontsize=14)
    p1.insert_text(
        (54, 140),
        "Retrieval-Augmented Generation (RAG) grounds language model generations on verifiable external knowledge. "
        "The accuracy of a RAG pipeline is bounded by the quality and relevance of the retrieved document chunks. "
        "Selecting optimal chunking strategies minimizes context fragmentation and hallucination rates.",
        fontsize=11
    )
    p1.insert_text((54, 230), "2. Vector Databases and Distance Metrics", fontsize=14)
    p1.insert_text(
        (54, 260),
        "Qdrant serves as the high-throughput vector search engine. It indexes high-dimensional dense embeddings "
        "and performs approximate nearest neighbor (ANN) search via Hierarchical Navigable Small World (HNSW) graphs. "
        "Cosine similarity measures the orientation between query vectors and document chunk vectors.",
        fontsize=11
    )

    # --- Page 2: Chunking Strategies ---
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((54, 72), "3. Comparative Chunking Methodologies", fontsize=18)
    p2.insert_text((54, 110), "3.1 Fixed-Size Window Chunking", fontsize=14)
    p2.insert_text(
        (54, 140),
        "Fixed-size chunking partitions raw text into fixed character or token intervals with sliding overlaps. "
        "While computationally trivial and predictable in size, it frequently bisects continuous sentences and phrases, "
        "resulting in broken syntactic dependencies and impaired embedding representations.",
        fontsize=11
    )
    p2.insert_text((54, 230), "3.2 Recursive Hierarchical Chunking", fontsize=14)
    p2.insert_text(
        (54, 260),
        "Recursive chunking employs a priority list of delimiters: double newlines, single newlines, and punctuation. "
        "It preserves structural document cohesion by keeping paragraphs together where possible and recursing "
        "to finer delimiters only when a segment exceeds the target token capacity.",
        fontsize=11
    )
    p2.insert_text((54, 350), "3.3 Semantic Distance Chunking", fontsize=14)
    p2.insert_text(
        (54, 380),
        "Semantic chunking analyzes sequential sentence similarity using cosine distance in embedding space. "
        "When the semantic divergence between adjacent sentences exceeds an adaptive threshold, a new chunk boundary "
        "is formed. This guarantees that each chunk encapsulates a cohesive semantic topic.",
        fontsize=11
    )

    # --- Page 3: Evaluation Metrics ---
    p3 = doc.new_page(width=612, height=792)
    p3.insert_text((54, 72), "4. Coordinate-Grounded Evaluation Engine", fontsize=18)
    p3.insert_text((54, 110), "4.1 Spatial IoU Relevance Verification", fontsize=14)
    p3.insert_text(
        (54, 140),
        "Traditional text-overlap matching can be susceptible to repetitive vocabulary. ChunkLab determines true relevance "
        "by computing the 2D bounding box Intersection over Union (IoU) on the source PDF page coordinate plane. "
        "A retrieved passage is marked relevant if its bounding box intersects the ground-truth target region.",
        fontsize=11
    )
    p3.insert_text((54, 240), "4.2 Standard Retrieval Metrics", fontsize=14)
    p3.insert_text(
        (54, 270),
        "Retrieval effectiveness is quantified using Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG. "
        "Furthermore, retrieval latency in milliseconds and total chunk counts are monitored to reflect system efficiency.",
        fontsize=11
    )

    doc.save(str(dest_path))
    doc.close()


def seed():
    print("🌱 Seeding ChunkLab demo document and benchmark dataset...")
    settings.ensure_directories()
    init_db()

    pdf_dest = settings.UPLOAD_DIR / "rag_chunking_whitepaper.pdf"
    generate_sample_pdf(pdf_dest)

    db = SessionLocal()
    try:
        # Check if already seeded
        existing_doc = crud.list_documents(db)
        if existing_doc:
            print("Database already contains documents. Cleaning up old demo data...")
            for d in existing_doc:
                crud.delete_document(db, d.id)

        # 1. Ingest PDF
        extracted = PDFLoader.load_pdf(pdf_dest)
        doc = crud.create_document(
            db=db,
            filename=pdf_dest.name,
            file_path=str(pdf_dest.resolve()),
            file_size=pdf_dest.stat().st_size,
            num_pages=extracted.num_pages
        )
        print(f"✓ Ingested PDF '{doc.filename}' ({doc.num_pages} pages, {len(extracted.blocks)} blocks)")

        # 2. Create Ground Truth Benchmark Dataset
        dataset = crud.create_dataset(
            db=db,
            document_id=doc.id,
            name="RAG Chunking Benchmark Suite",
            description="Gold standard question set with spatial coordinate targets"
        )

        queries_data = [
            {
                "query_text": "How does Qdrant perform approximate nearest neighbor search?",
                "ground_truth_answer": "Via Hierarchical Navigable Small World (HNSW) graphs and cosine similarity",
                "target_page": 1,
                "target_bbox": (54.0, 230.0, 560.0, 320.0),
                "target_text": "Qdrant serves as the high-throughput vector search engine. It indexes high-dimensional dense embeddings"
            },
            {
                "query_text": "What is the primary drawback of fixed-size window chunking?",
                "ground_truth_answer": "It frequently bisects continuous sentences, breaking syntactic dependencies",
                "target_page": 2,
                "target_bbox": (54.0, 110.0, 560.0, 200.0),
                "target_text": "Fixed-size chunking partitions raw text into fixed character or token intervals"
            },
            {
                "query_text": "How does semantic chunking decide where to place split boundaries?",
                "ground_truth_answer": "By analyzing sequential sentence similarity in embedding space with an adaptive threshold",
                "target_page": 2,
                "target_bbox": (54.0, 350.0, 560.0, 440.0),
                "target_text": "Semantic chunking analyzes sequential sentence similarity using cosine distance"
            },
            {
                "query_text": "How does ChunkLab verify passage relevance on PDF documents?",
                "ground_truth_answer": "By computing 2D bounding box Intersection over Union (IoU) on source PDF coordinates",
                "target_page": 3,
                "target_bbox": (54.0, 110.0, 560.0, 220.0),
                "target_text": "ChunkLab determines true relevance by computing the 2D bounding box Intersection over Union"
            }
        ]

        for q in queries_data:
            crud.add_query_to_dataset(
                db=db,
                dataset_id=dataset.id,
                query_text=q["query_text"],
                target_page=q["target_page"],
                target_bbox=q["target_bbox"],
                ground_truth_answer=q["ground_truth_answer"],
                target_text=q["target_text"]
            )
        print(f"✓ Created evaluation dataset '{dataset.name}' with {len(queries_data)} queries")

        # 3. Run Benchmark Experiment
        print("⚡ Running initial benchmark comparing Fixed, Recursive, and Semantic chunking...")
        experiment = crud.create_experiment(
            db=db,
            name="Baseline Chunking Benchmark",
            document_id=doc.id,
            dataset_id=dataset.id,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=5,
            iou_threshold=0.15
        )

        emb_service = EmbeddingService(model_name=experiment.embedding_model)
        retriever = QdrantRetriever()
        engine = BenchmarkEngine(retriever=retriever, embedding_service=emb_service)

        strategies = [
            ("fixed", {"chunk_size": 400, "chunk_overlap": 80}),
            ("recursive", {"chunk_size": 400, "chunk_overlap": 80}),
            ("semantic", {"similarity_threshold": 0.65, "min_chunk_size": 100, "max_chunk_size": 800})
        ]

        for strat_name, strat_params in strategies:
            run = engine.run_strategy_benchmark(
                db=db,
                document_id=doc.id,
                extracted_doc=extracted,
                dataset=dataset,
                strategy_name=strat_name,
                strategy_params=strat_params,
                top_k=5,
                iou_threshold=0.15,
                experiment_id=experiment.id
            )
            print(f"  • Strategy '{strat_name.upper()}': Recall@5={run.recall_at_k:.2f}, MRR={run.mrr:.2f}, Latency={run.avg_latency_ms:.1f}ms, Chunks={run.total_chunks}")

        print("✨ Demo seeding completed successfully! Open Streamlit Dashboard to explore.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()

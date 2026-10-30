from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db, crud, models
from app.ingestion.pdf_loader import PDFLoader
from app.embeddings.service import EmbeddingService
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.evaluation.evaluator import BenchmarkEngine
from app.api import schemas

router = APIRouter(prefix="/api/experiments", tags=["Experiments & Benchmark"])


@router.post("/run", response_model=schemas.ExperimentResponse, status_code=status.HTTP_201_CREATED)
def run_experiment(req: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    doc = crud.get_document(db, req.document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    dataset = crud.get_dataset(db, req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation Dataset not found")

    if not dataset.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation dataset has no queries. Add at least one query before running benchmark."
        )

    # 1. Parse PDF document
    try:
        extracted_doc = PDFLoader.load_pdf(doc.file_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to read PDF: {e}")

    # 2. Create Experiment record
    experiment = crud.create_experiment(
        db=db,
        name=req.name,
        document_id=req.document_id,
        dataset_id=req.dataset_id,
        embedding_model=req.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
        top_k=req.top_k,
        iou_threshold=req.iou_threshold
    )

    # 3. Setup embedding service and retriever for this experiment
    embedding_service = EmbeddingService(model_name=experiment.embedding_model)
    retriever = QdrantRetriever()
    engine = BenchmarkEngine(retriever=retriever, embedding_service=embedding_service)

    # 4. Run benchmark for each strategy requested
    for strat_cfg in req.strategies:
        try:
            engine.run_strategy_benchmark(
                db=db,
                document_id=doc.id,
                extracted_doc=extracted_doc,
                dataset=dataset,
                strategy_name=strat_cfg.strategy,
                strategy_params=strat_cfg.params,
                top_k=req.top_k,
                iou_threshold=req.iou_threshold,
                experiment_id=experiment.id
            )
        except Exception as e:
            # If a strategy fails, log error and continue with others
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error evaluating strategy '{strat_cfg.strategy}': {e}"
            )

    db.refresh(experiment)
    return experiment


@router.get("", response_model=List[schemas.ExperimentResponse])
def get_experiments(document_id: Optional[int] = None, db: Session = Depends(get_db)):
    experiments = crud.list_experiments(db, document_id)
    return experiments


@router.get("/{experiment_id}", response_model=schemas.ExperimentResponse)
def get_experiment_detail(experiment_id: int, db: Session = Depends(get_db)):
    exp = crud.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return exp


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    success = crud.delete_experiment(db, experiment_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return None


@router.post("/search")
def ad_hoc_search(req: schemas.SearchRequest, db: Session = Depends(get_db)):
    doc = crud.get_document(db, req.document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    embedding_service = EmbeddingService(model_name=req.embedding_model)
    query_vec = embedding_service.embed_query(req.query)

    retriever = QdrantRetriever()
    results = retriever.search(
        document_id=req.document_id,
        strategy=req.strategy,
        query_vector=query_vec,
        top_k=req.top_k
    )

    return {
        "document_id": req.document_id,
        "strategy": req.strategy,
        "query": req.query,
        "results": results
    }

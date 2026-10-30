from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.database import models


# --- Documents ---
def create_document(db: Session, filename: str, file_path: str, file_size: int, num_pages: int) -> models.Document:
    doc = models.Document(
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        num_pages=num_pages
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, document_id: int) -> Optional[models.Document]:
    return db.query(models.Document).filter(models.Document.id == document_id).first()


def list_documents(db: Session) -> List[models.Document]:
    return db.query(models.Document).order_by(models.Document.created_at.desc()).all()


def delete_document(db: Session, document_id: int) -> bool:
    doc = get_document(db, document_id)
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True


# --- Datasets ---
def create_dataset(db: Session, document_id: int, name: str, description: str = "") -> models.EvaluationDataset:
    dataset = models.EvaluationDataset(
        document_id=document_id,
        name=name,
        description=description
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def add_query_to_dataset(
    db: Session,
    dataset_id: int,
    query_text: str,
    target_page: int,
    target_bbox: tuple[float, float, float, float],
    ground_truth_answer: str = "",
    target_text: str = ""
) -> models.EvaluationQuery:
    query = models.EvaluationQuery(
        dataset_id=dataset_id,
        query_text=query_text,
        ground_truth_answer=ground_truth_answer,
        target_page=target_page,
        target_bbox_x0=target_bbox[0],
        target_bbox_y0=target_bbox[1],
        target_bbox_x1=target_bbox[2],
        target_bbox_y1=target_bbox[3],
        target_text=target_text
    )
    db.add(query)
    db.commit()
    db.refresh(query)
    return query


def get_dataset(db: Session, dataset_id: int) -> Optional[models.EvaluationDataset]:
    return db.query(models.EvaluationDataset).filter(models.EvaluationDataset.id == dataset_id).first()


def list_datasets(db: Session, document_id: Optional[int] = None) -> List[models.EvaluationDataset]:
    q = db.query(models.EvaluationDataset)
    if document_id is not None:
        q = q.filter(models.EvaluationDataset.document_id == document_id)
    return q.order_by(models.EvaluationDataset.created_at.desc()).all()


def delete_dataset(db: Session, dataset_id: int) -> bool:
    dataset = get_dataset(db, dataset_id)
    if not dataset:
        return False
    db.delete(dataset)
    db.commit()
    return True


# --- Experiments ---
def create_experiment(
    db: Session,
    name: str,
    document_id: int,
    dataset_id: int,
    embedding_model: str,
    top_k: int = 5,
    iou_threshold: float = 0.20
) -> models.Experiment:
    experiment = models.Experiment(
        name=name,
        document_id=document_id,
        dataset_id=dataset_id,
        embedding_model=embedding_model,
        top_k=top_k,
        iou_threshold=iou_threshold
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def get_experiment(db: Session, experiment_id: int) -> Optional[models.Experiment]:
    return db.query(models.Experiment).filter(models.Experiment.id == experiment_id).first()


def list_experiments(db: Session, document_id: Optional[int] = None) -> List[models.Experiment]:
    q = db.query(models.Experiment)
    if document_id is not None:
        q = q.filter(models.Experiment.document_id == document_id)
    return q.order_by(models.Experiment.created_at.desc()).all()


def delete_experiment(db: Session, experiment_id: int) -> bool:
    exp = get_experiment(db, experiment_id)
    if not exp:
        return False
    db.delete(exp)
    db.commit()
    return True


# --- Chunks ---
def get_chunks_by_document_and_strategy(
    db: Session,
    document_id: int,
    strategy: str
) -> List[models.ChunkRecord]:
    return db.query(models.ChunkRecord).filter(
        models.ChunkRecord.document_id == document_id,
        models.ChunkRecord.strategy == strategy
    ).order_by(models.ChunkRecord.chunk_index.asc()).all()

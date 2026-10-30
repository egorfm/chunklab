from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db, crud, models
from app.api import schemas

router = APIRouter(prefix="/api/datasets", tags=["Evaluation Datasets"])


@router.post("", response_model=schemas.EvaluationDatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(req: schemas.EvaluationDatasetCreate, db: Session = Depends(get_db)):
    doc = crud.get_document(db, req.document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    dataset = crud.create_dataset(
        db=db,
        document_id=req.document_id,
        name=req.name,
        description=req.description or ""
    )

    for q in req.queries:
        crud.add_query_to_dataset(
            db=db,
            dataset_id=dataset.id,
            query_text=q.query_text,
            target_page=q.target_page,
            target_bbox=q.target_bbox,
            ground_truth_answer=q.ground_truth_answer or "",
            target_text=q.target_text or ""
        )

    db.refresh(dataset)
    return schemas.EvaluationDatasetResponse(
        id=dataset.id,
        document_id=dataset.document_id,
        name=dataset.name,
        description=dataset.description,
        created_at=dataset.created_at,
        queries_count=len(dataset.queries)
    )


@router.get("", response_model=List[schemas.EvaluationDatasetResponse])
def get_datasets(document_id: Optional[int] = None, db: Session = Depends(get_db)):
    datasets = crud.list_datasets(db, document_id)
    return [
        schemas.EvaluationDatasetResponse(
            id=d.id,
            document_id=d.document_id,
            name=d.name,
            description=d.description,
            created_at=d.created_at,
            queries_count=len(d.queries)
        )
        for d in datasets
    ]


@router.get("/{dataset_id}", response_model=schemas.EvaluationDatasetDetailResponse)
def get_dataset_detail(dataset_id: int, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return schemas.EvaluationDatasetDetailResponse(
        id=dataset.id,
        document_id=dataset.document_id,
        name=dataset.name,
        description=dataset.description,
        created_at=dataset.created_at,
        queries_count=len(dataset.queries),
        queries=[
            schemas.EvaluationQueryResponse(
                id=q.id,
                dataset_id=q.dataset_id,
                query_text=q.query_text,
                ground_truth_answer=q.ground_truth_answer,
                target_page=q.target_page,
                target_bbox_x0=q.target_bbox_x0,
                target_bbox_y0=q.target_bbox_y0,
                target_bbox_x1=q.target_bbox_x1,
                target_bbox_y1=q.target_bbox_y1,
                target_text=q.target_text
            )
            for q in dataset.queries
        ]
    )


@router.post("/{dataset_id}/queries", response_model=schemas.EvaluationQueryResponse, status_code=status.HTTP_201_CREATED)
def add_query(dataset_id: int, req: schemas.EvaluationQueryCreate, db: Session = Depends(get_db)):
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    query = crud.add_query_to_dataset(
        db=db,
        dataset_id=dataset_id,
        query_text=req.query_text,
        target_page=req.target_page,
        target_bbox=req.target_bbox,
        ground_truth_answer=req.ground_truth_answer or "",
        target_text=req.target_text or ""
    )
    return schemas.EvaluationQueryResponse.model_validate(query)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    success = crud.delete_dataset(db, dataset_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return None

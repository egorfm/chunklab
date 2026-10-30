import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db, crud, models
from app.ingestion.pdf_loader import PDFLoader
from app.chunking.factory import ChunkerFactory
from app.embeddings.service import EmbeddingService
from app.api import schemas

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=schemas.DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    settings.ensure_directories()
    dest_path = settings.UPLOAD_DIR / file.filename

    # Avoid overwriting collisions with simple counter if needed
    counter = 1
    stem = Path(file.filename).stem
    while dest_path.exists():
        dest_path = settings.UPLOAD_DIR / f"{stem}_{counter}.pdf"
        counter += 1

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    try:
        extracted = PDFLoader.load_pdf(dest_path)
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse PDF document: {e}"
        )

    doc = crud.create_document(
        db=db,
        filename=dest_path.name,
        file_path=str(dest_path.resolve()),
        file_size=dest_path.stat().st_size,
        num_pages=extracted.num_pages
    )
    return doc


@router.get("", response_model=List[schemas.DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    return crud.list_documents(db)


@router.get("/{document_id}", response_model=schemas.DocumentDetailResponse)
def get_document_detail(document_id: int, db: Session = Depends(get_db)):
    doc = crud.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    blocks_sample = []
    if Path(doc.file_path).exists():
        try:
            extracted = PDFLoader.load_pdf(doc.file_path)
            for b in extracted.blocks[:30]:
                blocks_sample.append(
                    schemas.ExtractedBlockResponse(
                        page_number=b.page_number,
                        block_index=b.block_index,
                        text=b.text,
                        bbox=b.bbox
                    )
                )
        except Exception:
            pass

    return schemas.DocumentDetailResponse(
        id=doc.id,
        filename=doc.filename,
        file_path=doc.file_path,
        file_size=doc.file_size,
        num_pages=doc.num_pages,
        created_at=doc.created_at,
        blocks_sample=blocks_sample
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = crud.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    file_path = Path(doc.file_path)
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    crud.delete_document(db, document_id)
    return None


@router.post("/{document_id}/preview-chunks", response_model=List[schemas.ChunkResponse])
def preview_chunks(
    document_id: int,
    req: schemas.ChunkPreviewRequest,
    db: Session = Depends(get_db)
):
    doc = crud.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    extracted = PDFLoader.load_pdf(doc.file_path)
    embedding_service = EmbeddingService() if req.strategy == "semantic" else None
    chunker = ChunkerFactory.create(
        strategy=req.strategy,
        params=req.params,
        embedding_fn=embedding_service.embed_texts if embedding_service else None
    )
    chunks = chunker.chunk_document(extracted)

    return [
        schemas.ChunkResponse(
            chunk_index=c.chunk_index,
            text=c.text,
            page_number=c.page_number,
            bbox=c.bbox,
            char_length=c.char_length,
            token_count=c.token_count,
            strategy=c.strategy
        )
        for c in chunks
    ]


@router.get("/{document_id}/chunks/{strategy}", response_model=List[schemas.ChunkResponse])
def get_stored_chunks(
    document_id: int,
    strategy: str,
    db: Session = Depends(get_db)
):
    chunks = crud.get_chunks_by_document_and_strategy(db, document_id, strategy)
    return [
        schemas.ChunkResponse(
            chunk_index=c.chunk_index,
            text=c.text,
            page_number=c.page_number,
            bbox=(c.bbox_x0, c.bbox_y0, c.bbox_x1, c.bbox_y1),
            char_length=c.char_length,
            token_count=max(1, len(c.text.split())),
            strategy=c.strategy
        )
        for c in chunks
    ]

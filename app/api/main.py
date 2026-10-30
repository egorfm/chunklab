from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.config import settings
from app.database import init_db, get_db, crud
from app.api.routes_documents import router as documents_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_experiments import router as experiments_router
from app.utils.export import export_experiment_to_json, export_experiment_to_csv


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure directories and database tables
    settings.ensure_directories()
    init_db()
    yield
    # Shutdown clean up if needed


app = FastAPI(
    title=settings.APP_NAME,
    description="ChunkLab — PDF Chunking & Retrieval Benchmark API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(datasets_router)
app.include_router(experiments_router)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV
    }


@app.get("/api/experiments/{experiment_id}/export/json", tags=["Export"])
def export_json(experiment_id: int, db: Session = Depends(get_db)):
    exp = crud.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    json_data = export_experiment_to_json(exp)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.json"}
    )


@app.get("/api/experiments/{experiment_id}/export/csv", tags=["Export"])
def export_csv(experiment_id: int, db: Session = Depends(get_db)):
    exp = crud.get_experiment(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    csv_data = export_experiment_to_csv(exp)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.csv"}
    )

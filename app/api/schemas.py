from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


# --- Document Schemas ---
class DocumentBase(BaseModel):
    filename: str
    file_size: int
    num_pages: int


class DocumentResponse(DocumentBase):
    id: int
    file_path: str
    created_at: datetime

    class Config:
        from_attributes = True


class ExtractedBlockResponse(BaseModel):
    page_number: int
    block_index: int
    text: str
    bbox: Tuple[float, float, float, float]


class DocumentDetailResponse(DocumentResponse):
    blocks_sample: List[ExtractedBlockResponse] = []


# --- Evaluation Dataset & Query Schemas ---
class EvaluationQueryCreate(BaseModel):
    query_text: str = Field(..., min_length=2)
    ground_truth_answer: Optional[str] = ""
    target_page: int = Field(..., ge=1)
    target_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    target_text: Optional[str] = ""


class EvaluationQueryResponse(BaseModel):
    id: int
    dataset_id: int
    query_text: str
    ground_truth_answer: str
    target_page: int
    target_bbox_x0: float
    target_bbox_y0: float
    target_bbox_x1: float
    target_bbox_y1: float
    target_text: str

    class Config:
        from_attributes = True


class EvaluationDatasetCreate(BaseModel):
    document_id: int
    name: str = Field(..., min_length=1)
    description: Optional[str] = ""
    queries: List[EvaluationQueryCreate] = []


class EvaluationDatasetResponse(BaseModel):
    id: int
    document_id: int
    name: str
    description: str
    created_at: datetime
    queries_count: int = 0

    class Config:
        from_attributes = True


class EvaluationDatasetDetailResponse(EvaluationDatasetResponse):
    queries: List[EvaluationQueryResponse] = []


# --- Chunking & Experiment Schemas ---
class StrategyConfig(BaseModel):
    strategy: str  # fixed, recursive, semantic
    params: Dict[str, Any] = {}


class ChunkPreviewRequest(BaseModel):
    document_id: int
    strategy: str
    params: Dict[str, Any] = {}


class ChunkResponse(BaseModel):
    chunk_index: int
    text: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    char_length: int
    token_count: int
    strategy: str


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1)
    document_id: int
    dataset_id: int
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    iou_threshold: float = 0.20
    strategies: List[StrategyConfig] = [
        StrategyConfig(strategy="fixed", params={"chunk_size": 500, "chunk_overlap": 100}),
        StrategyConfig(strategy="recursive", params={"chunk_size": 500, "chunk_overlap": 100}),
        StrategyConfig(strategy="semantic", params={"similarity_threshold": 0.70, "min_chunk_size": 150, "max_chunk_size": 1000}),
    ]


class RetrievedChunkDetail(BaseModel):
    rank: int
    point_id: str
    score: float
    chunk_index: int
    page_number: int
    bbox: Tuple[float, float, float, float]
    text: str
    char_length: int
    is_relevant: bool = False
    match_score: float = 0.0


class RetrievalLogResponse(BaseModel):
    query_id: int
    query_text: str
    latency_ms: float
    retrieved_chunks: List[RetrievedChunkDetail] = []


class ExperimentRunResponse(BaseModel):
    id: int
    strategy: str
    strategy_params: Dict[str, Any]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    avg_latency_ms: float
    avg_chunk_size: float
    total_chunks: int
    created_at: datetime
    logs: Optional[List[RetrievalLogResponse]] = None

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    id: int
    name: str
    document_id: int
    dataset_id: int
    embedding_model: str
    top_k: int
    iou_threshold: float
    created_at: datetime
    runs: List[ExperimentRunResponse] = []

    class Config:
        from_attributes = True


# --- Single Query Search Schema ---
class SearchRequest(BaseModel):
    document_id: int
    strategy: str
    query: str
    top_k: int = 5
    embedding_model: Optional[str] = None

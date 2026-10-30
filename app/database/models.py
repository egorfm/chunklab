import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    JSON
)
from sqlalchemy.orm import relationship
from app.database.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, default=0)
    num_pages = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chunks = relationship("ChunkRecord", back_populates="document", cascade="all, delete-orphan")
    datasets = relationship("EvaluationDataset", back_populates="document", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="document", cascade="all, delete-orphan")


class ChunkRecord(Base):
    __tablename__ = "chunk_records"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    strategy = Column(String(50), nullable=False, index=True)  # fixed, recursive, semantic
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)  # 1-based page index
    bbox_x0 = Column(Float, nullable=False, default=0.0)
    bbox_y0 = Column(Float, nullable=False, default=0.0)
    bbox_x1 = Column(Float, nullable=False, default=0.0)
    bbox_y1 = Column(Float, nullable=False, default=0.0)
    char_length = Column(Integer, default=0)
    qdrant_point_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="datasets")
    queries = relationship("EvaluationQuery", back_populates="dataset", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="dataset", cascade="all, delete-orphan")


class EvaluationQuery(Base):
    __tablename__ = "evaluation_queries"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False)
    query_text = Column(Text, nullable=False)
    ground_truth_answer = Column(Text, default="")
    target_page = Column(Integer, nullable=False)  # 1-based target page
    target_bbox_x0 = Column(Float, nullable=False, default=0.0)
    target_bbox_y0 = Column(Float, nullable=False, default=0.0)
    target_bbox_x1 = Column(Float, nullable=False, default=0.0)
    target_bbox_y1 = Column(Float, nullable=False, default=0.0)
    target_text = Column(Text, default="")

    dataset = relationship("EvaluationDataset", back_populates="queries")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    top_k = Column(Integer, default=5)
    iou_threshold = Column(Float, default=0.2)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="experiments")
    dataset = relationship("EvaluationDataset", back_populates="experiments")
    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    strategy = Column(String(50), nullable=False)  # fixed, recursive, semantic
    strategy_params = Column(JSON, nullable=False, default=dict)
    
    # Evaluation Metrics
    recall_at_k = Column(Float, default=0.0)
    precision_at_k = Column(Float, default=0.0)
    mrr = Column(Float, default=0.0)
    ndcg_at_k = Column(Float, default=0.0)
    avg_latency_ms = Column(Float, default=0.0)
    avg_chunk_size = Column(Float, default=0.0)
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    experiment = relationship("Experiment", back_populates="runs")
    logs = relationship("RetrievalLog", back_populates="run", cascade="all, delete-orphan")


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_run_id = Column(Integer, ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False)
    query_id = Column(Integer, nullable=False)
    query_text = Column(Text, nullable=False)
    latency_ms = Column(Float, default=0.0)
    retrieved_chunks = Column(JSON, default=list)  # list of chunk dicts with scores and match status

    run = relationship("ExperimentRun", back_populates="logs")

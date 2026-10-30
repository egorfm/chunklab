import time
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.ingestion.pdf_loader import ExtractedDocument
from app.chunking.factory import ChunkerFactory
from app.chunking.base import Chunk
from app.embeddings.service import EmbeddingService
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.evaluation.coordinate_matcher import check_relevance
from app.evaluation.metrics import BenchmarkMetrics
from app.database import models

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """Orchestrates end-to-end chunking, indexing, retrieval, and metric calculation for comparison."""

    def __init__(
        self,
        retriever: Optional[QdrantRetriever] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.retriever = retriever or QdrantRetriever()
        self.embedding_service = embedding_service or EmbeddingService()

    def run_strategy_benchmark(
        self,
        db: Session,
        document_id: int,
        extracted_doc: ExtractedDocument,
        dataset: models.EvaluationDataset,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        top_k: int = 5,
        iou_threshold: float = 0.20,
        experiment_id: Optional[int] = None
    ) -> models.ExperimentRun:
        """Executes chunking, embedding, indexing, querying and metric calculation for a single strategy."""

        logger.info(f"Starting benchmark for strategy '{strategy_name}' on document {document_id}")

        # 1. Chunk document
        chunker = ChunkerFactory.create(
            strategy=strategy_name,
            params=strategy_params,
            embedding_fn=self.embedding_service.embed_texts if strategy_name == "semantic" else None
        )
        chunks: List[Chunk] = chunker.chunk_document(extracted_doc)

        if not chunks:
            raise ValueError(f"Chunker '{strategy_name}' produced 0 chunks for document {document_id}.")

        total_chunks = len(chunks)
        avg_chunk_size = sum(c.char_length for c in chunks) / total_chunks

        # Save chunks to database for record
        # (Clean up old chunks for this document and strategy first)
        db.query(models.ChunkRecord).filter(
            models.ChunkRecord.document_id == document_id,
            models.ChunkRecord.strategy == strategy_name
        ).delete()
        db.commit()

        chunk_records = []
        for c in chunks:
            chunk_records.append(
                models.ChunkRecord(
                    document_id=document_id,
                    strategy=strategy_name,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    page_number=c.page_number,
                    bbox_x0=c.bbox[0],
                    bbox_y0=c.bbox[1],
                    bbox_x1=c.bbox[2],
                    bbox_y1=c.bbox[3],
                    char_length=c.char_length,
                )
            )
        db.bulk_save_objects(chunk_records)
        db.commit()

        # 2. Embed chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # 3. Index in Qdrant
        self.retriever.index_chunks(
            document_id=document_id,
            strategy=strategy_name,
            chunks=chunks,
            embeddings=embeddings,
            recreate=True
        )

        # 4. Evaluate queries
        query_eval_results = []
        retrieval_logs = []

        for q in dataset.queries:
            target_bbox = (q.target_bbox_x0, q.target_bbox_y0, q.target_bbox_x1, q.target_bbox_y1)

            t0 = time.perf_counter()
            q_vector = self.embedding_service.embed_query(q.query_text)
            retrieved = self.retriever.search(
                document_id=document_id,
                strategy=strategy_name,
                query_vector=q_vector,
                top_k=top_k
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            binary_relevances = []
            detailed_retrieved = []

            for r in retrieved:
                r_bbox = r["bbox"]
                is_rel, match_score = check_relevance(
                    retrieved_page=r["page_number"],
                    retrieved_bbox=r_bbox,
                    retrieved_text=r["text"],
                    target_page=q.target_page,
                    target_bbox=target_bbox,
                    target_text=q.target_text,
                    iou_threshold=iou_threshold
                )
                binary_relevances.append(1 if is_rel else 0)
                detailed_retrieved.append({
                    **r,
                    "is_relevant": is_rel,
                    "match_score": match_score
                })

            query_eval_results.append({
                "query_id": q.id,
                "binary_relevances": binary_relevances,
                "total_ground_truth_relevant": 1,
                "latency_ms": latency_ms
            })

            retrieval_logs.append({
                "query_id": q.id,
                "query_text": q.query_text,
                "latency_ms": latency_ms,
                "retrieved_chunks": detailed_retrieved
            })

        # 5. Aggregate metrics
        metrics = BenchmarkMetrics.aggregate(
            query_results=query_eval_results,
            top_k=top_k,
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size
        )

        # 6. Save experiment run to DB
        exp_run = models.ExperimentRun(
            experiment_id=experiment_id if experiment_id else 0,
            strategy=strategy_name,
            strategy_params=strategy_params,
            recall_at_k=metrics["recall_at_k"],
            precision_at_k=metrics["precision_at_k"],
            mrr=metrics["mrr"],
            ndcg_at_k=metrics["ndcg_at_k"],
            avg_latency_ms=metrics["avg_latency_ms"],
            avg_chunk_size=metrics["avg_chunk_size"],
            total_chunks=metrics["total_chunks"],
        )
        db.add(exp_run)
        db.commit()
        db.refresh(exp_run)

        # Save retrieval logs
        for log_data in retrieval_logs:
            r_log = models.RetrievalLog(
                experiment_run_id=exp_run.id,
                query_id=log_data["query_id"],
                query_text=log_data["query_text"],
                latency_ms=log_data["latency_ms"],
                retrieved_chunks=log_data["retrieved_chunks"]
            )
            db.add(r_log)
        db.commit()

        logger.info(
            f"Finished benchmark for '{strategy_name}': Recall@{top_k}={metrics['recall_at_k']}, "
            f"MRR={metrics['mrr']}, nDCG@{top_k}={metrics['ndcg_at_k']}, Latency={metrics['avg_latency_ms']}ms"
        )
        return exp_run

import json
import io
import pandas as pd
from typing import Dict, Any, List
from app.database import models


def export_experiment_to_json(experiment: models.Experiment) -> str:
    """Serializes experiment, its runs, and retrieval logs to formatted JSON."""
    data = {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "document_id": experiment.document_id,
        "document_filename": experiment.document.filename if experiment.document else "",
        "dataset_id": experiment.dataset_id,
        "dataset_name": experiment.dataset.name if experiment.dataset else "",
        "embedding_model": experiment.embedding_model,
        "top_k": experiment.top_k,
        "iou_threshold": experiment.iou_threshold,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else "",
        "runs": []
    }

    for run in experiment.runs:
        run_data = {
            "strategy": run.strategy,
            "params": run.strategy_params,
            "metrics": {
                "recall_at_k": run.recall_at_k,
                "precision_at_k": run.precision_at_k,
                "mrr": run.mrr,
                "ndcg_at_k": run.ndcg_at_k,
                "avg_latency_ms": run.avg_latency_ms,
                "avg_chunk_size": run.avg_chunk_size,
                "total_chunks": run.total_chunks,
            },
            "retrieval_logs": [
                {
                    "query_id": log.query_id,
                    "query_text": log.query_text,
                    "latency_ms": log.latency_ms,
                    "retrieved_chunks": log.retrieved_chunks
                }
                for log in run.logs
            ]
        }
        data["runs"].append(run_data)

    return json.dumps(data, indent=2, ensure_ascii=False)


def export_experiment_to_csv(experiment: models.Experiment) -> str:
    """Exports high-level strategy comparison metrics to CSV format."""
    rows = []
    for run in experiment.runs:
        rows.append({
            "Experiment": experiment.name,
            "Strategy": run.strategy,
            "Recall@K": run.recall_at_k,
            "Precision@K": run.precision_at_k,
            "MRR": run.mrr,
            "nDCG@K": run.ndcg_at_k,
            "Avg Latency (ms)": run.avg_latency_ms,
            "Avg Chunk Size (chars)": run.avg_chunk_size,
            "Total Chunks": run.total_chunks,
            "Embedding Model": experiment.embedding_model,
            "Top-K": experiment.top_k,
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

import math
from typing import List, Dict, Any


def calculate_recall_at_k(binary_relevances: List[int], total_relevant: int = 1) -> float:
    """Recall@K: fraction of relevant items retrieved in top K."""
    if total_relevant <= 0:
        return 0.0
    relevant_retrieved = sum(binary_relevances)
    return min(1.0, relevant_retrieved / total_relevant)


def calculate_precision_at_k(binary_relevances: List[int], k: int) -> float:
    """Precision@K: fraction of top K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    # Only consider up to k items
    rel_k = binary_relevances[:k]
    return sum(rel_k) / k


def calculate_reciprocal_rank(binary_relevances: List[int]) -> float:
    """Reciprocal Rank (RR): 1 / rank of first relevant item, or 0.0 if not found."""
    for rank, rel in enumerate(binary_relevances, start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def calculate_dcg_at_k(binary_relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    rel_k = binary_relevances[:k]
    dcg = 0.0
    for idx, rel in enumerate(rel_k, start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 1)
    return dcg


def calculate_ndcg_at_k(binary_relevances: List[int], k: int, total_relevant: int = 1) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg = calculate_dcg_at_k(binary_relevances, k)
    # Ideal DCG with all relevant items sorted at top
    ideal_relevances = [1] * min(k, total_relevant)
    idcg = calculate_dcg_at_k(ideal_relevances, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


class BenchmarkMetrics:
    """Computes aggregated evaluation metrics across queries for a strategy run."""

    @staticmethod
    def aggregate(
        query_results: List[Dict[str, Any]],
        top_k: int,
        total_chunks: int,
        avg_chunk_size: float
    ) -> Dict[str, float | int]:
        if not query_results:
            return {
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
                "avg_latency_ms": 0.0,
                "avg_chunk_size": round(avg_chunk_size, 2),
                "total_chunks": total_chunks,
            }

        recalls = []
        precisions = []
        rrs = []
        ndcgs = []
        latencies = []

        for q in query_results:
            # list of 0 or 1 for each retrieved rank
            binary_rels = q.get("binary_relevances", [])
            total_rel = q.get("total_ground_truth_relevant", 1)
            latency = q.get("latency_ms", 0.0)

            recalls.append(calculate_recall_at_k(binary_rels, total_rel))
            precisions.append(calculate_precision_at_k(binary_rels, top_k))
            rrs.append(calculate_reciprocal_rank(binary_rels))
            ndcgs.append(calculate_ndcg_at_k(binary_rels, top_k, total_rel))
            latencies.append(latency)

        n = len(query_results)
        return {
            "recall_at_k": round(sum(recalls) / n, 4),
            "precision_at_k": round(sum(precisions) / n, 4),
            "mrr": round(sum(rrs) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "avg_chunk_size": round(avg_chunk_size, 2),
            "total_chunks": total_chunks,
        }

import pytest
from app.evaluation.coordinate_matcher import (
    compute_bbox_area,
    compute_bbox_iou,
    compute_bbox_containment,
    check_relevance,
)
from app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_reciprocal_rank,
    calculate_dcg_at_k,
    calculate_ndcg_at_k,
    BenchmarkMetrics,
)


def test_bbox_area_and_iou():
    box_a = (0.0, 0.0, 10.0, 10.0)
    assert compute_bbox_area(box_a) == 100.0

    # Identical box -> IoU = 1.0
    assert compute_bbox_iou(box_a, box_a) == 1.0

    # Half overlap
    box_b = (5.0, 0.0, 15.0, 10.0)
    # intersection: [5,0,10,10] = 50. union = 100 + 100 - 50 = 150. IoU = 50/150 = 0.3333
    iou = compute_bbox_iou(box_a, box_b)
    assert round(iou, 3) == 0.333

    # Disjoint box -> IoU = 0.0
    box_c = (20.0, 20.0, 30.0, 30.0)
    assert compute_bbox_iou(box_a, box_c) == 0.0


def test_bbox_containment():
    big_box = (0.0, 0.0, 100.0, 100.0)
    small_box = (20.0, 20.0, 40.0, 40.0)  # area 400

    # big_box contains small_box 100%
    cov = compute_bbox_containment(chunk_bbox=big_box, target_bbox=small_box)
    assert cov == 1.0


def test_check_relevance():
    # Different pages -> False
    is_rel, _ = check_relevance(
        retrieved_page=1,
        retrieved_bbox=(10, 10, 50, 50),
        retrieved_text="some text",
        target_page=2,
        target_bbox=(10, 10, 50, 50)
    )
    assert not is_rel

    # Same page, overlapping coordinates
    is_rel, score = check_relevance(
        retrieved_page=1,
        retrieved_bbox=(50, 100, 300, 200),
        retrieved_text="Text content",
        target_page=1,
        target_bbox=(50, 100, 280, 190),
        iou_threshold=0.2
    )
    assert is_rel
    assert score > 0.5

    # Text containment fallback when bboxes are zero
    is_rel, score = check_relevance(
        retrieved_page=1,
        retrieved_bbox=(0, 0, 0, 0),
        retrieved_text="This is a test chunk containing answer fragment inside.",
        target_page=1,
        target_bbox=(0, 0, 0, 0),
        target_text="answer fragment"
    )
    assert is_rel
    assert score == 1.0


def test_retrieval_metrics():
    # Scenario: 5 retrieved items, relevant item at rank 2
    binary_rels = [0, 1, 0, 0, 0]

    # Recall@5 = 1.0
    assert calculate_recall_at_k(binary_rels, total_relevant=1) == 1.0
    # Recall@1 = 0.0
    assert calculate_recall_at_k(binary_rels[:1], total_relevant=1) == 0.0

    # Precision@5 = 1/5 = 0.2
    assert calculate_precision_at_k(binary_rels, k=5) == 0.2
    # Precision@1 = 0.0
    assert calculate_precision_at_k(binary_rels, k=1) == 0.0

    # MRR: first relevant is at rank 2 -> 1/2 = 0.5
    assert calculate_reciprocal_rank(binary_rels) == 0.5

    # nDCG@5: ideal would have 1 at rank 1. Here it's at rank 2.
    ndcg = calculate_ndcg_at_k(binary_rels, k=5, total_relevant=1)
    assert 0.0 < ndcg < 1.0


def test_benchmark_metrics_aggregation():
    query_results = [
        {"binary_relevances": [1, 0, 0], "total_ground_truth_relevant": 1, "latency_ms": 10.0},
        {"binary_relevances": [0, 1, 0], "total_ground_truth_relevant": 1, "latency_ms": 20.0},
    ]

    metrics = BenchmarkMetrics.aggregate(
        query_results=query_results,
        top_k=3,
        total_chunks=15,
        avg_chunk_size=450.5
    )

    assert metrics["recall_at_k"] == 1.0  # Both found in top 3
    assert metrics["precision_at_k"] == round((1/3 + 1/3) / 2, 4)
    assert metrics["mrr"] == round((1.0 + 0.5) / 2, 4)  # (1/1 + 1/2)/2 = 0.75
    assert metrics["avg_latency_ms"] == 15.0
    assert metrics["total_chunks"] == 15
    assert metrics["avg_chunk_size"] == 450.5

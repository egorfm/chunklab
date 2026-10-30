import json
from pathlib import Path


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "ChunkLab"


def test_full_api_workflow(client, sample_pdf_path):
    # 1. Upload PDF
    with open(sample_pdf_path, "rb") as f:
        res = client.post(
            "/api/documents/upload",
            files={"file": ("sample_benchmark_doc.pdf", f, "application/pdf")}
        )
    assert res.status_code == 201
    doc_data = res.json()
    doc_id = doc_data["id"]
    assert doc_data["num_pages"] == 2

    # 2. Get document detail
    res = client.get(f"/api/documents/{doc_id}")
    assert res.status_code == 200
    detail = res.json()
    assert len(detail["blocks_sample"]) > 0

    # 3. Preview chunks for fixed, recursive, semantic
    for strat in ["fixed", "recursive", "semantic"]:
        res = client.post(
            f"/api/documents/{doc_id}/preview-chunks",
            json={"document_id": doc_id, "strategy": strat, "params": {}}
        )
        assert res.status_code == 200
        chunks = res.json()
        assert len(chunks) > 0
        assert chunks[0]["page_number"] in [1, 2]
        assert len(chunks[0]["bbox"]) == 4

    # 4. Create Evaluation Dataset with queries and ground-truth coordinates
    dataset_payload = {
        "document_id": doc_id,
        "name": "Benchmark QA Dataset",
        "description": "Test dataset evaluating RAG retrieval",
        "queries": [
            {
                "query_text": "What does semantic chunking base its split boundaries on?",
                "ground_truth_answer": "Semantic embedding distance between sentences",
                "target_page": 2,
                "target_bbox": [50.0, 130.0, 550.0, 200.0],
                "target_text": "semantic embedding distance between sentences"
            },
            {
                "query_text": "How does fixed-size chunking divide text?",
                "ground_truth_answer": "Divides text into equal windows with arbitrary overlap boundaries",
                "target_page": 1,
                "target_bbox": [50.0, 220.0, 550.0, 280.0],
                "target_text": "Fixed-size chunking divides text into equal windows"
            }
        ]
    }
    res = client.post("/api/datasets", json=dataset_payload)
    assert res.status_code == 201
    dataset = res.json()
    dataset_id = dataset["id"]
    assert dataset["queries_count"] == 2

    # 5. Run Benchmark Experiment
    experiment_payload = {
        "name": "Chunking Comparison Experiment",
        "document_id": doc_id,
        "dataset_id": dataset_id,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 3,
        "iou_threshold": 0.15,
        "strategies": [
            {"strategy": "fixed", "params": {"chunk_size": 250, "chunk_overlap": 50}},
            {"strategy": "recursive", "params": {"chunk_size": 250, "chunk_overlap": 50}},
            {"strategy": "semantic", "params": {"similarity_threshold": 0.65, "min_chunk_size": 80, "max_chunk_size": 400}}
        ]
    }
    res = client.post("/api/experiments/run", json=experiment_payload)
    assert res.status_code == 201
    exp_res = res.json()
    exp_id = exp_res["id"]
    assert len(exp_res["runs"]) == 3

    # Check metrics for each run
    for run in exp_res["runs"]:
        assert run["strategy"] in ["fixed", "recursive", "semantic"]
        assert run["total_chunks"] > 0
        assert run["avg_chunk_size"] > 0
        assert "recall_at_k" in run
        assert "precision_at_k" in run
        assert "mrr" in run
        assert "ndcg_at_k" in run

    # 6. Export to JSON
    res = client.get(f"/api/experiments/{exp_id}/export/json")
    assert res.status_code == 200
    export_json_data = json.loads(res.text)
    assert export_json_data["experiment_id"] == exp_id
    assert len(export_json_data["runs"]) == 3

    # 7. Export to CSV
    res = client.get(f"/api/experiments/{exp_id}/export/csv")
    assert res.status_code == 200
    assert "Experiment,Strategy,Recall@K" in res.text

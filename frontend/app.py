import os
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import requests

from frontend.components.styles import apply_custom_styles
from frontend.components.visualizer import (
    render_metrics_barchart,
    render_tradeoff_chart,
    render_chunk_profile_chart,
)

# Page configuration
st.set_page_config(
    page_title="ChunkLab — PDF Chunking & Retrieval Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
            <span style="font-size: 24px;">⚡</span>
            <div>
                <span style="font-size: 20px; font-weight: 700; color: #ffffff;">ChunkLab</span>
                <span class="badge badge-fixed" style="margin-left: 6px; font-size: 10px;">v1.0</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Documents", "Evaluation Datasets", "Experiments", "Results & Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    api_online = check_api_health()
    if api_online:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: #10b981;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background-color:#10b981;"></span>
                Backend API Connected
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: #f59e0b;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background-color:#f59e0b;"></span>
                Backend Offline (Start with uvicorn)
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-top: 30px; font-size: 11px; color: #64748b; line-height: 1.4;">
            <b>ChunkLab Benchmark Engine</b><br>
            Coordinate-grounded evaluation for PDF chunking strategies.
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if page == "Dashboard":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">⚡ ChunkLab Benchmark Dashboard</div>
            <div class="hero-subtitle">
                Systematic evaluation of PDF chunking strategies (Fixed-size, Recursive, Semantic)
                grounded in bounding box coordinates with Qdrant vector retrieval.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Fetch stats
    docs, datasets, experiments = [], [], []
    if api_online:
        try:
            docs = requests.get(f"{API_URL}/api/documents").json()
            datasets = requests.get(f"{API_URL}/api/datasets").json()
            experiments = requests.get(f"{API_URL}/api/experiments").json()
        except Exception:
            pass

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Documents</span>
                <span class="metric-value">{len(docs)}</span>
                <span class="metric-subtext">Indexed PDF files</span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Evaluation Datasets</span>
                <span class="metric-value">{len(datasets)}</span>
                <span class="metric-subtext">Ground truth benchmarks</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Experiments</span>
                <span class="metric-value">{len(experiments)}</span>
                <span class="metric-subtext">Benchmark runs conducted</span>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        # Determine top strategy across all runs
        top_strat = "None"
        if experiments:
            strat_recalls = {}
            for e in experiments:
                for r in e.get("runs", []):
                    s = r.get("strategy")
                    strat_recalls.setdefault(s, []).append(r.get("recall_at_k", 0))
            if strat_recalls:
                avg_recalls = {s: sum(vals)/len(vals) for s, vals in strat_recalls.items()}
                top_strat = max(avg_recalls, key=avg_recalls.get).capitalize()

        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Top Performer</span>
                <span class="metric-value" style="color: #60a5fa;">{top_strat}</span>
                <span class="metric-subtext">Highest average recall</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Recent Experiments")
    if experiments:
        table_rows = []
        for e in experiments:
            best_run = max(e.get("runs", []), key=lambda r: r.get("recall_at_k", 0), default={})
            table_rows.append({
                "ID": e["id"],
                "Experiment Name": e["name"],
                "Embedding Model": e["embedding_model"].split("/")[-1],
                "Top-K": e["top_k"],
                "Strategies Evaluated": len(e.get("runs", [])),
                "Best Recall@K": f"{best_run.get('recall_at_k', 0.0):.2f}" if best_run else "-",
                "Best MRR": f"{best_run.get('mrr', 0.0):.2f}" if best_run else "-",
                "Created At": e.get("created_at", "")[:19].replace("T", " ")
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No experiments have been run yet. Go to **Documents** to upload a PDF and **Experiments** to run your first benchmark!")


# ==========================================
# PAGE 2: DOCUMENTS
# ==========================================
elif page == "Documents":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">📄 PDF Documents Management</div>
            <div class="hero-subtitle">
                Upload PDFs, extract text with bounding box coordinates, and preview chunking strategies.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_upload, col_list = st.columns([1, 1])

    with col_upload:
        st.markdown("#### Upload PDF")
        uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])
        if uploaded_file and st.button("Ingest Document", type="primary"):
            if not api_online:
                st.error("Backend API is offline. Cannot upload.")
            else:
                with st.spinner("Extracting blocks and coordinates..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{API_URL}/api/documents/upload", files=files)
                    if res.status_code == 201:
                        st.success(f"Uploaded and indexed '{uploaded_file.name}'!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {res.text}")

    with col_list:
        st.markdown("#### Available Documents")
        docs = []
        if api_online:
            try:
                docs = requests.get(f"{API_URL}/api/documents").json()
            except Exception:
                pass

        if docs:
            for d in docs:
                with st.expander(f"📑 {d['filename']} ({d['num_pages']} pages, {d['file_size']/1024:.1f} KB)"):
                    st.write(f"**ID:** {d['id']} | **Uploaded:** {d['created_at'][:19].replace('T', ' ')}")
                    if st.button("Delete Document", key=f"del_doc_{d['id']}"):
                        requests.delete(f"{API_URL}/api/documents/{d['id']}")
                        st.rerun()
        else:
            st.write("No documents uploaded yet.")

    # Document Inspector & Chunk Preview
    if docs:
        st.markdown("---")
        st.markdown("### 🔍 Document Inspector & Chunk Preview")
        selected_doc_id = st.selectbox(
            "Select Document to Inspect",
            options=[d["id"] for d in docs],
            format_func=lambda x: next((f"{d['filename']} (ID: {d['id']})" for d in docs if d["id"] == x), str(x))
        )

        doc_detail = requests.get(f"{API_URL}/api/documents/{selected_doc_id}").json()

        tab_inspect, tab_preview = st.tabs(["Extracted Blocks & Coordinates", "Live Chunking Preview"])

        with tab_inspect:
            st.markdown(f"**Total Pages:** {doc_detail.get('num_pages')} | **Sample Extracted Blocks:**")
            blocks = doc_detail.get("blocks_sample", [])
            if blocks:
                block_df = pd.DataFrame([
                    {
                        "Page": b["page_number"],
                        "Block #": b["block_index"],
                        "BBox (x0, y0, x1, y1)": f"[{b['bbox'][0]:.1f}, {b['bbox'][1]:.1f}, {b['bbox'][2]:.1f}, {b['bbox'][3]:.1f}]",
                        "Text Preview": b["text"][:120] + ("..." if len(b["text"]) > 120 else "")
                    }
                    for b in blocks
                ])
                st.dataframe(block_df, use_container_width=True, hide_index=True)
            else:
                st.write("No text blocks extracted.")

        with tab_preview:
            col_strat, col_p1, col_p2 = st.columns(3)
            with col_strat:
                prev_strat = st.selectbox("Chunking Strategy", ["fixed", "recursive", "semantic"])
            with col_p1:
                prev_size = st.number_input("Chunk Size (chars)", min_value=100, max_value=2000, value=400, step=50)
            with col_p2:
                prev_overlap = st.number_input("Chunk Overlap (chars)", min_value=0, max_value=500, value=80, step=20)

            if st.button("Generate Preview"):
                with st.spinner("Chunking document..."):
                    payload = {
                        "document_id": selected_doc_id,
                        "strategy": prev_strat,
                        "params": {"chunk_size": prev_size, "chunk_overlap": prev_overlap}
                    }
                    res = requests.post(f"{API_URL}/api/documents/{selected_doc_id}/preview-chunks", json=payload)
                    if res.status_code == 200:
                        preview_chunks = res.json()
                        st.info(f"Generated **{len(preview_chunks)} chunks** using **{prev_strat}** strategy.")
                        for c in preview_chunks[:6]:
                            bbox = c['bbox']
                            st.markdown(f"""
                                <div class="chunk-box">
                                    <div class="chunk-header">
                                        <span><b>Chunk #{c['chunk_index']}</b> (Page {c['page_number']})</span>
                                        <span class="coords-pill">BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]</span>
                                        <span>{c['char_length']} chars • {c['token_count']} tokens</span>
                                    </div>
                                    <div class="chunk-text">{c['text']}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        if len(preview_chunks) > 6:
                            st.caption(f"... and {len(preview_chunks) - 6} more chunks.")
                    else:
                        st.error(f"Error previewing chunks: {res.text}")


# ==========================================
# PAGE 3: EVALUATION DATASETS
# ==========================================
elif page == "Evaluation Datasets":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">🎯 Evaluation Datasets</div>
            <div class="hero-subtitle">
                Create and manage ground truth datasets with queries, target pages, and coordinate bounding boxes.
            </div>
        </div>
    """, unsafe_allow_html=True)

    docs = []
    if api_online:
        try:
            docs = requests.get(f"{API_URL}/api/documents").json()
        except Exception:
            pass

    if not docs:
        st.warning("Please upload at least one PDF document first in the **Documents** tab.")
    else:
        selected_doc = st.selectbox(
            "Document for Evaluation Dataset",
            options=docs,
            format_func=lambda d: f"{d['filename']} (ID: {d['id']})"
        )

        tab_create, tab_import, tab_view = st.tabs(["Create New Dataset", "Import from JSON", "Existing Datasets"])

        with tab_create:
            with st.form("create_dataset_form"):
                ds_name = st.text_input("Dataset Name", placeholder="e.g. Legal Contract Benchmark QA")
                ds_desc = st.text_area("Description", placeholder="Ground truth questions and coordinate targets")

                st.markdown("##### Query 1 (Initial Question)")
                q_text = st.text_input("Question Text", placeholder="What are the indemnification terms?")
                q_ans = st.text_input("Ground Truth Answer", placeholder="Party A indemnifies Party B...")
                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    target_page = st.number_input("Target Page Number", min_value=1, max_value=selected_doc["num_pages"], value=1)
                with col_q2:
                    target_text_frag = st.text_input("Target Text Fragment (Optional)", placeholder="Exact text snippet")

                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                with col_b1:
                    x0 = st.number_input("Target x0", value=50.0, step=10.0)
                with col_b2:
                    y0 = st.number_input("Target y0", value=100.0, step=10.0)
                with col_b3:
                    x1 = st.number_input("Target x1", value=550.0, step=10.0)
                with col_b4:
                    y1 = st.number_input("Target y1", value=250.0, step=10.0)

                submitted = st.form_submit_button("Create Dataset", type="primary")
                if submitted:
                    if not ds_name.strip() or not q_text.strip():
                        st.error("Please fill in dataset name and question text.")
                    else:
                        payload = {
                            "document_id": selected_doc["id"],
                            "name": ds_name,
                            "description": ds_desc,
                            "queries": [
                                {
                                    "query_text": q_text,
                                    "ground_truth_answer": q_ans,
                                    "target_page": target_page,
                                    "target_bbox": [x0, y0, x1, y1],
                                    "target_text": target_text_frag
                                }
                            ]
                        }
                        res = requests.post(f"{API_URL}/api/datasets", json=payload)
                        if res.status_code == 201:
                            st.success(f"Dataset '{ds_name}' created successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error: {res.text}")

        with tab_import:
            st.markdown("Import a JSON file with question-target coordinate pairs.")
            st.markdown("""
                **JSON Format Sample:**
                ```json
                {
                  "name": "Sample Benchmark Dataset",
                  "description": "Evaluation set",
                  "queries": [
                    {
                      "query_text": "What is the primary objective?",
                      "ground_truth_answer": "Accelerate RAG retrieval",
                      "target_page": 1,
                      "target_bbox": [50.0, 120.0, 500.0, 200.0],
                      "target_text": "primary objective is to accelerate"
                    }
                  ]
                }
                ```
            """)
            json_file = st.file_uploader("Upload JSON Dataset", type=["json"])
            if json_file and st.button("Import JSON", type="primary"):
                try:
                    data = json.loads(json_file.getvalue().decode("utf-8"))
                    payload = {
                        "document_id": selected_doc["id"],
                        "name": data.get("name", "Imported Dataset"),
                        "description": data.get("description", ""),
                        "queries": data.get("queries", [])
                    }
                    res = requests.post(f"{API_URL}/api/datasets", json=payload)
                    if res.status_code == 201:
                        st.success("Dataset imported successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        with tab_view:
            datasets = requests.get(f"{API_URL}/api/datasets?document_id={selected_doc['id']}").json()
            if datasets:
                for ds in datasets:
                    with st.expander(f"🎯 {ds['name']} ({ds['queries_count']} queries)"):
                        st.write(f"**ID:** {ds['id']} | **Description:** {ds['description']}")
                        detail = requests.get(f"{API_URL}/api/datasets/{ds['id']}").json()
                        for idx, q in enumerate(detail.get("queries", []), start=1):
                            bbox = [q['target_bbox_x0'], q['target_bbox_y0'], q['target_bbox_x1'], q['target_bbox_y1']]
                            st.markdown(f"""
                                <div style="background-color: #0f172a; padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                                    <b>Q{idx}: {q['query_text']}</b><br>
                                    <span style="font-size: 12px; color: #94a3b8;">
                                        Page: {q['target_page']} • Target BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                        if st.button("Delete Dataset", key=f"del_ds_{ds['id']}"):
                            requests.delete(f"{API_URL}/api/datasets/{ds['id']}")
                            st.rerun()
            else:
                st.info("No evaluation datasets created for this document yet.")


# ==========================================
# PAGE 4: EXPERIMENTS
# ==========================================
elif page == "Experiments":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">⚡ Run Chunking Benchmark Comparison</div>
            <div class="hero-subtitle">
                Configure Fixed-size, Recursive, and Semantic chunking strategies, choose embedding models,
                and launch side-by-side evaluation against ground truth coordinates.
            </div>
        </div>
    """, unsafe_allow_html=True)

    docs = requests.get(f"{API_URL}/api/documents").json() if api_online else []
    if not docs:
        st.warning("Upload a document first in the **Documents** tab.")
    else:
        col_setup1, col_setup2 = st.columns(2)
        with col_setup1:
            selected_doc = st.selectbox(
                "Select Document",
                options=docs,
                format_func=lambda d: f"{d['filename']} (ID: {d['id']})"
            )
        with col_setup2:
            datasets = requests.get(f"{API_URL}/api/datasets?document_id={selected_doc['id']}").json()
            if not datasets:
                st.error("No evaluation dataset found for this document. Create one in the **Evaluation Datasets** tab first!")
                selected_dataset = None
            else:
                selected_dataset = st.selectbox(
                    "Select Evaluation Dataset",
                    options=datasets,
                    format_func=lambda ds: f"{ds['name']} ({ds['queries_count']} queries)"
                )

        if selected_dataset:
            st.markdown("---")
            st.markdown("#### Experiment Parameters")
            exp_name = st.text_input("Experiment Name", value=f"Benchmark_{selected_doc['filename'][:15]}_{int(time.time())%10000}")

            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            with col_cfg1:
                emb_model = st.selectbox(
                    "Embedding Model",
                    [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "BAAI/bge-m3",
                        "intfloat/multilingual-e5-base"
                    ]
                )
            with col_cfg2:
                top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)
            with col_cfg3:
                iou_thresh = st.slider("IoU Coordinate Match Threshold", min_value=0.05, max_value=0.50, value=0.20, step=0.05)

            st.markdown("#### Strategy Configurations")
            col_strat1, col_strat2, col_strat3 = st.columns(3)

            with col_strat1:
                st.markdown("""<span class="badge badge-fixed">Fixed-Size Chunking</span>""", unsafe_allow_html=True)
                fixed_size = st.number_input("Fixed Size (chars)", value=500, step=50, key="fixed_sz")
                fixed_overlap = st.number_input("Fixed Overlap (chars)", value=100, step=25, key="fixed_ov")

            with col_strat2:
                st.markdown("""<span class="badge badge-recursive">Recursive Chunking</span>""", unsafe_allow_html=True)
                rec_size = st.number_input("Recursive Size (chars)", value=500, step=50, key="rec_sz")
                rec_overlap = st.number_input("Recursive Overlap (chars)", value=100, step=25, key="rec_ov")

            with col_strat3:
                st.markdown("""<span class="badge badge-semantic">Semantic Chunking</span>""", unsafe_allow_html=True)
                sem_sim = st.slider("Similarity Threshold", min_value=0.4, max_value=0.9, value=0.70, step=0.05)
                sem_min = st.number_input("Min Chunk Size (chars)", value=150, step=25)
                sem_max = st.number_input("Max Chunk Size (chars)", value=1200, step=50)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Run Benchmark Experiment", type="primary", use_container_width=True):
                payload = {
                    "name": exp_name,
                    "document_id": selected_doc["id"],
                    "dataset_id": selected_dataset["id"],
                    "embedding_model": emb_model,
                    "top_k": top_k,
                    "iou_threshold": iou_thresh,
                    "strategies": [
                        {"strategy": "fixed", "params": {"chunk_size": fixed_size, "chunk_overlap": fixed_overlap}},
                        {"strategy": "recursive", "params": {"chunk_size": rec_size, "chunk_overlap": rec_overlap}},
                        {"strategy": "semantic", "params": {"similarity_threshold": sem_sim, "min_chunk_size": sem_min, "max_chunk_size": sem_max}}
                    ]
                }

                with st.spinner("Executing chunking, embeddings, Qdrant vector indexing, and evaluation queries..."):
                    t0 = time.time()
                    res = requests.post(f"{API_URL}/api/experiments/run", json=payload)
                    elapsed = round(time.time() - t0, 2)

                    if res.status_code == 201:
                        st.success(f"Benchmark completed successfully in {elapsed}s!")
                        exp_data = res.json()
                        st.session_state["last_experiment_id"] = exp_data["id"]
                        st.info("Head over to the **Results & Analysis** tab to view tables, charts, and detailed fragment inspection!")
                    else:
                        st.error(f"Benchmark failed: {res.text}")


# ==========================================
# PAGE 5: RESULTS & ANALYSIS
# ==========================================
elif page == "Results & Analysis":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">📈 Benchmark Results & Retrieval Analysis</div>
            <div class="hero-subtitle">
                Compare Recall@K, Precision@K, MRR, and Latency across chunking strategies.
                Inspect grounded coordinate matches and export results.
            </div>
        </div>
    """, unsafe_allow_html=True)

    experiments = requests.get(f"{API_URL}/api/experiments").json() if api_online else []

    if not experiments:
        st.info("No experiments available to analyze yet. Run one in the **Experiments** tab!")
    else:
        # Preselect last experiment if available
        default_index = 0
        if "last_experiment_id" in st.session_state:
            for idx, e in enumerate(experiments):
                if e["id"] == st.session_state["last_experiment_id"]:
                    default_index = idx
                    break

        selected_exp = st.selectbox(
            "Select Experiment to Analyze",
            options=experiments,
            index=default_index,
            format_func=lambda e: f"#{e['id']} — {e['name']} (Top-{e['top_k']}, {e['embedding_model'].split('/')[-1]})"
        )

        # Fetch full experiment detail
        exp_id = selected_exp["id"]
        exp_detail = requests.get(f"{API_URL}/api/experiments/{exp_id}").json()
        runs = exp_detail.get("runs", [])

        if runs:
            # Top summary cards
            best_recall = max(r.get("recall_at_k", 0) for r in runs)
            best_precision = max(r.get("precision_at_k", 0) for r in runs)
            best_mrr = max(r.get("mrr", 0) for r in runs)
            best_latency = min(r.get("avg_latency_ms", 9999) for r in runs)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Best Recall@{exp_detail['top_k']}</span>
                        <span class="metric-value" style="color: #6366f1;">{best_recall:.2f}</span>
                        <span class="metric-subtext">Passage coverage</span>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Best Precision@{exp_detail['top_k']}</span>
                        <span class="metric-value" style="color: #38bdf8;">{best_precision:.2f}</span>
                        <span class="metric-subtext">Retrieved relevance density</span>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Best MRR</span>
                        <span class="metric-value" style="color: #a855f7;">{best_mrr:.2f}</span>
                        <span class="metric-subtext">Reciprocal rank of 1st hit</span>
                    </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                    <div class="metric-card">
                        <span class="metric-label">Fastest Latency</span>
                        <span class="metric-value" style="color: #10b981;">{best_latency:.1f} ms</span>
                        <span class="metric-subtext">Average query time</span>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📊 Metrics Comparison Table")
            summary_table = [
                {
                    "Strategy": r["strategy"].upper(),
                    f"Recall@{exp_detail['top_k']}": f"{r['recall_at_k']:.4f}",
                    f"Precision@{exp_detail['top_k']}": f"{r['precision_at_k']:.4f}",
                    "MRR": f"{r['mrr']:.4f}",
                    f"nDCG@{exp_detail['top_k']}": f"{r['ndcg_at_k']:.4f}",
                    "Avg Latency (ms)": f"{r['avg_latency_ms']:.2f}",
                    "Avg Chunk Size (chars)": f"{r['avg_chunk_size']:.0f}",
                    "Total Chunks": r["total_chunks"],
                    "Parameters": str(r["strategy_params"])
                }
                for r in runs
            ]
            st.dataframe(pd.DataFrame(summary_table), use_container_width=True, hide_index=True)

            # Export row
            col_exp_csv, col_exp_json, _ = st.columns([1, 1, 3])
            with col_exp_csv:
                csv_data = requests.get(f"{API_URL}/api/experiments/{exp_id}/export/csv").text
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"chunklab_experiment_{exp_id}.csv",
                    mime="text/csv"
                )
            with col_exp_json:
                json_data = requests.get(f"{API_URL}/api/experiments/{exp_id}/export/json").text
                st.download_button(
                    label="📥 Export JSON",
                    data=json_data,
                    file_name=f"chunklab_experiment_{exp_id}.json",
                    mime="application/json"
                )

            # Visualizations
            st.markdown("### 📈 Visual Benchmark Charts")
            tab_chart1, tab_chart2, tab_chart3 = st.tabs(["Quality Metrics", "Quality vs Latency", "Chunk Profile"])

            with tab_chart1:
                st.plotly_chart(render_metrics_barchart(runs, exp_detail["top_k"]), use_container_width=True)
            with tab_chart2:
                st.plotly_chart(render_tradeoff_chart(runs, exp_detail["top_k"]), use_container_width=True)
            with tab_chart3:
                st.plotly_chart(render_chunk_profile_chart(runs), use_container_width=True)

            # Detailed Fragment Inspector
            st.markdown("### 🔎 Retrieved Fragment Inspector")
            st.markdown("Inspect every retrieved chunk with page numbers, coordinates, and ground truth relevance matching.")

            strat_tabs = st.tabs([r["strategy"].capitalize() for r in runs])

            for idx, run in enumerate(runs):
                with strat_tabs[idx]:
                    logs = run.get("logs", [])
                    if not logs:
                        st.write("No retrieval logs recorded.")
                    else:
                        for log in logs:
                            with st.expander(f"Query: \"{log['query_text']}\" (Latency: {log['latency_ms']} ms)"):
                                chunks = log.get("retrieved_chunks", [])
                                if not chunks:
                                    st.write("No chunks retrieved.")
                                for c in chunks:
                                    is_rel = c.get("is_relevant", False)
                                    status_badge = (
                                        f'<span class="badge badge-match">MATCH (Score: {c.get("match_score", 0):.2f})</span>'
                                        if is_rel else
                                        '<span class="badge badge-miss">NO MATCH</span>'
                                    )
                                    bbox = c.get("bbox", [0, 0, 0, 0])
                                    st.markdown(f"""
                                        <div class="chunk-box" style="border-left: 3px solid {'#10b981' if is_rel else '#64748b'};">
                                            <div class="chunk-header">
                                                <span><b>Rank #{c.get('rank', 1)}</b> • Page {c.get('page_number', 1)} • Cosine Score: <b>{c.get('score', 0):.3f}</b></span>
                                                <span class="coords-pill">BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]</span>
                                                {status_badge}
                                            </div>
                                            <div class="chunk-text">{c.get('text', '')}</div>
                                        </div>
                                    """, unsafe_allow_html=True)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any


def render_metrics_barchart(runs: List[Dict[str, Any]], top_k: int) -> go.Figure:
    """Generates an elegant grouped bar chart comparing core retrieval metrics across strategies."""
    strategies = [r["strategy"].capitalize() for r in runs]
    recalls = [r["recall_at_k"] for r in runs]
    precisions = [r["precision_at_k"] for r in runs]
    mrrs = [r["mrr"] for r in runs]
    ndcgs = [r["ndcg_at_k"] for r in runs]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"Recall@{top_k}",
        x=strategies,
        y=recalls,
        marker_color="#6366f1",
        text=[f"{v:.2f}" for v in recalls],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name=f"Precision@{top_k}",
        x=strategies,
        y=precisions,
        marker_color="#38bdf8",
        text=[f"{v:.2f}" for v in precisions],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="MRR",
        x=strategies,
        y=mrrs,
        marker_color="#a855f7",
        text=[f"{v:.2f}" for v in mrrs],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name=f"nDCG@{top_k}",
        x=strategies,
        y=ndcgs,
        marker_color="#10b981",
        text=[f"{v:.2f}" for v in ndcgs],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"Retrieval Quality Benchmark (Top-{top_k})", font=dict(size=16, color="#f8fafc")),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(255, 255, 255, 0.08)"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)")
    )
    return fig


def render_tradeoff_chart(runs: List[Dict[str, Any]], top_k: int) -> go.Figure:
    """Scatter plot illustrating Quality (Recall@K) vs Latency (ms)."""
    df = pd.DataFrame([
        {
            "Strategy": r["strategy"].capitalize(),
            "Recall": r["recall_at_k"],
            "Latency (ms)": r["avg_latency_ms"],
            "Total Chunks": r["total_chunks"],
            "Avg Chunk Size": r["avg_chunk_size"]
        }
        for r in runs
    ])

    fig = px.scatter(
        df,
        x="Latency (ms)",
        y="Recall",
        text="Strategy",
        size="Total Chunks",
        color="Strategy",
        color_discrete_sequence=["#6366f1", "#a855f7", "#10b981"],
        title="Quality vs. Latency Trade-off"
    )

    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(range=[0, 1.1], gridcolor="rgba(255, 255, 255, 0.08)"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)")
    )
    return fig


def render_chunk_profile_chart(runs: List[Dict[str, Any]]) -> go.Figure:
    """Compares total chunk count and average chunk size."""
    strategies = [r["strategy"].capitalize() for r in runs]
    counts = [r["total_chunks"] for r in runs]
    sizes = [r["avg_chunk_size"] for r in runs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Total Chunks",
        x=strategies,
        y=counts,
        marker_color="#f59e0b",
        yaxis="y",
        text=counts,
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Avg Size (chars)",
        x=strategies,
        y=sizes,
        marker_color="#06b6d4",
        yaxis="y2",
        text=[f"{s:.0f}" for s in sizes],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(text="Chunking Profile (Count vs. Average Characters)", font=dict(size=16, color="#f8fafc")),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Total Chunks", gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis2=dict(title="Avg Chunk Size (chars)", overlaying="y", side="right", gridcolor="rgba(255, 255, 255, 0.04)"),
        barmode="group"
    )
    return fig

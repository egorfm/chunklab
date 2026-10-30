import streamlit as st


def apply_custom_styles():
    st.markdown("""
        <style>
        /* Import clean modern font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Clean Linear / Vercel aesthetic */
        .main {
            background-color: #0b0f17;
            color: #f1f5f9;
        }

        /* Hero / Header Card */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        }

        .hero-title {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .hero-subtitle {
            font-size: 14px;
            color: #94a3b8;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-card {
            background-color: #131b2e;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: all 0.2s ease-in-out;
        }

        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.15);
        }

        .metric-label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
        }

        .metric-subtext {
            font-size: 12px;
            color: #64748b;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .badge-fixed {
            background-color: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .badge-recursive {
            background-color: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }

        .badge-semantic {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-match {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .badge-miss {
            background-color: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Chunk Fragment Box */
        .chunk-box {
            background-color: #0f172a;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }

        .chunk-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
            color: #94a3b8;
        }

        .chunk-text {
            font-size: 13px;
            line-height: 1.5;
            color: #e2e8f0;
            background-color: #090d16;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #6366f1;
        }

        /* Coordinates Pill */
        .coords-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            background-color: rgba(255, 255, 255, 0.05);
            padding: 2px 6px;
            border-radius: 4px;
            color: #cbd5e1;
        }
        </style>
    """, unsafe_allow_html=True)

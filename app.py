from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from qdrant_client import QdrantClient
from ingestion.embeddings import embed_text
from cli import (
    load_composite_text,
    # looks_like__query,
    is_ambiguous,
    MIN_SCORE,
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "notification_instruments_cloud"
TOP_K = 3

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# --------------------------------------------------
# PAGE CONFIG + THEME
# --------------------------------------------------
st.set_page_config(
    page_title="Legal Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }
        .title {
            font-size: 2.2rem;
            font-weight: 700;
        }
        .subtitle {
            font-size: 1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .card {
            background: #fafafa;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #eee;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-green {
            background: #e6f4ea;
            color: #137333;
        }
        .badge-yellow {
            background: #fff4e5;
            color: #b26a00;
        }
        .badge-red {
            background: #fdecea;
            color: #b3261e;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="title"> Legal Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Legally strict • Notification-based • No hallucinations '
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# QUERY INPUT
# --------------------------------------------------
query = st.text_input(
    "Ask a question",
    placeholder="e.g. gst exemption for construction services",
)

if query:
    with st.spinner("Analyzing notifications..."):
        vector = embed_text(query)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=TOP_K,
            with_payload=True
        ).points

    # --------------------------------------------------
    # NO RESULT / LOW CONFIDENCE
    # --------------------------------------------------
    if not results or results[0].score < MIN_SCORE:
        st.markdown(
            '<span class="badge badge-red">Rejected</span> '
            'No sufficiently relevant  notification found.',
            unsafe_allow_html=True
        )
        st.stop()

    # --------------------------------------------------
    # STATUTE MISMATCH
    # --------------------------------------------------
    # if not looks_like__query(query):
    #     st.markdown(
    #         '<span class="badge badge-red">Blocked</span> '
    #         'Query appears to relate to a non- statute '
    #         '(e.g. Income Tax, Customs, Excise).',
    #         unsafe_allow_html=True
    #     )
    #     st.stop()

    # --------------------------------------------------
    # AMBIGUITY WARNING
    # --------------------------------------------------
    # ambiguous = is_ambiguous(results)
    # if ambiguous:
    #     st.markdown(
    #         '<span class="badge badge-yellow">Ambiguous</span> '
    #         'Multiple  notifications may be relevant.',
    #         unsafe_allow_html=True
    #     )

    # --------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------
    # --------------------------------------------------
    # SHOW LEGAL TEXT DIRECTLY (PRIMARY VIEW)
    # --------------------------------------------------
    top = results[0]
    payload = top.payload

    st.markdown(
        '<span class="badge badge-green">Relevant  Notification</span>',
        unsafe_allow_html=True
    )

    st.text_area(
        label="",
        value=load_composite_text(payload["group_id"]),
        height=600
    )



    # --------------------------------------------------
    # FUTURE LLM PLACEHOLDER
    # --------------------------------------------------
    st.info(
        "LLM-based explanation can be enabled here later. "
        "Currently showing source notification only."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <small>
    This system explains only the retrieved  notification text.
    It does not infer, assume, or combine laws.
    </small>
    """,
    unsafe_allow_html=True
)

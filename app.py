"""
app.py — Streamlit UI for llm-wiki.

Three tabs: Ingest, Query, Lint.
Each tab calls the FastAPI server and streams the SSE response live.

Run with:
  streamlit run app.py

Requires the API server to be running:
  python3 -m uvicorn api:app --reload --port 8000
"""

import json
import sys
from pathlib import Path

import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="llm-wiki",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #1e1e1e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stMetric label {
        color: #888 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.4rem !important;
        color: #e0e0e0 !important;
    }

    /* Main area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 860px;
    }

    /* Stream output box */
    .stream-box {
        background: #0f0f0f;
        border: 1px solid #1e1e1e;
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.7;
        color: #c8c8c8;
        white-space: pre-wrap;
        min-height: 120px;
        max-height: 520px;
        overflow-y: auto;
    }

    /* Result card */
    .result-card {
        background: #f7f7f5;
        border: 1px solid #e0e0e0;
        border-left: 3px solid #1a1a1a;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.88rem;
    }
    .result-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
    }
    .result-card code {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        background: #efefed;
        padding: 1px 5px;
        border-radius: 3px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.6rem 1.4rem;
        color: #888;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a !important;
        background: transparent !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: #1a1a1a;
        color: #fff;
        border: none;
        border-radius: 3px;
        padding: 0.5rem 1.4rem;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background: #333;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1a1a1a;
        box-shadow: none;
    }

    /* Section labels */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        margin-bottom: 0.4rem;
    }

    /* Wiki header */
    .wiki-header {
        display: flex;
        align-items: baseline;
        gap: 0.8rem;
        margin-bottom: 0.2rem;
    }
    .wiki-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 500;
        color: #1a1a1a;
        letter-spacing: -0.02em;
    }
    .wiki-subtitle {
        font-size: 0.82rem;
        color: #888;
    }

    /* Severity badges */
    .badge-critical { color: #c0392b; font-weight: 500; }
    .badge-warning  { color: #d35400; font-weight: 500; }
    .badge-ok       { color: #27ae60; font-weight: 500; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"


def _api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def _stream_post(endpoint: str, payload: dict):
    """
    POST to endpoint and yield decoded SSE chunks.
    Yields (chunk_text, is_result) tuples.
    is_result=True means this chunk carries the __RESULT__: payload.
    """
    try:
        with requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:].replace("\\n", "\n")
                if payload_str.startswith("__RESULT__:"):
                    yield payload_str, True
                elif payload_str.startswith("__ERROR__:"):
                    yield f"❌ Error: {payload_str[10:]}", False
                else:
                    yield payload_str, False
    except requests.exceptions.ConnectionError:
        yield "❌ Cannot reach API server. Is it running?\n   python3 -m uvicorn api:app --reload --port 8000", False
    except Exception as e:
        yield f"❌ {e}", False


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.8rem 0 1.2rem 0;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
                        font-weight:500; color:#e0e0e0; letter-spacing:-0.02em;">
                📚 llm-wiki
            </div>
            <div style="font-size:0.75rem; color:#555; margin-top:0.2rem;">
                LLM-compiled knowledge base
            </div>
        </div>
        """, unsafe_allow_html=True)

        status = _api_get("/status")

        if status is None:
            st.warning("API offline", icon="⚠️")
            st.caption("Start the server:\n`python3 -m uvicorn api:app --reload`")
            return

        counts = status.get("page_counts", {})
        stale = status.get("stale_summary", {})

        st.markdown('<div class="section-label">Wiki pages</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sources",  counts.get("source", 0))
            st.metric("Concepts", counts.get("concept", 0))
        with col2:
            st.metric("Entities", counts.get("entity", 0))
            st.metric("Queries",  counts.get("query", 0))

        st.divider()

        st.markdown('<div class="section-label">Provenance</div>', unsafe_allow_html=True)
        total_src = stale.get("total_sources", 0)
        stale_src = stale.get("stale_sources", 0)
        open_flags = stale.get("open_flags", 0)

        st.metric("Tracked sources", total_src)
        if open_flags > 0:
            st.markdown(
                f'<div class="badge-warning">⚠ {open_flags} stale flag(s)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="badge-ok">✓ All sources fresh</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        if st.button("↻ Refresh stats"):
            st.rerun()

        st.markdown(
            '<div style="font-size:0.7rem; color:#444; margin-top:1rem;">'
            'API: <code style="color:#666">localhost:8000</code><br>'
            '<a href="http://localhost:8000/docs" target="_blank" '
            'style="color:#555; font-size:0.7rem;">Swagger docs ↗</a>'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Tab: Ingest ────────────────────────────────────────────────────────────────

def _render_ingest_tab():
    st.markdown('<div class="section-label">Source path</div>', unsafe_allow_html=True)
    source_path = st.text_input(
        "source_path",
        placeholder="raw/papers/attention.pdf",
        label_visibility="collapsed",
    )

    title_override = st.text_input(
        "Title override (optional — leave blank for LLM to extract)",
        placeholder="Attention Is All You Need",
    )

    run = st.button("Run ingest", key="ingest_btn")

    if run:
        if not source_path.strip():
            st.error("Enter a source path.")
            return

        output_placeholder = st.empty()
        result_placeholder = st.empty()

        accumulated = ""
        result_data = None

        output_placeholder.markdown(
            f'<div class="stream-box">Starting ingest for {source_path}...\n</div>',
            unsafe_allow_html=True,
        )

        payload = {"source_path": source_path.strip()}
        if title_override.strip():
            payload["title"] = title_override.strip()

        for chunk, is_result in _stream_post("/ingest", payload):
            if is_result:
                result_data = json.loads(chunk[len("__RESULT__:"):])
            else:
                accumulated += chunk
                output_placeholder.markdown(
                    f'<div class="stream-box">{accumulated}</div>',
                    unsafe_allow_html=True,
                )

        if result_data:
            pages = result_data.get("pages_written", [])
            entities = result_data.get("entities_touched", [])
            concepts = result_data.get("concepts_touched", [])
            title = result_data.get("title", "—")

            result_placeholder.markdown(f"""
<div class="result-card">
    <h4>Ingest result</h4>
    <b>Title:</b> {title}<br>
    <b>Pages written:</b> {len(pages)}<br>
    <b>Entities:</b> {', '.join(f'<code>{e}</code>' for e in entities) or '—'}<br>
    <b>Concepts:</b> {', '.join(f'<code>{c}</code>' for c in concepts) or '—'}
</div>
""", unsafe_allow_html=True)


# ── Tab: Query ─────────────────────────────────────────────────────────────────

def _render_query_tab():
    st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
    question = st.text_area(
        "question",
        placeholder="What are the key architectural differences between the Transformer and RNNs?",
        height=90,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        run = st.button("Ask wiki", key="query_btn")
    with col2:
        file_back = st.checkbox("File answer back into wiki", value=False)

    if run:
        if not question.strip():
            st.error("Enter a question.")
            return

        output_placeholder = st.empty()
        result_placeholder = st.empty()

        accumulated = ""
        result_data = None

        output_placeholder.markdown(
            '<div class="stream-box">Searching wiki...\n</div>',
            unsafe_allow_html=True,
        )

        payload = {
            "question": question.strip(),
            "file_back": file_back if file_back else None,
        }

        for chunk, is_result in _stream_post("/query", payload):
            if is_result:
                result_data = json.loads(chunk[len("__RESULT__:"):])
            else:
                accumulated += chunk
                output_placeholder.markdown(
                    f'<div class="stream-box">{accumulated}</div>',
                    unsafe_allow_html=True,
                )

        if result_data:
            pages = result_data.get("pages_consulted", [])
            filed_as = result_data.get("filed_as")
            insufficient = result_data.get("wiki_insufficient", False)

            badge = (
                '<span class="badge-warning">⚠ Wiki insufficient</span>'
                if insufficient
                else '<span class="badge-ok">✓ Answered from wiki</span>'
            )

            filed_line = (
                f'<b>Filed as:</b> <code>wiki/queries/{filed_as}.md</code><br>'
                if filed_as else ""
            )

            result_placeholder.markdown(f"""
<div class="result-card">
    <h4>Query result</h4>
    {badge}<br><br>
    <b>Pages consulted:</b> {', '.join(f'<code>{p}</code>' for p in pages) or '—'}<br>
    {filed_line}
</div>
""", unsafe_allow_html=True)


# ── Tab: Lint ──────────────────────────────────────────────────────────────────

def _render_lint_tab():
    st.markdown(
        "Runs all 6 health checks: orphans, missing stubs, stale provenance, "
        "contradictions, long pages, index drift.",
    )
    st.markdown("")

    run = st.button("Run lint", key="lint_btn")

    if run:
        output_placeholder = st.empty()
        result_placeholder = st.empty()

        accumulated = ""
        result_data = None

        output_placeholder.markdown(
            '<div class="stream-box">Starting lint...\n</div>',
            unsafe_allow_html=True,
        )

        for chunk, is_result in _stream_post("/lint", {}):
            if is_result:
                result_data = json.loads(chunk[len("__RESULT__:"):])
            else:
                accumulated += chunk
                output_placeholder.markdown(
                    f'<div class="stream-box">{accumulated}</div>',
                    unsafe_allow_html=True,
                )

        if result_data:
            findings = result_data.get("findings", [])
            critical = sum(1 for f in findings if f.get("severity") == "critical")
            warnings = sum(1 for f in findings if f.get("severity") == "warning")
            suggestions = sum(1 for f in findings if f.get("severity") == "suggestion")
            report_slug = result_data.get("report_slug", "")
            total_pages = result_data.get("total_pages", 0)

            health_badge = (
                '<span class="badge-critical">🔴 Issues found</span>'
                if critical > 0
                else '<span class="badge-warning">🟡 Warnings</span>'
                if warnings > 0
                else '<span class="badge-ok">✅ Wiki healthy</span>'
            )

            result_placeholder.markdown(f"""
<div class="result-card">
    <h4>Lint result</h4>
    {health_badge}<br><br>
    <b>Pages scanned:</b> {total_pages}<br>
    <b>Critical:</b> {critical} &nbsp;
    <b>Warnings:</b> {warnings} &nbsp;
    <b>Suggestions:</b> {suggestions}<br>
    <b>Report:</b> <code>wiki/queries/{report_slug}.md</code>
</div>
""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _render_sidebar()

    st.markdown("""
    <div class="wiki-header">
        <span class="wiki-title">llm-wiki</span>
        <span class="wiki-subtitle">LLM-compiled persistent knowledge base</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    ingest_tab, query_tab, lint_tab = st.tabs(["ingest", "query", "lint"])

    with ingest_tab:
        _render_ingest_tab()

    with query_tab:
        _render_query_tab()

    with lint_tab:
        _render_lint_tab()


if __name__ == "__main__":
    main()
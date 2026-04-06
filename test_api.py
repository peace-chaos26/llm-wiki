"""
test_api.py — FastAPI endpoint tests using httpx AsyncClient.
Run with: python3 test_api.py  (from repo root)
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent / "src" / "llm_wiki"))

import httpx
from fastapi.testclient import TestClient

from api import app
from config import ensure_dirs, settings
from wiki_manager import bootstrap_index, write_page


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_wiki():
    ensure_dirs()
    bootstrap_index()
    write_page(
        settings.wiki_path / "concepts" / "self-attention.md",
        {"type": "concept", "title": "Self-attention", "status": "developing",
         "sources": ["src-a"], "tags": ["transformers"], "related_concepts": []},
        "\n# Self-attention\n\nMaps queries to key-value pairs.\n"
    )


def _parse_sse_chunks(raw: str) -> list[str]:
    """Extract data payloads from raw SSE response text."""
    chunks = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            chunks.append(payload.replace("\\n", "\n"))
    return chunks


# ── Mock generators ────────────────────────────────────────────────────────────

async def _mock_ingest_gen(path, title=None):
    yield "📄 Reading source...\n"
    yield "✅ Ingest complete\n"
    yield '__RESULT__:{"source_path":"raw/articles/test.txt","title":"Test","source_page_slug":"2026-04-06_test","entities_touched":[],"concepts_touched":[],"pages_written":[],"claims_per_page":{},"warnings":[]}'


async def _mock_query_gen(question, file_back=None):
    yield "🔍 Scanning index...\n"
    yield "Self-attention maps queries to key-value pairs.\n"
    yield '__RESULT__:{"question":"test","answer":"Self-attention maps queries.","pages_consulted":["concepts/self-attention"],"filed_as":null,"wiki_insufficient":false}'


async def _mock_lint_gen():
    yield "🔎 Starting lint...\n"
    yield "✅ Lint complete — 0 critical  0 warnings  0 suggestions\n"
    yield '__RESULT__:{"run_at":"2026-04-06T00:00:00","total_pages":1,"findings":[],"report_slug":"2026-04-06_lint-report"}'


# ── Tests ──────────────────────────────────────────────────────────────────────

def run_tests():
    import shutil

    # Clean state
    for d in [settings.wiki_path, settings.raw_path]:
        if d.exists():
            shutil.rmtree(d)
    if settings.provenance_db_path.exists():
        settings.provenance_db_path.unlink()

    _seed_wiki()

    # Create a fake source file
    src = settings.raw_path / "articles" / "test.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("Test article content.")

    client = TestClient(app, raise_server_exceptions=True)

    print("=== API endpoint tests ===\n")

    # ── GET /health ────────────────────────────────────────────────────────────
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("✓ GET /health")

    # ── GET /status ────────────────────────────────────────────────────────────
    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert "page_counts" in data
    assert "stale_summary" in data
    assert data["index_exists"] is True
    print(f"✓ GET /status  — page_counts: {data['page_counts']}")

    # ── GET /index ─────────────────────────────────────────────────────────────
    r = client.get("/index")
    assert r.status_code == 200
    assert "content" in r.json()
    print("✓ GET /index")

    # ── GET /pages ─────────────────────────────────────────────────────────────
    r = client.get("/pages")
    assert r.status_code == 200
    data = r.json()
    assert "concept" in data
    assert "self-attention" in data["concept"]
    print(f"✓ GET /pages  — concepts: {data['concept']}")

    # ── POST /ingest (mocked pipeline) ────────────────────────────────────────
    with patch("api.ingest_source", side_effect=_mock_ingest_gen):
        r = client.post("/ingest", json={
            "source_path": "raw/articles/test.txt",
            "title": "Test Article",
        })
    assert r.status_code == 200
    chunks = _parse_sse_chunks(r.text)
    assert any("Ingest complete" in c for c in chunks), f"No completion chunk: {chunks}"
    result_chunks = [c for c in chunks if c.startswith("__RESULT__:")]
    assert len(result_chunks) == 1
    result = json.loads(result_chunks[0][11:])
    assert result["title"] == "Test"
    print(f"✓ POST /ingest  — streamed {len(chunks)} chunks, result title: {result['title']}")

    # ── POST /ingest (missing file) ────────────────────────────────────────────
    r = client.post("/ingest", json={"source_path": "raw/nonexistent.txt"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()
    print("✓ POST /ingest  — 404 for missing file")

    # ── POST /query (mocked pipeline) ─────────────────────────────────────────
    with patch("api.query_wiki", side_effect=_mock_query_gen):
        r = client.post("/query", json={"question": "How does self-attention work?"})
    assert r.status_code == 200
    chunks = _parse_sse_chunks(r.text)
    assert any("self-attention" in c.lower() for c in chunks)
    result_chunks = [c for c in chunks if c.startswith("__RESULT__:")]
    assert len(result_chunks) == 1
    result = json.loads(result_chunks[0][11:])
    assert result["wiki_insufficient"] is False
    print(f"✓ POST /query   — streamed {len(chunks)} chunks, wiki_insufficient: {result['wiki_insufficient']}")

    # ── POST /query (empty question) ──────────────────────────────────────────
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 422
    print("✓ POST /query   — 422 for empty question")

    # ── POST /lint (mocked pipeline) ──────────────────────────────────────────
    with patch("api.lint_wiki", side_effect=_mock_lint_gen):
        r = client.post("/lint", json={})
    assert r.status_code == 200
    chunks = _parse_sse_chunks(r.text)
    assert any("Lint complete" in c for c in chunks)
    result_chunks = [c for c in chunks if c.startswith("__RESULT__:")]
    assert len(result_chunks) == 1
    result = json.loads(result_chunks[0][11:])
    assert result["findings"] == []
    print(f"✓ POST /lint    — streamed {len(chunks)} chunks, findings: {len(result['findings'])}")

    print("\n✅ All API tests passed.\n")


if __name__ == "__main__":
    run_tests()
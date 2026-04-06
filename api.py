"""
api.py — FastAPI server for llm-wiki.

Three endpoints, all streaming via Server-Sent Events:
  POST /ingest   — ingest a source file into the wiki
  POST /query    — query the compiled wiki
  POST /lint     — run wiki health checks

All endpoints stream text chunks as SSE events so the UI can display
progress in real time. The final SSE event carries the structured result.

Also exposes:
  GET  /status   — wiki stats (page counts, provenance summary)
  GET  /index    — raw index.md content
  GET  /pages    — list all wiki pages by type

Run with:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/llm_wiki importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src" / "llm_wiki"))

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import ensure_dirs, settings
from ingest import IngestResult, ingest_source
from lint import LintReport, lint_wiki
from provenance import get_source_stats, get_stale_summary
from query import QueryResult, query_wiki
from wiki_manager import bootstrap_index, list_pages, read_index


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bootstrap wiki directories and index on startup."""
    ensure_dirs()
    bootstrap_index()
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="llm-wiki",
    description="LLM-compiled persistent wiki with provenance tracking",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source_path: str = Field(
        ...,
        description="Path to source file relative to repo root. E.g. 'raw/articles/paper.txt'",
        examples=["raw/papers/attention.pdf"],
    )
    title: str | None = Field(
        None,
        description="Optional title override. If omitted, extracted by LLM.",
    )


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to answer from the compiled wiki.",
        examples=["How does self-attention work?"],
    )
    file_back: bool | None = Field(
        None,
        description="File the answer back as a wiki page. None = auto-decide.",
    )


class LintRequest(BaseModel):
    pass   # no parameters needed — lint scans the whole wiki


class WikiStatus(BaseModel):
    wiki_path: str
    page_counts: dict[str, int]
    source_stats: list[dict]
    stale_summary: dict
    index_exists: bool


# ── SSE streaming helper ───────────────────────────────────────────────────────

async def _sse_stream(
    generator: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Wrap an AsyncGenerator of string chunks as SSE events.

    Normal chunks  → data: <chunk>\n\n
    __RESULT__:... → data: __RESULT__:<json>\n\n  (final event)
    Errors         → data: __ERROR__:<message>\n\n
    """
    try:
        async for chunk in generator:
            if chunk.startswith("__RESULT__:") or chunk.startswith("__ERROR__:"):
                yield f"data: {chunk}\n\n"
            else:
                escaped = chunk.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
    except Exception as e:
        yield f"data: __ERROR__:{str(e)}\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    summary="Ingest a source file into the wiki",
    response_description="SSE stream of progress chunks ending with __RESULT__",
)
async def ingest_endpoint(req: IngestRequest) -> StreamingResponse:
    """
    Ingest a raw source file. Streams progress in real time.

    The source file must already exist at the given path relative to repo root.
    Runs the 4-pass LLM pipeline: extract → source page → entity pages → concept pages.
    """
    source_path = Path(req.source_path)
    if not source_path.is_absolute():
        source_path = settings.provenance_db_path.parent / source_path

    if not source_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source file not found: {req.source_path}. "
                   f"Place it inside raw/ first.",
        )

    return StreamingResponse(
        _sse_stream(ingest_source(source_path, title=req.title)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.post(
    "/query",
    summary="Query the compiled wiki",
    response_description="SSE stream of answer chunks ending with __RESULT__",
)
async def query_endpoint(req: QueryRequest) -> StreamingResponse:
    """
    Answer a question from the compiled wiki.

    Streams the answer in real time. The final SSE event carries a
    QueryResult JSON object prefixed with __RESULT__:.
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    return StreamingResponse(
        _sse_stream(query_wiki(req.question, file_back=req.file_back)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/lint",
    summary="Run wiki health checks",
    response_description="SSE stream of lint progress ending with __RESULT__",
)
async def lint_endpoint(_req: LintRequest = LintRequest()) -> StreamingResponse:
    """
    Run all 6 wiki health checks and produce a lint report.

    Checks: orphans, missing stubs, stale provenance, contradictions,
    long pages, index drift. Report is filed as a wiki query page.
    """
    return StreamingResponse(
        _sse_stream(lint_wiki()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/status",
    response_model=WikiStatus,
    summary="Wiki health and stats",
)
async def status_endpoint() -> WikiStatus:
    """
    Returns page counts per type, provenance stats, and stale summary.
    Fast — no LLM calls.
    """
    page_counts = {
        page_type: len(list_pages(page_type))   # type: ignore[arg-type]
        for page_type in ("source", "entity", "concept", "query")
    }
    return WikiStatus(
        wiki_path=str(settings.wiki_path),
        page_counts=page_counts,
        source_stats=get_source_stats(),
        stale_summary=get_stale_summary(),
        index_exists=settings.index_path.exists(),
    )


@app.get(
    "/index",
    summary="Raw index.md content",
    response_description="Markdown text of wiki/index.md",
)
async def index_endpoint() -> dict[str, str]:
    """Returns the raw content of wiki/index.md."""
    content = read_index()
    if not content:
        return {"content": "", "message": "Index is empty — ingest some sources first."}
    return {"content": content}


@app.get(
    "/pages",
    summary="List all wiki pages",
)
async def pages_endpoint() -> dict[str, list[str]]:
    """Returns all wiki page slugs grouped by type."""
    return {
        page_type: [p.stem for p in list_pages(page_type)]  # type: ignore[arg-type]
        for page_type in ("source", "entity", "concept", "query")
    }


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
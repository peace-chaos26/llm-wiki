"""
test_query.py — validates query pipeline logic with mocked LLM calls.
Run with: python3 test_query.py  (from src/llm_wiki/)
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

# ── Mock LLM responses ─────────────────────────────────────────────────────────

MOCK_SELECTION_SUFFICIENT = {
    "relevant_pages": [
        {"path": "concepts/transformer-architecture", "reason": "Core concept page for transformers."},
        {"path": "concepts/self-attention", "reason": "Directly covers self-attention mechanism."},
    ],
    "wiki_sufficient": True,
    "insufficiency_note": "",
}

MOCK_SELECTION_INSUFFICIENT = {
    "relevant_pages": [],
    "wiki_sufficient": False,
    "insufficiency_note": "No pages on quantum computing found in the wiki.",
}

MOCK_ANSWER = """\
## Self-attention in the Transformer

Self-attention is a mechanism that allows each token in a sequence to attend
to every other token, weighting their relevance via scaled dot-product
computation (→ [[concepts/self-attention]]).

The Transformer architecture uses multi-head attention — running 8 parallel
attention heads that each learn different relationship patterns
(→ [[concepts/transformer-architecture]]).

This design replaced recurrent architectures entirely, enabling much greater
parallelism during training (→ [[concepts/transformer-architecture]]).
"""

MOCK_QUERY_PAGE = f"""\
---
type: query
title: "How does self-attention work in the Transformer?"
asked_at: "2026-04-06T00:00:00"
tags: [transformers, attention]
sources_used: [concepts/transformer-architecture, concepts/self-attention]
---

# How does self-attention work in the Transformer?

{MOCK_ANSWER}

## Sources consulted
- [[concepts/transformer-architecture]]
- [[concepts/self-attention]]
"""


# ── Async mock stream ──────────────────────────────────────────────────────────

async def _mock_stream(text: str):
    chunk_size = 60
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_wiki_pages():
    """Write minimal concept pages so _fetch_pages() finds them."""
    from config import settings
    from wiki_manager import write_page

    write_page(
        settings.wiki_path / "concepts" / "transformer-architecture.md",
        {
            "type": "concept",
            "title": "Transformer architecture",
            "status": "developing",
            "sources": ["2026-04-06_attention-is-all-you-need"],
        },
        "\n# Transformer architecture\n\nEncoder-decoder, 6 layers, 8 heads.\n"
        "Multi-head attention enables parallel computation.\n"
    )
    write_page(
        settings.wiki_path / "concepts" / "self-attention.md",
        {
            "type": "concept",
            "title": "Self-attention",
            "status": "developing",
            "sources": ["2026-04-06_attention-is-all-you-need"],
        },
        "\n# Self-attention\n\nMaps queries to key-value pairs via scaled dot-product.\n"
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

async def run_tests():
    import shutil
    from config import ensure_dirs, settings
    from wiki_manager import bootstrap_index

    # Clean state
    for d in [settings.wiki_path, settings.raw_path]:
        if d.exists():
            shutil.rmtree(d)
    if settings.provenance_db_path.exists():
        settings.provenance_db_path.unlink()

    ensure_dirs()
    bootstrap_index()
    _seed_wiki_pages()

    stream_call_count = {"n": 0}

    async def mock_complete(prompt, temperature, system=None):
        # First complete call = page selection (sufficient)
        # Could also be insufficient depending on test
        return json.dumps(MOCK_SELECTION_SUFFICIENT)

    async def mock_stream(prompt, temperature, system=None):
        stream_call_count["n"] += 1
        # Call 1 = answer synthesis, Call 2 = file-back formatting
        responses = [MOCK_ANSWER, MOCK_QUERY_PAGE]
        text = responses[(stream_call_count["n"] - 1) % len(responses)]
        async for chunk in _mock_stream(text):
            yield chunk

    # ── Test 1: Normal query with file-back ────────────────────────────────────
    print("=== Test 1: query with file-back ===\n")
    stream_call_count["n"] = 0

    with patch("query._llm_complete", side_effect=mock_complete), \
         patch("query._llm_stream", side_effect=mock_stream):

        from query import query_wiki, QueryResult

        result = None
        async for chunk in query_wiki(
            "How does self-attention work in the Transformer?",
            file_back=True,
        ):
            if chunk.startswith("__RESULT__:"):
                result = QueryResult(**json.loads(chunk[11:]))
            else:
                print(chunk, end="", flush=True)

    print("\n")
    assert result is not None, "No QueryResult emitted"
    assert result.question == "How does self-attention work in the Transformer?"
    assert len(result.pages_consulted) == 2, f"Expected 2 pages, got {result.pages_consulted}"
    assert "transformer-architecture" in result.pages_consulted[0]
    assert result.filed_as is not None, "Expected answer to be filed back"
    assert not result.wiki_insufficient
    print(f"✓ QueryResult correct: {len(result.pages_consulted)} pages consulted")
    print(f"✓ Filed back as: wiki/queries/{result.filed_as}.md")

    # Query page on disk
    from config import settings
    query_files = list((settings.wiki_path / "queries").glob("*.md"))
    assert len(query_files) == 1, f"Expected 1 query page, found {len(query_files)}"
    print(f"✓ Query page written to disk: {query_files[0].name}")

    # index.md updated
    index_text = settings.index_path.read_text()
    assert "How does self-attention" in index_text, "Query not in index"
    print(f"✓ index.md updated with query")

    # log.md updated
    log_text = settings.log_path.read_text()
    assert "query" in log_text
    assert "self-attention" in log_text
    print(f"✓ log.md has query entry")

    # LLM calls: 1 complete (selection) + 2 stream (answer + file-back)
    assert stream_call_count["n"] == 2, f"Expected 2 stream calls, got {stream_call_count['n']}"
    print(f"✓ LLM calls: 1 complete (selection) + 2 stream (answer + file-back)")

    # ── Test 2: Wiki insufficient ──────────────────────────────────────────────
    print("\n=== Test 2: insufficient wiki ===\n")
    stream_call_count["n"] = 0

    async def mock_complete_insufficient(prompt, temperature, system=None):
        return json.dumps(MOCK_SELECTION_INSUFFICIENT)

    with patch("query._llm_complete", side_effect=mock_complete_insufficient), \
         patch("query._llm_stream", side_effect=mock_stream):

        result2 = None
        async for chunk in query_wiki("How does quantum computing work?"):
            if chunk.startswith("__RESULT__:"):
                result2 = QueryResult(**json.loads(chunk[11:]))
            else:
                print(chunk, end="", flush=True)

    print()
    assert result2 is not None
    assert result2.wiki_insufficient, "Should be marked insufficient"
    assert result2.filed_as is None, "Insufficient answers should not be filed"
    assert stream_call_count["n"] == 0, "No stream calls for insufficient query"
    print(f"✓ Insufficient query handled correctly — no file-back, no stream calls")

    # ── Test 3: Auto file-back heuristic ──────────────────────────────────────
    print("\n=== Test 3: auto file-back heuristic ===\n")
    from query import _should_file_back

    long_answer = "word " * 100
    short_answer = "word " * 30
    insufficient_answer = "The wiki doesn't have enough on this yet. Consider ingesting papers."

    assert _should_file_back(long_answer, "q") is True,  "Long answer should file back"
    assert _should_file_back(short_answer, "q") is False, "Short answer should not file back"
    assert _should_file_back(insufficient_answer, "q") is False, "Insufficient should not file back"
    print(f"✓ _should_file_back heuristic correct for all three cases")

    print("\n✅ All query tests passed.\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
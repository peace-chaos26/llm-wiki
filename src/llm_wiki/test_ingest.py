"""
test_ingest.py — validates ingest pipeline logic with mocked LLM calls.
Run with: python3 test_ingest.py
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# ── Mock LLM responses ─────────────────────────────────────────────────────────

MOCK_EXTRACTION = {
    "title": "Attention Is All You Need",
    "source_type": "paper",
    "tags": ["transformers", "nlp", "attention"],
    "summary": "Introduces the Transformer, a sequence model based entirely on attention mechanisms, dispensing with recurrence and convolutions.",
    "key_claims": [
        "The Transformer relies entirely on attention mechanisms, dispensing with recurrence.",
        "Achieves 28.4 BLEU on WMT 2014 English-to-German translation.",
    ],
    "entities": [
        {"name": "Google Brain", "type": "org", "slug": "google-brain", "exists_in_wiki": False},
        {"name": "Ashish Vaswani", "type": "person", "slug": "ashish-vaswani", "exists_in_wiki": False},
    ],
    "concepts": [
        {
            "name": "Transformer architecture",
            "slug": "transformer-architecture",
            "exists_in_wiki": False,
            "claims_from_this_source": [
                "Encoder-decoder structure, each with 6 stacked layers.",
                "Multi-head attention with 8 parallel attention heads.",
            ],
        },
        {
            "name": "Self-attention",
            "slug": "self-attention",
            "exists_in_wiki": False,
            "claims_from_this_source": [
                "Self-attention maps queries to key-value pairs using scaled dot-product.",
            ],
        },
    ],
    "contradictions": [],
    "questions_raised": ["Does attention fully replace convolutions for vision tasks?"],
}

MOCK_SOURCE_PAGE = """\
---
type: source
title: "Attention Is All You Need"
source_path: "raw/papers/attention.txt"
source_hash: "__HASH__"
ingested_at: "2026-04-06T00:00:00"
tags: [transformers, nlp, attention]
key_entities: [google-brain, ashish-vaswani]
key_concepts: [transformer-architecture, self-attention]
---

# Attention Is All You Need

## Bibliographic info
- **Authors / Origin**: Vaswani et al., Google Brain
- **Published**: 2017
- **Type**: paper

## Summary
Introduces the Transformer architecture based entirely on attention.

## Key claims
- The Transformer relies entirely on attention, dispensing with recurrence.

## Entities mentioned
- [[entities/google-brain]]
- [[entities/ashish-vaswani]]

## Concepts touched
- [[concepts/transformer-architecture]]
- [[concepts/self-attention]]

## Connections
No prior wiki pages exist yet — first source ingested.

## Questions raised
- Does attention fully replace convolutions for vision tasks?
"""

MOCK_ENTITY_PAGE = """\
---
type: entity
entity_type: org
title: "Google Brain"
aliases: []
tags: [ai, research]
sources: [2026-04-06_attention-is-all-you-need]
---

# Google Brain

## Overview
AI research team at Google, responsible for the Transformer paper.

## Appearances in this wiki
- [[sources/2026-04-06_attention-is-all-you-need]]
"""

MOCK_CONCEPT_PAGE = """\
---
type: concept
title: "Transformer architecture"
aliases: []
tags: [transformers, nlp]
sources: [2026-04-06_attention-is-all-you-need]
related_concepts: [self-attention]
status: developing
---

# Transformer architecture

## Overview
Encoder-decoder structure with 6 stacked layers each.

## Claims from sources
- [[sources/2026-04-06_attention-is-all-you-need]]: Encoder-decoder, 6 layers, 8 attention heads.
"""


# ── Async mock helpers ────────────────────────────────────────────────────────

async def _mock_stream(text: str):
    """Yield text in chunks, simulating an LLM stream."""
    chunk_size = 80
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# ── Test runner ───────────────────────────────────────────────────────────────

async def run_tests():
    from config import ensure_dirs, settings
    from wiki_manager import bootstrap_index

    # Reset state for clean test
    import shutil
    for d in [settings.wiki_path, settings.raw_path]:
        if d.exists():
            shutil.rmtree(d)
    if settings.provenance_db_path.exists():
        settings.provenance_db_path.unlink()

    ensure_dirs()
    bootstrap_index()

    # Create a fake source file
    raw_pdf = settings.raw_path / "papers" / "attention.txt"
    raw_pdf.write_text("Fake PDF content for testing.")

    # ── Patch LLM calls ────────────────────────────────────────────────────────
    call_count = {"complete": 0, "stream": 0}

    async def mock_complete(prompt, temperature, system=None):
        call_count["complete"] += 1
        return json.dumps(MOCK_EXTRACTION)

    async def mock_stream(prompt, temperature, system=None):
        call_count["stream"] += 1
        # Rotate through: source page, 2 entity pages, 2 concept pages
        responses = [
            MOCK_SOURCE_PAGE,
            MOCK_ENTITY_PAGE,
            MOCK_ENTITY_PAGE.replace("Google Brain", "Ashish Vaswani").replace("org", "person"),
            MOCK_CONCEPT_PAGE,
            MOCK_CONCEPT_PAGE.replace("Transformer architecture", "Self-attention").replace("transformer-architecture", "self-attention"),
        ]
        text = responses[(call_count["stream"] - 1) % len(responses)]
        async for chunk in _mock_stream(text):
            yield chunk

    with patch("ingest._llm_complete", side_effect=mock_complete), \
         patch("ingest._llm_stream", side_effect=mock_stream):

        from ingest import ingest_source, IngestResult

        print("=== Running ingest pipeline (mocked LLM) ===\n")

        chunks = []
        result = None

        async for chunk in ingest_source(raw_pdf):
            if chunk.startswith("__RESULT__:"):
                result_dict = json.loads(chunk[len("__RESULT__:"):])
                result = IngestResult(**result_dict)
            else:
                print(chunk, end="", flush=True)
                chunks.append(chunk)

    # ── Assertions ─────────────────────────────────────────────────────────────
    print("\n\n=== Assertions ===\n")

    assert result is not None, "No IngestResult emitted"
    assert result.title == "Attention Is All You Need", f"Wrong title: {result.title}"
    assert len(result.entities_touched) == 2, f"Expected 2 entities, got {result.entities_touched}"
    assert len(result.concepts_touched) == 2, f"Expected 2 concepts, got {result.concepts_touched}"
    print(f"✓ IngestResult correct: {result.title}, {len(result.pages_written)} pages")

    # Source page written
    source_page = settings.wiki_path / "sources" / f"2026-04-06_attention-is-all-you-need.md"
    assert source_page.exists(), f"Source page not written: {source_page}"
    content = source_page.read_text()
    assert "__HASH__" not in content, "Hash placeholder not replaced"
    assert len(content.split("source_hash:")[1].split("\n")[0].strip()) > 5, "Empty hash"
    print(f"✓ Source page written with real hash")

    # Entity pages
    for slug in ["google-brain", "ashish-vaswani"]:
        ep = settings.wiki_path / "entities" / f"{slug}.md"
        assert ep.exists(), f"Entity page missing: {slug}"
    print(f"✓ Entity pages written: {result.entities_touched}")

    # Concept pages
    for slug in ["transformer-architecture", "self-attention"]:
        cp = settings.wiki_path / "concepts" / f"{slug}.md"
        assert cp.exists(), f"Concept page missing: {slug}"
    print(f"✓ Concept pages written: {result.concepts_touched}")

    # index.md updated
    index_text = settings.index_path.read_text()
    assert "Attention Is All You Need" in index_text, "Source not in index"
    assert "google-brain" in index_text, "Entity not in index"
    assert "transformer-architecture" in index_text, "Concept not in index"
    print(f"✓ index.md updated with source, entities, concepts")

    # log.md updated
    log_text = settings.log_path.read_text()
    assert "ingest" in log_text, "Log entry missing"
    assert "Attention Is All You Need" in log_text, "Title not in log"
    print(f"✓ log.md has ingest entry")

    # Provenance recorded
    from provenance import get_source_stats, get_page_provenance
    stats = get_source_stats()
    assert len(stats) == 1, f"Expected 1 source stat, got {len(stats)}"
    assert stats[0]["pages_touched"] == len(result.pages_written), (
        f"Provenance pages mismatch: {stats[0]['pages_touched']} vs {len(result.pages_written)}"
    )
    print(f"✓ Provenance: {stats[0]['pages_touched']} pages tracked")

    # Concept provenance has claims
    concept_rel = f"wiki/concepts/transformer-architecture.md"
    prov = get_page_provenance(concept_rel)
    assert len(prov) == 1, f"Expected 1 provenance record for concept, got {len(prov)}"
    assert len(prov[0]["claims"]) > 0, "No claims recorded for concept"
    assert not prov[0]["is_stale"], "Freshly ingested page should not be stale"
    print(f"✓ Claims recorded for transformer-architecture: {prov[0]['claims']}")

    # LLM call counts
    assert call_count["complete"] == 1, f"Expected 1 complete call (extraction), got {call_count['complete']}"
    assert call_count["stream"] == 5, f"Expected 5 stream calls (1 source + 2 entities + 2 concepts), got {call_count['stream']}"
    print(f"✓ LLM calls: 1 extraction + 5 stream = 6 total")

    print("\n✅ All assertions passed.\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
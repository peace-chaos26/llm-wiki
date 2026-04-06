"""
test_lint.py — validates lint pipeline logic with mocked LLM calls.
Run with: python3 test_lint.py  (from src/llm_wiki/)
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_wiki(settings, write_page, bootstrap_index):
    """Seed the wiki with pages that exercise every lint check."""
    bootstrap_index()

    # Concept page with 2 sources — candidate for contradiction check
    write_page(
        settings.wiki_path / "concepts" / "transformer-architecture.md",
        {
            "type": "concept", "title": "Transformer architecture",
            "status": "developing", "sources": ["src-a", "src-b"],
            "tags": ["transformers"], "related_concepts": [],
        },
        "\n# Transformer architecture\n\nEncoder-decoder with 6 layers.\n"
        "Source A says 512 hidden dim. Source B says 1024 hidden dim.\n"
    )

    # Concept page with no inbound links → orphan
    write_page(
        settings.wiki_path / "concepts" / "orphaned-concept.md",
        {
            "type": "concept", "title": "Orphaned Concept",
            "status": "stub", "sources": [], "tags": [], "related_concepts": [],
        },
        "\n# Orphaned Concept\n\nNo other page links here.\n"
    )

    # Entity page — also orphaned
    write_page(
        settings.wiki_path / "entities" / "google-brain.md",
        {
            "type": "entity", "title": "Google Brain",
            "entity_type": "org", "aliases": [], "sources": ["src-a"], "tags": [],
        },
        "\n# Google Brain\n\nAI research team.\n"
    )

    # Source page that references a concept page that doesn't exist yet
    write_page(
        settings.wiki_path / "sources" / "2026-04-06_src-a.md",
        {
            "type": "source", "title": "Source A",
            "source_path": "raw/articles/src-a.txt",
            "source_hash": "abc123", "ingested_at": "2026-04-06T00:00:00",
            "tags": ["test"], "key_entities": [], "key_concepts": ["missing-concept"],
        },
        "\n# Source A\n\nSee [[concepts/missing-concept]] and [[entities/google-brain]].\n"
    )

    # Long concept page — over threshold
    long_body = "\n# Long Concept\n\n" + ("word " * 700)
    write_page(
        settings.wiki_path / "concepts" / "long-concept.md",
        {
            "type": "concept", "title": "Long Concept",
            "status": "mature", "sources": ["src-a", "src-b", "src-c", "src-d"],
            "tags": [], "related_concepts": [],
        },
        long_body,
    )

    # Page NOT in index — index drift check
    write_page(
        settings.wiki_path / "concepts" / "unlisted-concept.md",
        {
            "type": "concept", "title": "Unlisted Concept",
            "status": "stub", "sources": [], "tags": [], "related_concepts": [],
        },
        "\n# Unlisted Concept\n\nNot in the index yet.\n"
    )


def _seed_stale_provenance(settings):
    """Create a source + provenance record, then modify the file to make it stale."""
    from provenance import IngestRecord, record_ingest

    src = settings.raw_path / "articles" / "src-a.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("Original content.")

    record_ingest(IngestRecord(
        source_path="raw/articles/src-a.txt",
        title="Source A",
        wiki_pages_touched=["wiki/concepts/transformer-architecture.md"],
        claims_per_page={
            "wiki/concepts/transformer-architecture.md": [
                "Source A says 512 hidden dim."
            ]
        }
    ))

    # Modify the file → triggers staleness
    time.sleep(0.05)
    src.write_text("Updated content — hidden dim changed to 1024.")


# ── Tests ──────────────────────────────────────────────────────────────────────

async def run_tests():
    import shutil
    from config import ensure_dirs, settings
    from wiki_manager import bootstrap_index, write_page

    # Clean state
    for d in [settings.wiki_path, settings.raw_path]:
        if d.exists():
            shutil.rmtree(d)
    if settings.provenance_db_path.exists():
        settings.provenance_db_path.unlink()

    ensure_dirs()
    _seed_wiki(settings, write_page, bootstrap_index)
    _seed_stale_provenance(settings)

    # ── Individual check tests (no LLM needed) ─────────────────────────────────
    print("=== Unit tests for individual checks ===\n")

    from lint import (
        check_orphans, check_missing_stubs, check_stale_provenance,
        check_long_pages, check_index_drift,
    )

    # Orphans
    orphans = check_orphans()
    orphan_pages = [f.page for f in orphans]
    assert any("orphaned-concept" in p for p in orphan_pages), \
        f"orphaned-concept not flagged. Found: {orphan_pages}"
    print(f"✓ Orphan check: {len(orphans)} orphan(s) found — {orphan_pages}")

    # Missing stubs
    stubs = check_missing_stubs()
    stub_pages = [f.page for f in stubs]
    assert any("missing-concept" in p for p in stub_pages), \
        f"missing-concept not flagged. Found: {stub_pages}"
    # Stub should now exist on disk
    assert (settings.wiki_path / "concepts" / "missing-concept.md").exists(), \
        "Stub file not created on disk"
    print(f"✓ Missing stub check: {len(stubs)} stub(s) created — {stub_pages}")

    # Stale provenance
    stale = check_stale_provenance()
    assert len(stale) >= 1, f"Expected at least 1 stale finding, got {len(stale)}"
    assert stale[0].severity == "critical"  # has claims
    print(f"✓ Stale provenance check: {len(stale)} stale finding(s)")

    # Long pages
    long = check_long_pages()
    long_pages = [f.page for f in long]
    assert any("long-concept" in p for p in long_pages), \
        f"long-concept not flagged. Found: {long_pages}"
    print(f"✓ Long page check: {len(long)} long page(s) — {long_pages}")

    # Index drift
    drift = check_index_drift()
    drift_pages = [f.page for f in drift]
    assert any("unlisted-concept" in p for p in drift_pages), \
        f"unlisted-concept not flagged. Found: {drift_pages}"
    # Should be auto-added to index
    index_text = settings.index_path.read_text()
    assert "unlisted-concept" in index_text, "unlisted-concept not added to index"
    print(f"✓ Index drift check: {len(drift)} drifted page(s) — auto-added to index")

    # ── Full lint pipeline (mocked LLM contradiction check) ────────────────────
    print("\n=== Full lint pipeline (mocked contradiction check) ===\n")

    mock_contradictions = [
        {
            "page": "concepts/transformer-architecture",
            "contradiction": "Conflicting hidden dimension sizes.",
            "claim_a": "Source A says 512 hidden dim.",
            "claim_b": "Source B says 1024 hidden dim.",
        }
    ]

    async def mock_complete(prompt, temperature, system=None):
        return json.dumps(mock_contradictions)

    with patch("lint._llm_complete", side_effect=mock_complete):
        from lint import lint_wiki, LintReport

        report = None
        async for chunk in lint_wiki():
            if chunk.startswith("__RESULT__:"):
                raw = json.loads(chunk[11:])
                # Reconstruct LintReport manually from dict
                from lint import LintFinding
                findings = [LintFinding(**f) for f in raw["findings"]]
                report = LintReport(
                    run_at=raw["run_at"],
                    total_pages=raw["total_pages"],
                    findings=findings,
                    report_slug=raw["report_slug"],
                )
            else:
                print(chunk, end="", flush=True)

    print()
    assert report is not None, "No LintReport emitted"
    assert report.total_pages > 0
    assert report.critical_count >= 1, "Expected at least 1 critical (contradiction)"
    assert report.warning_count >= 1,  "Expected at least 1 warning (orphan/stale)"

    # Report page written to disk
    report_file = settings.wiki_path / "queries" / f"{report.report_slug}.md"
    assert report_file.exists(), f"Report file not on disk: {report_file}"
    report_text = report_file.read_text()
    assert "Lint report" in report_text
    assert "Contradictions" in report_text
    print(f"\n✓ LintReport: {report.summary_line()}")
    print(f"✓ Report filed: {report_file.name}")

    # Log updated
    log_text = settings.log_path.read_text()
    assert "lint" in log_text
    print(f"✓ log.md updated with lint entry")

    # Index updated
    index_text = settings.index_path.read_text()
    assert "lint-report" in index_text
    print(f"✓ index.md updated with lint report")

    print("\n✅ All lint tests passed.\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
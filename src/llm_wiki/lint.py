"""
lint.py — wiki health check pipeline.

Audits the wiki for structural problems and surfaces them as a lint report.
Does not rewrite content — only flags issues, creates stubs for missing pages,
and adds cross-links where obviously missing.

Checks (in order):
  1. Orphan pages       — wiki pages with 0 inbound [[wikilinks]]
  2. Missing stubs      — entities/concepts mentioned in sources but no page exists
  3. Stale provenance   — source files changed since ingest (from provenance.db)
  4. Contradictions     — concept pages with conflicting claims across sources
  5. Long concept pages — pages exceeding settings.concept_split_threshold words
  6. Index drift        — pages on disk not listed in index.md

Each check produces a list of LintFinding objects.
All findings are assembled into a lint report filed as a query page.

Public API
----------
  lint_wiki() -> AsyncGenerator[str, None]
  LintFinding, LintReport  (dataclasses)
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Literal

from openai import AsyncOpenAI

from config import settings
from provenance import check_staleness, get_stale_summary
from wiki_manager import (
    add_index_row,
    append_log,
    build_backlink_map,
    extract_wikilinks_from_page,
    find_long_concept_pages,
    list_all_wiki_pages,
    list_pages,
    page_path,
    parse_page,
    read_index,
    read_schema,
    write_page,
)


# ── Dataclasses ────────────────────────────────────────────────────────────────

Severity = Literal["critical", "warning", "suggestion"]

SEVERITY_EMOJI = {
    "critical":   "🔴",
    "warning":    "🟡",
    "suggestion": "🟢",
}


@dataclass
class LintFinding:
    check: str           # which check produced this
    severity: Severity
    page: str            # wiki page path this finding is about
    message: str         # human-readable description
    action_taken: str = ""   # what lint did automatically (if anything)


@dataclass
class LintReport:
    run_at: str
    total_pages: int
    findings: list[LintFinding] = field(default_factory=list)
    report_slug: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def suggestion_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "suggestion")

    def summary_line(self) -> str:
        return (
            f"{self.critical_count} critical  "
            f"{self.warning_count} warnings  "
            f"{self.suggestion_count} suggestions"
        )


# ── OpenAI client ──────────────────────────────────────────────────────────────

def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


# ── LLM helpers ────────────────────────────────────────────────────────────────

async def _llm_complete(prompt: str, temperature: float, system: str | None = None) -> str:
    client = _client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=settings.max_tokens,
    )
    return response.choices[0].message.content or ""


def _parse_json_response(text: str) -> dict | list:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw:\n{text[:500]}") from e


# ── Check 1: Orphan pages ──────────────────────────────────────────────────────

def check_orphans() -> list[LintFinding]:
    """
    Pages with 0 inbound [[wikilinks]] from other wiki pages.
    index.md links don't count — they're navigation, not knowledge links.
    """
    findings = []
    backlinks = build_backlink_map()

    # Special pages that are exempt from orphan check
    exempt = {"index", "log", "overview"}

    for page in list_all_wiki_pages():
        slug = page.stem
        if slug in exempt:
            continue

        # Get relative path key as used in backlink map
        try:
            rel = str(page.relative_to(settings.wiki_path))
        except ValueError:
            continue

        # Strip .md from key
        rel_key = rel.removesuffix(".md")

        inbound = backlinks.get(rel_key, [])
        # Filter out index.md self-references
        real_inbound = [b for b in inbound if b not in ("index", "log", "overview")]

        if not real_inbound:
            fm, _ = parse_page(page)
            page_type = fm.get("type", "unknown")
            severity: Severity = "warning" if page_type in ("entity", "concept") else "suggestion"
            findings.append(LintFinding(
                check="orphans",
                severity=severity,
                page=rel_key,
                message=f"No inbound links from other wiki pages.",
            ))

    return findings


# ── Check 2: Missing stubs ─────────────────────────────────────────────────────

def check_missing_stubs() -> list[LintFinding]:
    """
    Scan all source pages for entity/concept references that lack wiki pages.
    Creates stub pages automatically and flags them.
    """
    findings = []

    for source_page in list_pages("source"):
        links = extract_wikilinks_from_page(source_page)
        for link in links:
            # Only care about entity and concept links
            if not (link.startswith("entities/") or link.startswith("concepts/")):
                continue

            page_type = "entity" if link.startswith("entities/") else "concept"
            slug = link.split("/", 1)[1]
            target = page_path(page_type, slug)  # type: ignore[arg-type]

            if not target.exists():
                # Create a stub automatically
                title = slug.replace("-", " ").title()
                fm: dict = {
                    "type": page_type,
                    "title": title,
                    "tags": [],
                    "status": "stub" if page_type == "concept" else None,
                    "sources": [],
                }
                if page_type == "concept":
                    fm["related_concepts"] = []
                elif page_type == "entity":
                    fm["entity_type"] = "other"
                    fm["aliases"] = []
                # Remove None values
                fm = {k: v for k, v in fm.items() if v is not None}

                body = (
                    f"\n# {title}\n\n"
                    f"*Stub — created by lint. "
                    f"Referenced in [[{str(source_page.relative_to(settings.wiki_path)).removesuffix('.md')}]] "
                    f"but no page existed.*\n"
                )
                write_page(target, fm, body)

                findings.append(LintFinding(
                    check="missing_stubs",
                    severity="warning",
                    page=link,
                    message=f"Referenced in source pages but no wiki page existed.",
                    action_taken=f"Stub created at wiki/{link}.md",
                ))

    return findings


# ── Check 3: Stale provenance ──────────────────────────────────────────────────

def check_stale_provenance() -> list[LintFinding]:
    """
    Surface stale provenance flags from provenance.db.
    These are wiki pages whose source files have changed since ingest.
    """
    findings = []
    stale_warnings = check_staleness()

    for w in stale_warnings:
        n_claims = len(w.claims)
        claim_word = "claim" if n_claims == 1 else "claims"
        findings.append(LintFinding(
            check="stale_provenance",
            severity="critical" if n_claims > 0 else "warning",
            page=w.wiki_page_path,
            message=(
                f"Source '{w.source_path}' has changed since ingest "
                f"({w.old_hash[:10]}… → {w.new_hash[:10]}…). "
                f"{n_claims} {claim_word} may be outdated."
            ),
        ))

    return findings


# ── Check 4: Contradictions (LLM-assisted) ────────────────────────────────────

async def check_contradictions() -> list[LintFinding]:
    """
    Ask the LLM to scan mature/developing concept pages for contradictions.
    Only runs on pages with 2+ sources — stubs can't contradict anything.
    """
    findings = []

    # Gather candidate pages
    candidates = []
    for p in list_pages("concept"):
        fm, body = parse_page(p)
        sources = fm.get("sources", [])
        status = fm.get("status", "stub")
        if status != "stub" and len(sources) >= 2:
            candidates.append((p, body))

    if not candidates:
        return findings

    # Build a single prompt scanning all candidates
    pages_block = ""
    for p, body in candidates[:10]:  # cap at 10 to stay within context
        rel = str(p.relative_to(settings.wiki_path)).removesuffix(".md")
        pages_block += f"\n=== {rel} ===\n{body[:1500]}\n"

    prompt = textwrap.dedent(f"""\
        Scan the following wiki concept pages for contradictory claims.
        A contradiction is when two statements in the same or related pages
        assert incompatible facts — not just different emphases or nuances.

        {pages_block}

        Return a JSON array. Each element has:
        {{
          "page": "concepts/slug",
          "contradiction": "<describe the contradiction in one sentence>",
          "claim_a": "<first conflicting claim>",
          "claim_b": "<second conflicting claim>"
        }}

        Return an empty array [] if no contradictions are found.
        Return only the JSON array. No preamble.
    """)

    try:
        raw = await _llm_complete(prompt, temperature=settings.lint_temperature)
        result = _parse_json_response(raw)
        if isinstance(result, list):
            for item in result:
                findings.append(LintFinding(
                    check="contradictions",
                    severity="critical",
                    page=item.get("page", "unknown"),
                    message=(
                        f"Contradiction detected: {item.get('contradiction', '')} "
                        f"| Claim A: \"{item.get('claim_a', '')}\" "
                        f"| Claim B: \"{item.get('claim_b', '')}\""
                    ),
                ))
    except (ValueError, KeyError):
        pass  # If LLM fails, skip — don't crash the whole lint

    return findings


# ── Check 5: Long concept pages ────────────────────────────────────────────────

def check_long_pages() -> list[LintFinding]:
    """Flag concept pages exceeding settings.concept_split_threshold words."""
    findings = []
    for page, word_count in find_long_concept_pages():
        rel = str(page.relative_to(settings.wiki_path)).removesuffix(".md")
        findings.append(LintFinding(
            check="long_pages",
            severity="suggestion",
            page=rel,
            message=(
                f"{word_count} words — exceeds threshold of "
                f"{settings.concept_split_threshold}. Consider splitting."
            ),
        ))
    return findings


# ── Check 6: Index drift ───────────────────────────────────────────────────────

def check_index_drift() -> list[LintFinding]:
    """
    Pages that exist on disk but are not listed in index.md.
    Adds missing rows to the index automatically.
    """
    findings = []
    if not settings.index_path.exists():
        return findings

    index_text = read_index()
    exempt_stems = {"index", "log", "overview"}

    for page in list_all_wiki_pages():
        slug = page.stem
        if slug in exempt_stems:
            continue

        # Check if this page slug appears anywhere in index.md
        if slug not in index_text:
            fm, _ = parse_page(page)
            page_type = fm.get("type", "unknown")
            title = fm.get("title", slug)

            # Auto-add to index
            try:
                if page_type == "source":
                    date_str = slug[:10] if len(slug) > 10 else "unknown"
                    add_index_row("source", {
                        "date": date_str,
                        "title": title,
                        "tags": ", ".join(fm.get("tags", [])),
                        "slug": slug,
                    })
                elif page_type == "entity":
                    add_index_row("entity", {
                        "name": title,
                        "entity_type": fm.get("entity_type", "other"),
                        "slug": slug,
                    })
                elif page_type == "concept":
                    add_index_row("concept", {
                        "name": title,
                        "status": fm.get("status", "stub"),
                        "tags": ", ".join(fm.get("tags", [])),
                        "slug": slug,
                    })
                elif page_type == "query":
                    add_index_row("query", {
                        "date": slug[:10] if len(slug) > 10 else "unknown",
                        "question": title,
                        "slug": slug,
                    })
                action = "Added to index.md automatically."
            except Exception:
                action = "Could not auto-add to index — manual fix needed."

            findings.append(LintFinding(
                check="index_drift",
                severity="warning",
                page=str(page.relative_to(settings.wiki_path)).removesuffix(".md"),
                message="Page exists on disk but was missing from index.md.",
                action_taken=action,
            ))

    return findings


# ── Report writer ──────────────────────────────────────────────────────────────

def _render_report(report: LintReport) -> str:
    """Render a LintReport as a markdown query page."""
    now_display = report.run_at.replace("T", " ")
    lines = [
        f"---",
        f"type: query",
        f'title: "Lint report — {now_display}"',
        f"asked_at: \"{report.run_at}\"",
        f"tags: [lint, health-check]",
        f"sources_used: []",
        f"---",
        f"",
        f"# Lint report — {now_display}",
        f"",
        f"**Total pages:** {report.total_pages}  ",
        f"**Findings:** {report.summary_line()}",
        f"",
    ]

    # Group by check
    checks_seen: dict[str, list[LintFinding]] = {}
    for f in report.findings:
        checks_seen.setdefault(f.check, []).append(f)

    if not report.findings:
        lines += ["## Result", "", "✅ No issues found. Wiki is healthy.", ""]
    else:
        check_titles = {
            "orphans": "Orphan pages",
            "missing_stubs": "Missing stubs",
            "stale_provenance": "Stale provenance",
            "contradictions": "Contradictions",
            "long_pages": "Long concept pages",
            "index_drift": "Index drift",
        }
        for check, findings in checks_seen.items():
            title = check_titles.get(check, check)
            lines += [f"## {title} ({len(findings)})", ""]
            for f in findings:
                emoji = SEVERITY_EMOJI[f.severity]
                lines.append(f"{emoji} **`{f.page}`**  ")
                lines.append(f"   {f.message}  ")
                if f.action_taken:
                    lines.append(f"   *Action taken: {f.action_taken}*  ")
                lines.append("")

    lines += [
        "## Stats",
        "",
        f"- Total wiki pages scanned: {report.total_pages}",
        f"- Critical issues: {report.critical_count}",
        f"- Warnings: {report.warning_count}",
        f"- Suggestions: {report.suggestion_count}",
    ]

    stale_summary = get_stale_summary()
    lines += [
        "",
        "## Provenance summary",
        "",
        f"- Total sources tracked: {stale_summary['total_sources']}",
        f"- Sources with changes: {stale_summary['stale_sources']}",
        f"- Wiki pages with stale claims: {stale_summary['stale_wiki_pages']}",
    ]

    return "\n".join(lines)


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def lint_wiki() -> AsyncGenerator[str, None]:
    """
    Run all wiki health checks and produce a lint report.

    Yields string chunks for streaming UI display.
    Final chunk is __RESULT__:{json} containing the LintReport.

    Usage:
        async for chunk in lint_wiki():
            if chunk.startswith("__RESULT__:"):
                report_dict = json.loads(chunk[11:])
            else:
                print(chunk, end="", flush=True)
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    all_pages = list_all_wiki_pages()
    total_pages = len(all_pages)

    report = LintReport(run_at=now, total_pages=total_pages)

    yield f"\n🔎 Starting lint — {total_pages} pages in wiki\n"

    # ── Check 1: Orphans ───────────────────────────────────────────────────────
    yield "\n  [1/6] Checking for orphan pages...\n"
    orphan_findings = check_orphans()
    report.findings.extend(orphan_findings)
    yield f"       Found {len(orphan_findings)} orphan(s)\n"

    # ── Check 2: Missing stubs ─────────────────────────────────────────────────
    yield "\n  [2/6] Checking for missing entity/concept stubs...\n"
    stub_findings = check_missing_stubs()
    report.findings.extend(stub_findings)
    yield f"       Found {len(stub_findings)} missing page(s) — stubs created\n"

    # ── Check 3: Stale provenance ──────────────────────────────────────────────
    yield "\n  [3/6] Checking source provenance for staleness...\n"
    stale_findings = check_stale_provenance()
    report.findings.extend(stale_findings)
    yield f"       Found {len(stale_findings)} stale provenance flag(s)\n"

    # ── Check 4: Contradictions ────────────────────────────────────────────────
    yield "\n  [4/6] Scanning concept pages for contradictions (LLM)...\n"
    contradiction_findings = await check_contradictions()
    report.findings.extend(contradiction_findings)
    yield f"       Found {len(contradiction_findings)} contradiction(s)\n"

    # ── Check 5: Long pages ────────────────────────────────────────────────────
    yield "\n  [5/6] Checking for over-long concept pages...\n"
    long_findings = check_long_pages()
    report.findings.extend(long_findings)
    yield f"       Found {len(long_findings)} page(s) exceeding word threshold\n"

    # ── Check 6: Index drift ───────────────────────────────────────────────────
    yield "\n  [6/6] Checking index for drift...\n"
    drift_findings = check_index_drift()
    report.findings.extend(drift_findings)
    yield f"       Found {len(drift_findings)} page(s) missing from index\n"

    # ── Write report ───────────────────────────────────────────────────────────
    yield "\n📋 Writing lint report...\n"
    report_content = _render_report(report)
    date_str = now[:10]
    report_slug = f"{date_str}_lint-report"
    report.report_slug = report_slug

    report_path = page_path("query", report_slug)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")

    add_index_row("query", {
        "date": date_str,
        "question": f"Lint report — {now[:16].replace('T', ' ')}",
        "slug": report_slug,
    })

    # ── Log ────────────────────────────────────────────────────────────────────
    append_log(
        operation="lint",
        description=f"Lint pass — {report.summary_line()}",
        detail=(
            f"Scanned {total_pages} pages. "
            f"{report.critical_count} critical, "
            f"{report.warning_count} warnings, "
            f"{report.suggestion_count} suggestions."
        ),
        pages_touched=[f"queries/{report_slug}"],
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    yield f"\n✅ Lint complete — {report.summary_line()}\n"
    yield f"   Report filed: wiki/queries/{report_slug}.md\n"

    if report.critical_count > 0:
        yield f"\n🔴 Critical issues:\n"
        for f in report.findings:
            if f.severity == "critical":
                yield f"   • {f.page}: {f.message}\n"

    yield f"__RESULT__:{json.dumps(report.__dict__, default=lambda o: o.__dict__)}"
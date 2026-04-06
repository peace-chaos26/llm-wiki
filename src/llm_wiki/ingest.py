"""
ingest.py — core ingest pipeline.

Orchestrates the full ingest workflow for a single source file:
  1. Read source content
  2. Build LLM context (schema + recent log + current index)
  3. LLM pass 1 — extract key claims, entities, concepts (structured JSON)
  4. LLM pass 2 — write the source wiki page
  5. LLM pass 3 — update/create entity pages (one call per entity)
  6. LLM pass 4 — update/create concept pages (one call per concept)
  7. Update index.md and log.md
  8. Write provenance records to DB

Each LLM pass is a separate, focused call. Smaller prompts with a single
responsibility produce better output than one giant prompt that does everything.

All LLM calls stream responses. The caller receives an AsyncGenerator of
string chunks so the UI can display progress in real time.

Public API
----------
  ingest_source(path, title, *, stream) -> AsyncGenerator[str, None]
  IngestResult  (dataclass returned at the end of the stream)
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from openai import AsyncOpenAI

from config import settings, slugify, source_slug
from provenance import IngestRecord, record_ingest
from wiki_manager import (
    add_index_row,
    append_log,
    extract_wikilinks,
    get_or_create_stub,
    page_path,
    parse_page,
    read_index,
    read_schema,
    recent_log_entries,
    write_page,
)

# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Returned after a successful ingest. Contains a full summary."""
    source_path: str
    title: str
    source_page_slug: str
    entities_touched: list[str] = field(default_factory=list)
    concepts_touched: list[str] = field(default_factory=list)
    pages_written: list[str] = field(default_factory=list)
    claims_per_page: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ── OpenAI client ──────────────────────────────────────────────────────────────

def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


# ── Source reading ─────────────────────────────────────────────────────────────

def read_source(path: Path) -> str:
    """
    Read a raw source file as text.
    Handles .txt, .md, .html (stripped), .pdf (placeholder — wire up
    a PDF extractor like pypdf or pdfminer in production).
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".html":
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Strip tags naively — good enough for most articles
        return re.sub(r"<[^>]+>", " ", raw)

    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            return (
                f"[PDF extraction requires pypdf: pip install pypdf]\n"
                f"File: {path}"
            )

    # Fallback — try reading as text
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Could not read {path}: {e}]"


# ── Context builders ───────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    System prompt for all ingest LLM calls.
    Includes the full CLAUDE.md schema so the LLM knows the wiki conventions.
    """
    schema = read_schema()
    return textwrap.dedent(f"""\
        You are the wiki maintainer for an LLM-compiled knowledge base.
        You follow the schema in CLAUDE.md exactly.

        === CLAUDE.md ===
        {schema}
        ================

        Write all wiki pages as valid markdown with YAML frontmatter.
        Be precise, cite claims to sources, use [[wikilinks]] for all
        internal references. Never invent facts not present in the source.
    """)


def _build_extraction_prompt(
    source_content: str,
    source_path: str,
    index_text: str,
    log_text: str,
) -> str:
    """
    Pass 1 prompt: extract structured metadata from the source.
    Returns JSON — no wiki pages written yet.
    """
    return textwrap.dedent(f"""\
        A new source has been added to the wiki. Analyse it carefully.

        === CURRENT WIKI INDEX ===
        {index_text}
        =========================

        === RECENT LOG (last {settings.log_context_entries} entries) ===
        {log_text}
        ==========================

        === SOURCE FILE: {source_path} ===
        {source_content[:12000]}
        ==================================

        Extract and return a JSON object with this exact structure:
        {{
          "title": "<concise, accurate title for this source>",
          "source_type": "paper | article | transcript | note | dataset",
          "tags": ["tag1", "tag2"],
          "summary": "<2-4 sentence summary of what this source argues or reports>",
          "key_claims": [
            "<specific, citable claim 1>",
            "<specific, citable claim 2>"
          ],
          "entities": [
            {{
              "name": "<canonical entity name>",
              "type": "person | org | model | system | dataset | product | other",
              "slug": "<kebab-case-slug>",
              "exists_in_wiki": true | false
            }}
          ],
          "concepts": [
            {{
              "name": "<concept name>",
              "slug": "<kebab-case-slug>",
              "exists_in_wiki": true | false,
              "claims_from_this_source": [
                "<what this source specifically says about this concept>"
              ]
            }}
          ],
          "contradictions": [
            "<description of any contradiction with existing wiki pages, or empty list>"
          ],
          "questions_raised": [
            "<open question this source leaves unanswered>"
          ]
        }}

        Use the wiki index above to determine whether entities/concepts already
        exist. Return only the JSON object — no preamble, no explanation.
    """)


def _build_source_page_prompt(
    extraction: dict,
    source_path: str,
    source_slug_str: str,
    date_str: str,
) -> str:
    """Pass 2 prompt: write the source wiki page."""
    return textwrap.dedent(f"""\
        Write the wiki source page for this ingested document.
        Use the strict source page template from CLAUDE.md exactly.

        Source path: {source_path}
        Source slug: {source_slug_str}
        Date: {date_str}
        Extraction summary:
        {json.dumps(extraction, indent=2, ensure_ascii=False)}

        Requirements:
        - Frontmatter must include: type, title, source_path, source_hash,
          ingested_at, tags, key_entities, key_concepts
        - Set source_hash to the placeholder string "__HASH__" —
          it will be replaced by ingest.py with the real SHA-256.
        - Use [[wikilinks]] for every entity and concept reference.
        - The "Connections" section must reference specific existing wiki
          pages if any exist (check key_entities and key_concepts for slugs).
        - Return only the raw markdown file content. No explanation.
    """)


def _build_entity_page_prompt(
    entity: dict,
    source_title: str,
    source_slug_str: str,
    existing_content: str,
) -> str:
    """Pass 3 prompt: write or update a single entity page."""
    action = "Update" if existing_content.strip() else "Create"
    return textwrap.dedent(f"""\
        {action} the entity wiki page for: {entity['name']}

        Entity type: {entity['type']}
        Slug: {entity['slug']}
        Source being ingested: {source_title} ([[sources/{source_slug_str}]])

        {'=== EXISTING PAGE CONTENT ===' if existing_content.strip() else '(No existing page — create from scratch.)'}
        {existing_content if existing_content.strip() else ''}
        {'==============================' if existing_content.strip() else ''}

        Requirements:
        - Use the entity page template from CLAUDE.md (flexible structure,
          required frontmatter).
        - Add or update a section for this source's mentions of the entity.
        - Add [[sources/{source_slug_str}]] to the sources list in frontmatter.
        - Use [[wikilinks]] for all cross-references.
        - Do not remove existing content — only add and update.
        - Return only the raw markdown file content. No explanation.
    """)


def _build_concept_page_prompt(
    concept: dict,
    source_title: str,
    source_slug_str: str,
    existing_content: str,
) -> str:
    """Pass 4 prompt: write or update a single concept page."""
    action = "Update" if existing_content.strip() else "Create"
    return textwrap.dedent(f"""\
        {action} the concept wiki page for: {concept['name']}

        Slug: {concept['slug']}
        Source being ingested: {source_title} ([[sources/{source_slug_str}]])

        What this source specifically says about this concept:
        {json.dumps(concept.get('claims_from_this_source', []), indent=2)}

        {'=== EXISTING PAGE CONTENT ===' if existing_content.strip() else '(No existing page — create from scratch.)'}
        {existing_content if existing_content.strip() else ''}
        {'==============================' if existing_content.strip() else ''}

        Requirements:
        - Use the concept page template from CLAUDE.md (flexible structure,
          required frontmatter).
        - Synthesise across all sources, not just this one.
        - If this source contradicts existing content, add a
          ## Tensions and open questions section.
        - Update status: stub → developing if this is the second source;
          developing → mature if well-sourced (4+ sources) and thoroughly covered.
        - Add [[sources/{source_slug_str}]] to the sources list in frontmatter.
        - Use [[wikilinks]] for all cross-references.
        - Do not remove existing content — synthesise and extend.
        - Return only the raw markdown file content. No explanation.
    """)


# ── LLM helpers ────────────────────────────────────────────────────────────────

async def _llm_complete(
    prompt: str,
    temperature: float,
    system: str | None = None,
) -> str:
    """Single non-streaming LLM call. Returns full response text."""
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


async def _llm_stream(
    prompt: str,
    temperature: float,
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming LLM call. Yields text chunks as they arrive."""
    client = _client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _parse_json_response(text: str) -> dict:
    """
    Extract and parse JSON from an LLM response.
    Handles responses with markdown code fences.
    """
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}\n\nRaw response:\n{text[:500]}"
        ) from e


def _fix_source_hash(content: str, real_hash: str) -> str:
    """Replace the __HASH__ placeholder with the real SHA-256 hash."""
    return content.replace("__HASH__", real_hash)


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def ingest_source(
    path: Path | str,
    title: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Full ingest pipeline for a single source file.

    Yields string chunks (progress messages + LLM output) so the caller
    can stream to the UI. The final chunk is a JSON-serialised IngestResult
    prefixed with '__RESULT__:' so the caller can detect and parse it.

    Usage:
        async for chunk in ingest_source(path):
            if chunk.startswith('__RESULT__:'):
                result = IngestResult(**json.loads(chunk[11:]))
            else:
                print(chunk, end='', flush=True)

    Args:
        path:  Path to the source file (inside raw/).
        title: Optional title override. If None, extracted by LLM.
    """
    path = Path(path)
    system_prompt = _build_system_prompt()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = IngestResult(source_path=str(path), title=title or "", source_page_slug="")

    # ── Read source ────────────────────────────────────────────────────────────
    yield f"\n📄 Reading source: {path.name}\n"
    source_content = read_source(path)
    if not source_content.strip():
        yield "⚠️  Source file appears empty. Aborting.\n"
        return

    # Compute hash now — before any LLM calls
    from wiki_manager import hash_file
    source_hash = hash_file(path)

    # ── Pass 1: Extract structured metadata ───────────────────────────────────
    yield "\n🔍 Pass 1/4 — Extracting entities, concepts, and claims...\n"
    index_text = read_index()
    log_text = recent_log_entries()

    extraction_prompt = _build_extraction_prompt(
        source_content, str(path), index_text, log_text
    )
    raw_extraction = await _llm_complete(
        extraction_prompt,
        temperature=settings.ingest_temperature,
        system=system_prompt,
    )

    try:
        extraction = _parse_json_response(raw_extraction)
    except ValueError as e:
        yield f"⚠️  JSON parse error in extraction: {e}\n"
        yield f"    Raw response snippet: {raw_extraction[:300]}\n"
        return

    # Use LLM-extracted title unless caller provided one
    if not result.title:
        result.title = extraction.get("title", path.stem)

    # Build slug now that we have the title
    slug_str = source_slug(date_str, result.title)
    result.source_page_slug = slug_str

    yield f"   Title    : {result.title}\n"
    yield f"   Entities : {[e['name'] for e in extraction.get('entities', [])]}\n"
    yield f"   Concepts : {[c['name'] for c in extraction.get('concepts', [])]}\n"
    if extraction.get("contradictions"):
        yield f"   ⚡ Contradictions flagged: {extraction['contradictions']}\n"

    # ── Pass 2: Write source page ──────────────────────────────────────────────
    yield f"\n📝 Pass 2/4 — Writing source page: wiki/sources/{slug_str}.md\n"
    source_page_prompt = _build_source_page_prompt(
        extraction, str(path), slug_str, date_str
    )

    source_page_content = ""
    async for chunk in _llm_stream(
        source_page_prompt,
        temperature=settings.ingest_temperature,
        system=system_prompt,
    ):
        source_page_content += chunk
        yield chunk

    # Replace hash placeholder
    source_page_content = _fix_source_hash(source_page_content, source_hash)

    # Write to disk
    dest = page_path("source", slug_str)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source_page_content, encoding="utf-8")
    result.pages_written.append(str(dest.relative_to(settings.wiki_path.parent)))

    yield f"\n   ✓ Written: {dest.relative_to(settings.wiki_path.parent)}\n"

    # ── Pass 3: Entity pages ───────────────────────────────────────────────────
    entities = extraction.get("entities", [])
    yield f"\n👤 Pass 3/4 — Updating {len(entities)} entity page(s)...\n"

    for entity in entities:
        entity_slug = entity.get("slug") or slugify(entity["name"])
        yield f"\n   Entity: {entity['name']} ({entity_slug})\n"

        epath = page_path("entity", entity_slug)
        existing = epath.read_text(encoding="utf-8") if epath.exists() else ""

        entity_prompt = _build_entity_page_prompt(
            entity, result.title, slug_str, existing
        )
        entity_content = ""
        async for chunk in _llm_stream(
            entity_prompt,
            temperature=settings.ingest_temperature,
            system=system_prompt,
        ):
            entity_content += chunk
            yield chunk

        epath.parent.mkdir(parents=True, exist_ok=True)
        epath.write_text(entity_content, encoding="utf-8")

        rel = str(epath.relative_to(settings.wiki_path.parent))
        result.pages_written.append(rel)
        result.entities_touched.append(entity_slug)

        # Add index row only for new entity pages
        if not existing.strip():
            add_index_row("entity", {
                "name": entity["name"],
                "entity_type": entity.get("type", "other"),
                "slug": entity_slug,
            })

        yield f"\n   ✓ Written: {rel}\n"

    # ── Pass 4: Concept pages ──────────────────────────────────────────────────
    concepts = extraction.get("concepts", [])
    yield f"\n💡 Pass 4/4 — Updating {len(concepts)} concept page(s)...\n"

    for concept in concepts:
        concept_slug = concept.get("slug") or slugify(concept["name"])
        yield f"\n   Concept: {concept['name']} ({concept_slug})\n"

        cpath = page_path("concept", concept_slug)
        existing = cpath.read_text(encoding="utf-8") if cpath.exists() else ""

        concept_prompt = _build_concept_page_prompt(
            concept, result.title, slug_str, existing
        )
        concept_content = ""
        async for chunk in _llm_stream(
            concept_prompt,
            temperature=settings.ingest_temperature,
            system=system_prompt,
        ):
            concept_content += chunk
            yield chunk

        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(concept_content, encoding="utf-8")

        rel = str(cpath.relative_to(settings.wiki_path.parent))
        result.pages_written.append(rel)
        result.concepts_touched.append(concept_slug)

        # Collect claims for provenance
        claims = concept.get("claims_from_this_source", [])
        result.claims_per_page[rel] = claims

        # Add index row only for new concept pages
        if not existing.strip():
            add_index_row("concept", {
                "name": concept["name"],
                "status": "stub",
                "tags": ", ".join(extraction.get("tags", [])),
                "slug": concept_slug,
            })

        yield f"\n   ✓ Written: {rel}\n"

    # ── Update index.md ────────────────────────────────────────────────────────
    yield "\n📚 Updating index.md...\n"
    add_index_row("source", {
        "date": date_str,
        "title": result.title,
        "tags": ", ".join(extraction.get("tags", [])),
        "slug": slug_str,
    })
    yield "   ✓ index.md updated\n"

    # ── Append to log.md ───────────────────────────────────────────────────────
    pages_touched = result.pages_written
    append_log(
        operation="ingest",
        description=result.title,
        detail=(
            f"Ingested {path.name}. "
            f"Wrote {len(result.pages_written)} wiki pages. "
            f"Entities: {result.entities_touched}. "
            f"Concepts: {result.concepts_touched}."
        ),
        pages_touched=pages_touched,
    )
    yield "   ✓ log.md updated\n"

    # ── Write provenance ───────────────────────────────────────────────────────
    yield "\n🔗 Writing provenance records...\n"

    # Add source page to claims map (no claims — it's the summary page)
    source_page_rel = str(
        page_path("source", slug_str).relative_to(settings.wiki_path.parent)
    )
    result.claims_per_page[source_page_rel] = []

    provenance_record = IngestRecord(
        source_path=str(path),
        title=result.title,
        wiki_pages_touched=result.pages_written,
        claims_per_page=result.claims_per_page,
    )
    record_ingest(provenance_record)
    yield "   ✓ Provenance recorded\n"

    # ── Done ───────────────────────────────────────────────────────────────────
    yield f"\n✅ Ingest complete: {result.title}\n"
    yield f"   Pages written : {len(result.pages_written)}\n"
    yield f"   Entities      : {len(result.entities_touched)}\n"
    yield f"   Concepts      : {len(result.concepts_touched)}\n"

    # Emit result as final chunk for programmatic callers
    yield f"__RESULT__:{json.dumps(result.__dict__)}"
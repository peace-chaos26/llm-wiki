"""
query.py — wiki query pipeline.

Answers questions by reasoning over the compiled wiki, not raw sources.
The LLM never sees raw/ — only wiki/index.md and the pages it selects.

Flow:
  1. Read index.md to identify relevant pages
  2. LLM selects which pages to read (structured JSON)
  3. Fetch selected pages from disk
  4. LLM synthesises an answer with inline wiki citations
  5. Optionally file the answer back as a query page (compounds the wiki)

Public API
----------
  query_wiki(question, *, file_back) -> AsyncGenerator[str, None]
  QueryResult  (dataclass returned as __RESULT__: sentinel chunk)
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

from config import settings, slugify
from wiki_manager import (
    add_index_row,
    append_log,
    page_path,
    parse_page,
    read_index,
    read_schema,
    recent_log_entries,
    resolve_wikilink,
    write_page,
)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    question: str
    answer: str
    pages_consulted: list[str] = field(default_factory=list)
    filed_as: str | None = None          # query page slug if filed back
    wiki_insufficient: bool = False      # True if wiki lacked enough info


# ── OpenAI client ──────────────────────────────────────────────────────────────

def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


# ── Context builders ───────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    schema = read_schema()
    return textwrap.dedent(f"""\
        You are the query interface for an LLM-compiled knowledge base.
        You answer questions strictly from compiled wiki pages.
        You never answer from raw sources or general knowledge.
        If the wiki doesn't contain enough to answer well, say so clearly.

        === CLAUDE.md ===
        {schema}
        ================

        Citation format: every factual claim must cite its wiki page inline:
        (→ [[concepts/slug]]) or (→ [[sources/slug]]).
        Never cite raw source files directly.
    """)


def _build_page_selection_prompt(question: str, index_text: str, log_text: str) -> str:
    """
    Pass 1: given the question and index, select which pages to read.
    Returns JSON — no answer generated yet.
    """
    return textwrap.dedent(f"""\
        A user has asked the following question:

        "{question}"

        === CURRENT WIKI INDEX ===
        {index_text}
        =========================

        === RECENT LOG ===
        {log_text}
        ==================

        Identify the wiki pages most relevant to answering this question.
        Return a JSON object with this exact structure:

        {{
          "relevant_pages": [
            {{
              "path": "concepts/slug-name",
              "reason": "<one sentence: why this page is relevant>"
            }}
          ],
          "wiki_sufficient": true | false,
          "insufficiency_note": "<if wiki_sufficient is false: what is missing>"
        }}

        Rules:
        - Include only pages listed in the index above.
        - Use the path format without leading 'wiki/' — e.g. 'concepts/rag', 'sources/2026-04-06_x'.
        - Include between 1 and 8 pages. More than 8 is usually noise.
        - Set wiki_sufficient to TRUE if any relevant pages exist in the index, even if they are stubs or partially developed. The answer LLM will determine if content is sufficient — your job is only page selection.
        - Only set wiki_sufficient to FALSE if the index has zero pages related to the question whatsoever.
        - Return only the JSON object. No preamble, no explanation.
    """)


def _build_answer_prompt(
    question: str,
    pages: dict[str, str],   # path → content
) -> str:
    """
    Pass 2: synthesise an answer from the fetched wiki pages.
    """
    pages_block = ""
    for path, content in pages.items():
        pages_block += f"\n=== [[{path}]] ===\n{content}\n"

    return textwrap.dedent(f"""\
        Answer the following question using only the wiki pages provided below.

        Question: {question}

        {pages_block}

        Requirements:
        - Synthesise across pages — don't just summarise one.
        - Cite every factual claim inline: (→ [[path/slug]]).
        - If pages partially answer the question, answer what you can and
          note what the wiki doesn't yet cover.
        - If pages are insufficient, say: "The wiki doesn't have enough on
          [X] yet. Consider ingesting [source type] to answer this fully."
        - Write in clear prose. Use headers if the answer is multi-part.
        - Do not invent facts not present in the provided pages.
    """)


def _build_file_back_prompt(
    question: str,
    answer: str,
    pages_consulted: list[str],
) -> str:
    """
    Pass 3 (optional): reformat the answer as a standalone query wiki page.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    sources_links = "\n".join(f"- [[{p}]]" for p in pages_consulted)
    return textwrap.dedent(f"""\
        Reformat the following Q&A as a standalone wiki query page.
        Use the query page template from CLAUDE.md exactly.

        Question: {question}
        Asked at: {now}
        Pages consulted:
        {sources_links}

        Answer to reformat:
        {answer}

        Requirements:
        - Write as a standalone piece a future reader could understand
          without the chat context.
        - Preserve all [[wikilink]] citations from the answer.
        - Add a ## Sources consulted section at the end.
        - Return only the raw markdown file content. No explanation.
    """)


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


async def _llm_stream(
    prompt: str, temperature: float, system: str | None = None
) -> AsyncGenerator[str, None]:
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
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw:\n{text[:500]}") from e


# ── Page fetching ──────────────────────────────────────────────────────────────

def _fetch_pages(page_paths: list[str]) -> dict[str, str]:
    """
    Read wiki pages from disk given their relative paths.
    Returns {path: full_content}. Silently skips missing pages.
    """
    pages = {}
    for rel_path in page_paths:
        # Normalise: strip leading wiki/ if present
        rel_path = rel_path.removeprefix("wiki/")
        full_path = settings.wiki_path / f"{rel_path}.md"
        if full_path.exists():
            pages[rel_path] = full_path.read_text(encoding="utf-8")
        else:
            # Try resolving as a wikilink
            resolved = resolve_wikilink(rel_path)
            if resolved.exists():
                pages[rel_path] = resolved.read_text(encoding="utf-8")
    return pages


def _should_file_back(answer: str, question: str) -> bool:
    """
    Heuristic: file back answers that are substantive synthesis.
    Skip short answers, "wiki insufficient" responses, and simple lookups.
    """
    if len(answer.split()) < 80:
        return False
    if "doesn't have enough" in answer.lower():
        return False
    if "consider ingesting" in answer.lower():
        return False
    return True


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def query_wiki(
    question: str,
    file_back: bool | None = None,
) -> AsyncGenerator[str, None]:
    """
    Query the compiled wiki and stream an answer.

    Args:
        question:  The question to answer.
        file_back: If True, file the answer back as a query page.
                   If None (default), auto-decide based on answer quality.
                   If False, never file back.

    Yields string chunks. Final chunk is __RESULT__:{json} sentinel.

    Usage:
        async for chunk in query_wiki("What is self-attention?"):
            if chunk.startswith("__RESULT__:"):
                result = QueryResult(**json.loads(chunk[11:]))
            else:
                print(chunk, end="", flush=True)
    """
    system_prompt = _build_system_prompt()
    result = QueryResult(question=question, answer="")

    # ── Pass 1: Page selection ─────────────────────────────────────────────────
    yield f"\n🔍 Scanning index for pages relevant to: \"{question}\"\n"

    index_text = read_index()
    if not index_text.strip():
        yield "⚠️  Wiki index is empty. Ingest some sources first.\n"
        result.wiki_insufficient = True
        yield f"__RESULT__:{json.dumps(result.__dict__)}"
        return

    log_text = recent_log_entries()
    selection_prompt = _build_page_selection_prompt(question, index_text, log_text)

    raw_selection = await _llm_complete(
        selection_prompt,
        temperature=settings.query_temperature,
        system=system_prompt,
    )

    try:
        selection = _parse_json_response(raw_selection)
    except ValueError as e:
        yield f"⚠️  Page selection parse error: {e}\n"
        yield f"__RESULT__:{json.dumps(result.__dict__)}"
        return

    relevant_pages = selection.get("relevant_pages", [])
    wiki_sufficient = selection.get("wiki_sufficient", True)

    if not relevant_pages or not wiki_sufficient:
        note = selection.get("insufficiency_note", "No relevant pages found.")
        yield f"\n⚠️  Wiki insufficient: {note}\n"
        yield "   Tip: ingest more sources on this topic first.\n"
        result.wiki_insufficient = True
        result.answer = f"The wiki doesn't have enough on this yet. {note}"
        append_log(
            operation="query",
            description=question[:80],
            detail=f"Wiki insufficient: {note}",
        )
        yield f"__RESULT__:{json.dumps(result.__dict__)}"
        return

    page_paths = [p["path"] for p in relevant_pages]
    yield f"\n   Selected {len(page_paths)} page(s):\n"
    for p in relevant_pages:
        yield f"   • {p['path']} — {p['reason']}\n"

    # ── Fetch pages ────────────────────────────────────────────────────────────
    yield "\n📖 Reading wiki pages...\n"
    pages = _fetch_pages(page_paths)

    if not pages:
        yield "⚠️  Selected pages not found on disk. Wiki may need a lint pass.\n"
        result.wiki_insufficient = True
        yield f"__RESULT__:{json.dumps(result.__dict__)}"
        return

    missing = set(page_paths) - set(pages.keys())
    if missing:
        yield f"   ⚠️  Could not find: {list(missing)} — skipping.\n"

    result.pages_consulted = list(pages.keys())
    yield f"   Read {len(pages)} page(s).\n"

    # ── Pass 2: Synthesise answer ──────────────────────────────────────────────
    yield f"\n💬 Synthesising answer...\n\n"

    answer_prompt = _build_answer_prompt(question, pages)
    answer_text = ""

    async for chunk in _llm_stream(
        answer_prompt,
        temperature=settings.query_temperature,
        system=system_prompt,
    ):
        answer_text += chunk
        yield chunk

    result.answer = answer_text
    yield "\n"

    # ── Pass 3 (optional): File back ───────────────────────────────────────────
    should_file = (
        file_back
        if file_back is not None
        else _should_file_back(answer_text, question)
    )

    if should_file:
        yield "\n📁 Filing answer back into wiki...\n"

        file_back_prompt = _build_file_back_prompt(
            question, answer_text, result.pages_consulted
        )
        query_page_content = ""
        async for chunk in _llm_stream(
            file_back_prompt,
            temperature=settings.query_temperature,
            system=system_prompt,
        ):
            query_page_content += chunk

        # Write query page
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query_slug = f"{date_str}_{slugify(question[:60])}"
        qpath = page_path("query", query_slug)
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text(query_page_content, encoding="utf-8")

        result.filed_as = query_slug
        rel = str(qpath.relative_to(settings.wiki_path.parent))

        add_index_row("query", {
            "date": date_str,
            "question": question[:80],
            "slug": query_slug,
        })

        yield f"   ✓ Filed: {rel}\n"

    # ── Log + done ─────────────────────────────────────────────────────────────
    append_log(
        operation="query",
        description=question[:80],
        detail=(
            f"Consulted {len(result.pages_consulted)} page(s). "
            f"{'Filed back as ' + result.filed_as if result.filed_as else 'Not filed back.'}"
        ),
        pages_touched=result.pages_consulted,
    )

    yield f"\n✅ Query complete.\n"
    if result.filed_as:
        yield f"   Answer filed as: wiki/queries/{result.filed_as}.md\n"

    result_payload = {
        "question": result.question,
        "pages_consulted": result.pages_consulted,
        "filed_as": result.filed_as,
        "wiki_insufficient": result.wiki_insufficient,
    }
    yield f"__RESULT__:{json.dumps(result_payload)}"
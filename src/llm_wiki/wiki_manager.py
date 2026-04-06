"""
wiki_manager.py — all filesystem I/O for the wiki.

Every other module that needs to read, write, or list wiki pages
calls this module. Nothing else touches wiki/ directly.

Responsibilities:
  - Read / write / list wiki pages
  - Parse and write YAML frontmatter
  - Maintain index.md and log.md
  - Compute source file hashes
  - Wikilink extraction and backlink helpers
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import (
    PageType,
    WIKI_SUBDIRS,
    settings,
    slugify,
    source_slug,
)


# ── Frontmatter parsing ────────────────────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_page(path: Path) -> tuple[dict[str, Any], str]:
    """
    Read a wiki page and split it into (frontmatter dict, body markdown).
    Returns ({}, raw_text) if the file has no frontmatter block.
    """
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    body = raw[match.end():]
    return fm, body


def write_page(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    """
    Write a wiki page: YAML frontmatter block + markdown body.
    Creates parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_text = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    path.write_text(f"---\n{fm_text}---\n{body}", encoding="utf-8")


def update_frontmatter(path: Path, updates: dict[str, Any]) -> None:
    """
    Merge `updates` into the frontmatter of an existing page.
    Body is left untouched.
    """
    fm, body = parse_page(path)
    fm.update(updates)
    write_page(path, fm, body)


# ── Page resolution ────────────────────────────────────────────────────────────

def page_path(page_type: PageType, slug: str) -> Path:
    """
    Return the canonical Path for a wiki page given its type and slug.
    slug may include subdirectory parts (e.g. '2026-04-06_some-title').
    Does not require the file to exist.
    """
    subdir = WIKI_SUBDIRS[page_type]
    return settings.wiki_path / subdir / f"{slug}.md"


def resolve_wikilink(link: str) -> Path:
    """
    Resolve a [[wikilink]] string to an absolute Path.
    Strips [[ ]] and .md if present. Relative to wiki root.
    Examples:
      '[[concepts/rag]]'       → wiki/concepts/rag.md
      'entities/openai'        → wiki/entities/openai.md
      '[[sources/2026-04-06_x]]' → wiki/sources/2026-04-06_x.md
    """
    link = link.strip("[]").strip()
    if link.endswith(".md"):
        link = link[:-3]
    return settings.wiki_path / f"{link}.md"


# ── Listing pages ──────────────────────────────────────────────────────────────

def list_pages(page_type: PageType) -> list[Path]:
    """Return all .md files under the given page type directory, sorted."""
    subdir = WIKI_SUBDIRS[page_type]
    base = settings.wiki_path / subdir
    if not base.exists():
        return []
    return sorted(base.glob("*.md"))


def list_all_wiki_pages() -> list[Path]:
    """Return every .md file under wiki/, including index/log/overview."""
    if not settings.wiki_path.exists():
        return []
    return sorted(settings.wiki_path.rglob("*.md"))


# ── Wikilink extraction ────────────────────────────────────────────────────────

WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


def extract_wikilinks(text: str) -> list[str]:
    """
    Return all [[wikilink]] targets found in text.
    Returns raw inner strings, e.g. ['concepts/rag', 'entities/openai'].
    """
    return WIKILINK_RE.findall(text)


def extract_wikilinks_from_page(path: Path) -> list[str]:
    """Extract all wikilinks from a page file (frontmatter + body)."""
    if not path.exists():
        return []
    return extract_wikilinks(path.read_text(encoding="utf-8"))


def build_backlink_map() -> dict[str, list[str]]:
    """
    Scan every wiki page and return a map of:
      target_slug → [list of page slugs that link to it]

    Used by lint to detect orphan pages.
    """
    backlinks: dict[str, list[str]] = {}
    for page in list_all_wiki_pages():
        links = extract_wikilinks_from_page(page)
        source_slug_str = page.stem
        for link in links:
            # Normalise: strip leading wiki/ if present
            target = link.removeprefix("wiki/")
            backlinks.setdefault(target, [])
            if source_slug_str not in backlinks[target]:
                backlinks[target].append(source_slug_str)
    return backlinks


# ── Source file hashing ────────────────────────────────────────────────────────

def hash_file(path: Path) -> str:
    """
    Compute SHA-256 hash of a file. Used to detect when raw sources change.
    Returns a 64-char hex string.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── index.md management ───────────────────────────────────────────────────────

def read_index() -> str:
    """Return the full text of wiki/index.md, or '' if it doesn't exist."""
    if not settings.index_path.exists():
        return ""
    return settings.index_path.read_text(encoding="utf-8")


def _index_counts() -> dict[str, int]:
    """Count pages per type for the index header."""
    return {t: len(list_pages(t)) for t in WIKI_SUBDIRS}  # type: ignore[arg-type]


def bootstrap_index() -> None:
    """
    Write an empty index.md if it doesn't exist yet.
    Called by ensure_dirs() on first run.
    """
    if settings.index_path.exists():
        return
    now = _now_iso()
    text = textwrap.dedent(f"""\
        ---
        last_updated: "{now}"
        source_count: 0
        page_count: 0
        ---

        # Wiki index

        ## Sources (0)
        | Date | Title | Tags | Page |
        |------|-------|------|------|

        ## Entities (0)
        | Name | Type | Page |
        |------|------|------|

        ## Concepts (0)
        | Name | Status | Tags | Page |
        |------|--------|------|------|

        ## Recent queries (0)
        | Date | Question | Page |
        |------|----------|------|
    """)
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_path.write_text(text, encoding="utf-8")


def add_index_row(page_type: PageType, row_data: dict[str, str]) -> None:
    """
    Append a row to the correct section table in index.md.

    row_data keys vary by page_type:
      source:  date, title, tags, slug
      entity:  name, entity_type, slug
      concept: name, status, tags, slug
      query:   date, question, slug
    """
    if not settings.index_path.exists():
        bootstrap_index()

    text = settings.index_path.read_text(encoding="utf-8")
    section_headers = {
        "source":  "## Sources",
        "entity":  "## Entities",
        "concept": "## Concepts",
        "query":   "## Recent queries",
    }

    def _make_row(pt: PageType, d: dict[str, str]) -> str:
        slug = d["slug"]
        if pt == "source":
            tags = d.get("tags", "")
            return f"| {d['date']} | {d['title']} | {tags} | [[sources/{slug}]] |"
        if pt == "entity":
            return f"| {d['name']} | {d['entity_type']} | [[entities/{slug}]] |"
        if pt == "concept":
            tags = d.get("tags", "")
            return f"| {d['name']} | {d.get('status', 'stub')} | {tags} | [[concepts/{slug}]] |"
        if pt == "query":
            return f"| {d['date']} | {d['question']} | [[queries/{slug}]] |"
        return ""

    new_row = _make_row(page_type, row_data)
    header = section_headers[page_type]

    # Find the section, locate the table, append after last row
    lines = text.splitlines()
    insert_at = None
    in_section = False
    in_table = False
    for i, line in enumerate(lines):
        if line.startswith(header):
            in_section = True
            continue
        if in_section and line.startswith("| "):
            in_table = True
            insert_at = i + 1      # keep moving to end of table
        elif in_section and in_table and not line.startswith("| "):
            break                  # past end of table

    if insert_at is None:
        # Section exists but no rows yet — insert after the header row
        for i, line in enumerate(lines):
            if line.startswith(header):
                # Skip header + separator row
                insert_at = i + 3
                break

    if insert_at is not None:
        lines.insert(insert_at, new_row)

    # Update frontmatter counts
    counts = _index_counts()
    new_text = "\n".join(lines)
    new_text = re.sub(
        r"source_count: \d+", f"source_count: {counts['source']}", new_text
    )
    page_total = sum(counts.values())
    new_text = re.sub(r"page_count: \d+", f"page_count: {page_total}", new_text)
    new_text = re.sub(
        r'last_updated: ".*?"',
        f'last_updated: "{_now_iso()}"',
        new_text,
    )
    settings.index_path.write_text(new_text, encoding="utf-8")


# ── log.md management ─────────────────────────────────────────────────────────

_LOG_OPERATIONS = {"ingest", "query", "lint", "update", "schema-change"}


def append_log(
    operation: str,
    description: str,
    detail: str = "",
    pages_touched: list[str] | None = None,
) -> None:
    """
    Prepend a new entry to wiki/log.md (newest-first).

    Args:
        operation:     One of the valid log operation strings.
        description:   Short title after the pipe in the header.
        detail:        1–3 sentences of context (optional).
        pages_touched: List of wiki slugs or [[wikilinks]] affected.
    """
    assert operation in _LOG_OPERATIONS, (
        f"Unknown log operation '{operation}'. "
        f"Valid: {_LOG_OPERATIONS}"
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    pages_line = ""
    if pages_touched:
        links = ", ".join(
            p if p.startswith("[[") else f"[[{p}]]" for p in pages_touched
        )
        pages_line = f"\nPages touched: {links}"

    entry = textwrap.dedent(f"""\
        ## [{now}] {operation} | {description}

        {detail.strip()}{pages_line}

    """)

    if not settings.log_path.exists():
        settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        settings.log_path.write_text(
            "# Wiki log\n\nNewest entries first.\n\n" + entry,
            encoding="utf-8",
        )
        return

    existing = settings.log_path.read_text(encoding="utf-8")
    # Insert after the first header block (title + subtitle line)
    header_end = existing.find("\n\n", existing.find("\n")) + 2
    settings.log_path.write_text(
        existing[:header_end] + entry + existing[header_end:],
        encoding="utf-8",
    )


# ── Wiki summary helpers (used by ingest / query context builders) ─────────────

def read_schema() -> str:
    """Return the full text of CLAUDE.md."""
    if not settings.schema_path.exists():
        return ""
    return settings.schema_path.read_text(encoding="utf-8")


def recent_log_entries(n: int | None = None) -> str:
    """
    Return the first n log entries from log.md as a string.
    Defaults to settings.log_context_entries.
    """
    if not settings.log_path.exists():
        return ""
    n = n or settings.log_context_entries
    text = settings.log_path.read_text(encoding="utf-8")
    # Each entry starts with '## ['
    entries = re.split(r"(?=^## \[)", text, flags=re.MULTILINE)
    # First element is the file header, not an entry
    header = entries[0] if entries else ""
    log_entries = entries[1:] if len(entries) > 1 else []
    return header + "".join(log_entries[:n])


def page_word_count(path: Path) -> int:
    """Return approximate word count of a wiki page body (excluding frontmatter)."""
    _, body = parse_page(path)
    return len(body.split())


def find_long_concept_pages() -> list[tuple[Path, int]]:
    """
    Return concept pages exceeding settings.concept_split_threshold words.
    Returns list of (path, word_count) tuples.
    """
    results = []
    for p in list_pages("concept"):
        wc = page_word_count(p)
        if wc > settings.concept_split_threshold:
            results.append((p, wc))
    return results


# ── Concept / entity existence checks ─────────────────────────────────────────

def page_exists(page_type: PageType, slug: str) -> bool:
    return page_path(page_type, slug).exists()


def get_or_create_stub(
    page_type: PageType,
    slug: str,
    title: str,
    extra_frontmatter: dict[str, Any] | None = None,
) -> tuple[Path, bool]:
    """
    Return the path for a page. If it doesn't exist, create a minimal stub.

    Returns (path, created: bool).
    The caller is responsible for filling in the stub with real content.
    """
    path = page_path(page_type, slug)
    if path.exists():
        return path, False

    fm: dict[str, Any] = {
        "type": page_type,
        "title": title,
        "tags": [],
    }
    if page_type == "concept":
        fm["status"] = "stub"
        fm["sources"] = []
        fm["related_concepts"] = []
    elif page_type == "entity":
        fm["entity_type"] = "other"
        fm["aliases"] = []
        fm["sources"] = []
    if extra_frontmatter:
        fm.update(extra_frontmatter)

    body = f"\n# {title}\n\n*Stub — to be filled in during next ingest or lint.*\n"
    write_page(path, fm, body)
    return path, True


# ── Utility ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
"""
provenance.py — SQLite-backed source provenance tracking.

Every time a source is ingested, we record:
  - which wiki pages it touched
  - which specific claims it contributed to each page
  - the SHA-256 hash of the source file at ingest time

On subsequent runs, we re-hash source files and compare. If a source has
changed on disk, every claim derived from it is flagged as potentially stale.

This is the key differentiator over vanilla LLM-wiki implementations.
The LLM compiles knowledge once. We track whether that compiled knowledge
is still valid.

Schema
------
  sources        — one row per ingested source file
  wiki_pages     — one row per wiki page (populated lazily on first touch)
  provenance     — many-to-many: source → wiki_page, with claims + hash
  stale_flags    — set when a source hash changes; cleared on re-ingest
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from config import settings
from wiki_manager import hash_file


# ── ORM models ─────────────────────────────────────────────────────────────────

class SourceRecord(SQLModel, table=True):
    """One row per raw source file."""
    __tablename__ = "sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_path: str = Field(index=True)           # relative to repo root
    current_hash: str                              # sha256 at last ingest
    title: str
    ingested_at: str                               # ISO datetime
    ingest_count: int = Field(default=1)           # how many times re-ingested


class WikiPageRecord(SQLModel, table=True):
    """One row per wiki page that has at least one provenance record."""
    __tablename__ = "wiki_pages"

    id: Optional[int] = Field(default=None, primary_key=True)
    page_path: str = Field(index=True, unique=True)  # relative to repo root
    page_type: str                                   # source | entity | concept | query
    slug: str


class ProvenanceRecord(SQLModel, table=True):
    """
    Links a source to a wiki page.
    claims_json is a JSON array of strings — the specific assertions
    this source contributed to this wiki page.
    """
    __tablename__ = "provenance"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="sources.id", index=True)
    wiki_page_id: int = Field(foreign_key="wiki_pages.id", index=True)
    source_hash_at_ingest: str          # sha256 when this record was written
    claims_json: str = Field(default="[]")  # JSON list[str]
    recorded_at: str                    # ISO datetime


class StaleFlagRecord(SQLModel, table=True):
    """
    Set when a source file's hash no longer matches the stored hash.
    Cleared (deleted) when the source is re-ingested.
    """
    __tablename__ = "stale_flags"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="sources.id", index=True)
    wiki_page_id: int = Field(foreign_key="wiki_pages.id", index=True)
    detected_at: str                    # ISO datetime
    old_hash: str
    new_hash: str
    resolved: bool = Field(default=False)


# ── Python dataclasses (returned to callers, no ORM coupling) ─────────────────

@dataclass
class IngestRecord:
    """What to write to provenance DB after a successful ingest."""
    source_path: str               # e.g. "raw/papers/attention.pdf"
    title: str
    wiki_pages_touched: list[str]  # relative paths, e.g. ["wiki/sources/...md"]
    claims_per_page: dict[str, list[str]] = field(default_factory=dict)
    # ^ page_path → list of claim strings from this source


@dataclass
class StaleWarning:
    """Returned by check_staleness() for each outdated provenance record."""
    source_path: str
    source_title: str
    wiki_page_path: str
    claims: list[str]
    old_hash: str
    new_hash: str

    def __str__(self) -> str:
        n = len(self.claims)
        claim_word = "claim" if n == 1 else "claims"
        return (
            f"[STALE] {self.wiki_page_path}\n"
            f"  Source : {self.source_path}\n"
            f"  Affects: {n} {claim_word}\n"
            f"  Hash   : {self.old_hash[:12]}… → {self.new_hash[:12]}…"
        )


# ── Engine + session factory ───────────────────────────────────────────────────

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        db_url = f"sqlite:///{settings.provenance_db_path}"
        _engine = create_engine(db_url, echo=False)
        SQLModel.metadata.create_all(_engine)
    return _engine


def _session() -> Session:
    return Session(get_engine())


# ── Public API ─────────────────────────────────────────────────────────────────

def record_ingest(record: IngestRecord) -> None:
    """
    Write provenance records after a successful ingest.

    - Upserts the SourceRecord (creates or updates hash + count).
    - Upserts WikiPageRecords for every touched page.
    - Writes ProvenanceRecord for each (source, page) pair.
    - Clears any existing stale flags for this source (re-ingest resolves them).
    """
    now = _now_iso()
    source_hash = hash_file(
        settings.provenance_db_path.parent / record.source_path
    )

    with _session() as session:
        # ── Upsert source ──────────────────────────────────────────────────
        src = session.exec(
            select(SourceRecord).where(
                SourceRecord.source_path == record.source_path
            )
        ).first()

        if src is None:
            src = SourceRecord(
                source_path=record.source_path,
                current_hash=source_hash,
                title=record.title,
                ingested_at=now,
                ingest_count=1,
            )
            session.add(src)
        else:
            src.current_hash = source_hash
            src.ingested_at = now
            src.ingest_count += 1
            session.add(src)

        session.commit()
        session.refresh(src)

        # ── Resolve stale flags for this source ───────────────────────────
        stale_records = session.exec(
            select(StaleFlagRecord).where(
                StaleFlagRecord.source_id == src.id,
                StaleFlagRecord.resolved == False,  # noqa: E712
            )
        ).all()
        for flag in stale_records:
            flag.resolved = True
            session.add(flag)
        session.commit()

        # ── Upsert wiki pages + write provenance ──────────────────────────
        for page_path_str in record.wiki_pages_touched:
            # Upsert wiki page row
            wp = session.exec(
                select(WikiPageRecord).where(
                    WikiPageRecord.page_path == page_path_str
                )
            ).first()

            if wp is None:
                page_path_obj = Path(page_path_str)
                slug = page_path_obj.stem
                # Infer type from directory name
                page_type = _infer_page_type(page_path_str)
                wp = WikiPageRecord(
                    page_path=page_path_str,
                    page_type=page_type,
                    slug=slug,
                )
                session.add(wp)
                session.commit()
                session.refresh(wp)

            # Write provenance record
            claims = record.claims_per_page.get(page_path_str, [])
            prov = ProvenanceRecord(
                source_id=src.id,
                wiki_page_id=wp.id,
                source_hash_at_ingest=source_hash,
                claims_json=json.dumps(claims, ensure_ascii=False),
                recorded_at=now,
            )
            session.add(prov)

        session.commit()


def check_staleness() -> list[StaleWarning]:
    """
    Re-hash every tracked source file and compare against stored hashes.

    Returns a list of StaleWarning for every (source, wiki_page) pair
    where the source file has changed since ingest.

    Also writes StaleFlagRecords to the DB for new detections.
    """
    warnings: list[StaleWarning] = []
    now = _now_iso()

    with _session() as session:
        sources = session.exec(select(SourceRecord)).all()

        for src in sources:
            src_path = settings.provenance_db_path.parent / src.source_path
            if not src_path.exists():
                # Source file deleted — flag all its pages
                _flag_deleted_source(session, src, now)
                continue

            current_hash = hash_file(src_path)
            if current_hash == src.current_hash:
                continue  # unchanged, skip

            # Hash mismatch — find all wiki pages this source contributed to
            provenance_rows = session.exec(
                select(ProvenanceRecord).where(
                    ProvenanceRecord.source_id == src.id
                )
            ).all()

            for prov in provenance_rows:
                wp = session.get(WikiPageRecord, prov.wiki_page_id)
                if wp is None:
                    continue

                claims = json.loads(prov.claims_json)

                warnings.append(StaleWarning(
                    source_path=src.source_path,
                    source_title=src.title,
                    wiki_page_path=wp.page_path,
                    claims=claims,
                    old_hash=prov.source_hash_at_ingest,
                    new_hash=current_hash,
                ))

                # Write stale flag if not already flagged
                existing_flag = session.exec(
                    select(StaleFlagRecord).where(
                        StaleFlagRecord.source_id == src.id,
                        StaleFlagRecord.wiki_page_id == wp.id,
                        StaleFlagRecord.resolved == False,  # noqa: E712
                    )
                ).first()

                if existing_flag is None:
                    session.add(StaleFlagRecord(
                        source_id=src.id,
                        wiki_page_id=wp.id,
                        detected_at=now,
                        old_hash=prov.source_hash_at_ingest,
                        new_hash=current_hash,
                        resolved=False,
                    ))

        session.commit()

    return warnings


def get_page_provenance(page_path_str: str) -> list[dict]:
    """
    Return all provenance records for a given wiki page.

    Each dict has: source_path, source_title, claims, ingested_at,
                   source_hash_at_ingest, is_stale (bool)
    """
    results = []
    with _session() as session:
        wp = session.exec(
            select(WikiPageRecord).where(
                WikiPageRecord.page_path == page_path_str
            )
        ).first()
        if wp is None:
            return []

        provenance_rows = session.exec(
            select(ProvenanceRecord).where(
                ProvenanceRecord.wiki_page_id == wp.id
            )
        ).all()

        for prov in provenance_rows:
            src = session.get(SourceRecord, prov.source_id)
            if src is None:
                continue

            # Check staleness
            is_stale = (src.current_hash != prov.source_hash_at_ingest)

            results.append({
                "source_path": src.source_path,
                "source_title": src.title,
                "claims": json.loads(prov.claims_json),
                "ingested_at": prov.recorded_at,
                "source_hash_at_ingest": prov.source_hash_at_ingest,
                "is_stale": is_stale,
            })

    return results


def get_source_stats() -> list[dict]:
    """
    Summary statistics per source: title, path, ingest count,
    pages touched, stale flag count.
    """
    stats = []
    with _session() as session:
        sources = session.exec(select(SourceRecord)).all()
        for src in sources:
            page_count = len(session.exec(
                select(ProvenanceRecord).where(
                    ProvenanceRecord.source_id == src.id
                )
            ).all())

            stale_count = len(session.exec(
                select(StaleFlagRecord).where(
                    StaleFlagRecord.source_id == src.id,
                    StaleFlagRecord.resolved == False,  # noqa: E712
                )
            ).all())

            stats.append({
                "title": src.title,
                "source_path": src.source_path,
                "ingested_at": src.ingested_at,
                "ingest_count": src.ingest_count,
                "pages_touched": page_count,
                "stale_flags": stale_count,
                "current_hash": src.current_hash[:12] + "…",
            })

    return stats


def get_stale_summary() -> dict:
    """
    Return a high-level staleness report dict.
    Used by lint.py to surface issues.
    """
    with _session() as session:
        total_sources = len(session.exec(select(SourceRecord)).all())
        open_flags = session.exec(
            select(StaleFlagRecord).where(
                StaleFlagRecord.resolved == False  # noqa: E712
            )
        ).all()

        stale_source_ids = {f.source_id for f in open_flags}
        stale_page_ids = {f.wiki_page_id for f in open_flags}

        return {
            "total_sources": total_sources,
            "stale_sources": len(stale_source_ids),
            "stale_wiki_pages": len(stale_page_ids),
            "open_flags": len(open_flags),
        }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _infer_page_type(page_path_str: str) -> str:
    """Infer page type from directory name in the path."""
    p = page_path_str.lower()
    if "/sources/" in p:
        return "source"
    if "/entities/" in p:
        return "entity"
    if "/concepts/" in p:
        return "concept"
    if "/queries/" in p:
        return "query"
    return "unknown"


def _flag_deleted_source(
    session: Session, src: SourceRecord, now: str
) -> None:
    """
    Flag all wiki pages from a deleted source as stale.
    Called when check_staleness() finds a source file no longer on disk.
    """
    provenance_rows = session.exec(
        select(ProvenanceRecord).where(
            ProvenanceRecord.source_id == src.id
        )
    ).all()

    for prov in provenance_rows:
        existing = session.exec(
            select(StaleFlagRecord).where(
                StaleFlagRecord.source_id == src.id,
                StaleFlagRecord.wiki_page_id == prov.wiki_page_id,
                StaleFlagRecord.resolved == False,  # noqa: E712
            )
        ).first()
        if existing is None:
            session.add(StaleFlagRecord(
                source_id=src.id,
                wiki_page_id=prov.wiki_page_id,
                detected_at=now,
                old_hash=prov.source_hash_at_ingest,
                new_hash="[deleted]",
                resolved=False,
            ))
"""
config.py — centralized settings for llm-wiki.

All paths, model parameters, and environment variables live here.
Every other module imports from this file — nothing hardcodes paths or keys.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


# ── Repo root ──────────────────────────────────────────────────────────────────
# config.py lives at src/llm_wiki/config.py — walk up two levels to repo root.
# This makes all paths and .env resolution work regardless of cwd.
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


# ── Settings (reads from .env, then environment, then defaults) ────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),   # absolute path — works from any cwd
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # Model used for all wiki operations.
    # GPT-4o is the default — high context window matters for multi-page reads.
    llm_model: str = Field("gpt-4o", alias="LLM_MODEL")

    # Temperature: low for ingest/lint (factual, precise), higher for query
    # (synthesis benefits from a little more creativity). Override per-call.
    ingest_temperature: float = Field(0.2, alias="INGEST_TEMPERATURE")
    query_temperature: float = Field(0.4, alias="QUERY_TEMPERATURE")
    lint_temperature: float = Field(0.1, alias="LINT_TEMPERATURE")

    # Max tokens per LLM call. 4096 is safe for most operations.
    # Ingest may need more when touching many pages in one pass.
    max_tokens: int = Field(4096, alias="MAX_TOKENS")

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Relative to REPO_ROOT. Override in .env if your layout differs.
    raw_dir: Path = Field(Path("raw"), alias="RAW_DIR")
    wiki_dir: Path = Field(Path("wiki"), alias="WIKI_DIR")
    provenance_db: Path = Field(Path("provenance.db"), alias="PROVENANCE_DB")

    # ── Wiki behaviour ────────────────────────────────────────────────────────
    # How many recent log entries to include in the LLM context on each session.
    log_context_entries: int = Field(10, alias="LOG_CONTEXT_ENTRIES")

    # Lint: trigger automatically after this many ingests (0 = never auto-lint).
    auto_lint_every: int = Field(10, alias="AUTO_LINT_EVERY")

    # Concept page word count before we suggest splitting it.
    concept_split_threshold: int = Field(600, alias="CONCEPT_SPLIT_THRESHOLD")

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(True, alias="API_RELOAD")    # set False in prod

    # ── Streamlit ─────────────────────────────────────────────────────────────
    streamlit_page_title: str = Field("llm-wiki", alias="STREAMLIT_PAGE_TITLE")

    # ── Computed fields (derived, not user-configurable) ──────────────────────
    @computed_field
    @property
    def raw_path(self) -> Path:
        return REPO_ROOT / self.raw_dir

    @computed_field
    @property
    def wiki_path(self) -> Path:
        return REPO_ROOT / self.wiki_dir

    @computed_field
    @property
    def provenance_db_path(self) -> Path:
        return REPO_ROOT / self.provenance_db

    @computed_field
    @property
    def sources_path(self) -> Path:
        return self.wiki_path / "sources"

    @computed_field
    @property
    def entities_path(self) -> Path:
        return self.wiki_path / "entities"

    @computed_field
    @property
    def concepts_path(self) -> Path:
        return self.wiki_path / "concepts"

    @computed_field
    @property
    def queries_path(self) -> Path:
        return self.wiki_path / "queries"

    @computed_field
    @property
    def index_path(self) -> Path:
        return self.wiki_path / "index.md"

    @computed_field
    @property
    def log_path(self) -> Path:
        return self.wiki_path / "log.md"

    @computed_field
    @property
    def overview_path(self) -> Path:
        return self.wiki_path / "overview.md"

    @computed_field
    @property
    def schema_path(self) -> Path:
        return REPO_ROOT / "CLAUDE.md"


# ── Wiki subdirectory names (used by wiki_manager.py) ─────────────────────────
# Single source of truth for directory names — don't hardcode these elsewhere.

PageType = Literal["source", "entity", "concept", "query"]

WIKI_SUBDIRS: dict[PageType, str] = {
    "source": "sources",
    "entity": "entities",
    "concept": "concepts",
    "query": "queries",
}

# Raw subdirectories (human-owned, never written by the LLM)
RAW_SUBDIRS = ["papers", "articles", "transcripts", "notes", "assets"]

# Supported source file extensions for ingest
INGESTABLE_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".docx"}


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import `settings` everywhere — don't instantiate Settings() per-module.

settings = Settings()


# ── Directory bootstrap ────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """
    Create all required wiki and raw directories if they don't exist.
    Call once at app startup (api.py and app.py both call this).
    Safe to call repeatedly — uses exist_ok=True.
    """
    for subdir in WIKI_SUBDIRS.values():
        (settings.wiki_path / subdir).mkdir(parents=True, exist_ok=True)

    for subdir in RAW_SUBDIRS:
        (settings.raw_path / subdir).mkdir(parents=True, exist_ok=True)


# ── Slug utilities ─────────────────────────────────────────────────────────────

import re

def slugify(text: str) -> str:
    """
    Convert a title or name to a filesystem-safe slug.
    'Retrieval-Augmented Generation' → 'retrieval-augmented-generation'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)       # remove punctuation
    text = re.sub(r"[\s_]+", "-", text)         # spaces/underscores → hyphens
    text = re.sub(r"-+", "-", text)             # collapse multiple hyphens
    return text.strip("-")


def source_slug(date: str, title: str) -> str:
    """
    Build a source page filename stem.
    '2026-04-06', 'Attention Is All You Need' → '2026-04-06_attention-is-all-you-need'
    """
    return f"{date}_{slugify(title)}"
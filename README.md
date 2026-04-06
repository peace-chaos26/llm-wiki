# llm-wiki

> A persistent, LLM-compiled knowledge base with provenance tracking and evaluation harness.

Instead of RAG — which re-derives knowledge from raw documents on every query — llm-wiki **compiles** sources into a structured, interlinked markdown wiki that grows richer with every ingestion. Knowledge accumulates. Cross-references are built once, not re-discovered each time.

Built on an idea by Andrej Karpathy ([original gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)).

---

## The core difference from RAG

| | Standard RAG | llm-wiki |
|---|---|---|
| Knowledge store | Raw documents | Compiled wiki pages |
| Query time | Re-derive from scratch | Read compiled synthesis |
| Cross-references | Re-discovered per query | Built once, always available |
| Source changes | Silent drift | SHA-256 staleness detection |
| Evaluation | Retrieval metrics | Wiki quality + faithfulness |

**Benchmark result** (5 questions, BM25 RAG baseline):

| Question type | Wiki score | RAG score | Δ |
|---|---|---|---|
| Synthesis | 7.0 / 10 | 4.5 / 10 | **+2.5** |
| Lookup | 7.0 / 10 | 10.0 / 10 | -3.0 |

Wiki outperforms BM25 RAG by +2.5 points on synthesis questions requiring multi-source reasoning. RAG wins on simple factual lookups — expected, and acceptable. Citation accuracy: 100% across all queries.

---

---

## What I built on top of Karpathy's idea

Karpathy's gist describes the *pattern* — three layers, three operations, the insight that LLMs handle wiki maintenance cheaply. It is intentionally abstract: "The exact directory structure, the schema conventions, the page formats, the tooling — all of that will depend on your domain."

This repo is a full implementation of that pattern with four additions that address gaps the original idea leaves open.

---

**1. Source provenance tracking**

The gist has no answer for this question: *what happens when a source file changes?* The wiki was compiled from that source — but now the compiled knowledge may be wrong, and nothing knows it.

I built a SQLite provenance layer (`provenance.py`) that records the SHA-256 hash of every source file at ingest time, and tracks which specific claims each source contributed to each wiki page. When `check_staleness()` runs, it re-hashes every source file and flags any wiki page whose source has drifted. Re-ingest to resolve. The wiki never silently serves stale knowledge.

This is the most technically differentiated piece. Most llm-wiki implementations skip it entirely.

---

**2. Multi-pass ingest pipeline**

The gist describes ingest as a single operation: the LLM reads the source, writes a summary page, updates entity and concept pages. One prompt doing all of this produces inconsistent output — conflated responsibilities, missed entity pages, malformed frontmatter.

I decomposed ingest into four focused passes (`ingest.py`):

- Pass 1: structured JSON extraction — title, entities, concepts, claims, contradictions
- Pass 2: source wiki page (strict template)
- Pass 3: entity pages — one LLM call per entity, update-aware
- Pass 4: concept pages — synthesis-focused, contradiction-aware

Each pass has a single responsibility. Pass 1 is non-streaming JSON that drives all subsequent passes. Passes 2–4 stream in real time. The result is more consistent, debuggable, and independently testable — mock any pass without touching the others.

---

**3. Structured evaluation harness**

The gist describes lint as a health-check but gives no metrics. There is no way to know whether a wiki is *good* — well-synthesised, well-connected, current — versus just large.

I built two evaluation modules (`eval/`):

- **Wiki quality eval** — six structural metrics (entity coverage, cross-link density, orphan rate, stub rate, source freshness, concept depth) combined into a 0–100 composite score.
- **Query eval** — LLM-as-judge comparison of wiki answers versus a BM25 RAG baseline over the same raw sources, broken down by question type (lookup vs synthesis). This produces the benchmark result above and gives a principled answer to the question "is the compiled wiki actually better than just using RAG?"

Without this, you can't claim the wiki is better — you can only hope it is. The eval harness makes the claim measurable.

---

**4. Production API and UI layer**

The gist assumes Claude Code or a similar agentic coding environment as the interface. All operations are manual — you instruct the LLM in a chat window.

I wrapped the three operations in a FastAPI server with Server-Sent Events streaming (`api.py`) and a Streamlit frontend (`app.py`). Every ingest, query, and lint operation streams progress in real time. The API is fully documented via auto-generated Swagger UI at `/docs`. This makes the system usable without a coding environment and demonstrable without a terminal.

---

These four additions address the production gaps in the original idea: *correctness over time* (provenance), *consistency at scale* (multi-pass pipeline), *measurability* (eval harness), and *usability* (API + UI). The core insight — that LLMs should compile knowledge rather than re-derive it on every query — is Karpathy's. The engineering to make that insight reliable and measurable is the work here.


## Architecture

Three layers:

```
raw/                    ← immutable source documents (human-owned)
  papers/
  articles/
  transcripts/

wiki/                   ← LLM-compiled knowledge (LLM-owned)
  index.md              ← master catalog, rebuilt after every ingest
  log.md                ← append-only operations log
  sources/              ← one page per ingested source
  entities/             ← people, orgs, models, systems
  concepts/             ← ideas, methods, findings — the synthesis layer
  queries/              ← valuable answers filed back as wiki pages

CLAUDE.md               ← schema: wiki conventions + LLM instructions
provenance.db           ← SQLite: source hashes, claims, staleness flags
```

Three operations:

- **Ingest** — read a source, extract entities and concepts, compile wiki pages, record provenance
- **Query** — scan index, fetch relevant pages, synthesise answer with inline citations
- **Lint** — health-check the wiki: orphans, stale provenance, contradictions, index drift

---

## Key features

**Source provenance tracking.** Every wiki page records which source files contributed which claims, along with the SHA-256 hash of each source at ingest time. When a source file changes on disk, `check_staleness()` re-hashes and flags every affected wiki page automatically. Re-ingest to resolve.

**4-pass ingest pipeline.** Each ingest runs four focused LLM calls: structured extraction (JSON) → source page → entity pages → concept pages. Smaller, single-responsibility prompts produce more consistent output than one large prompt. All passes stream in real time.

**Compounding wiki.** Valuable query answers are filed back as `wiki/queries/` pages. The wiki grows not just from sources but from the questions asked of it.

**6-check lint.** Orphan detection, missing stub creation, stale provenance surfacing, LLM contradiction detection, long-page flagging, and index drift repair — all idempotent, safe to run repeatedly.

**Structured evaluation.** Wiki quality score (0–100) across entity coverage, cross-link density, orphan rate, stub rate, source freshness, and concept depth. Query quality via LLM-as-judge with citation accuracy and faithfulness checks.

---

## Stack

| Component | Tool |
|---|---|
| LLM | GPT-4o (configurable) |
| Wiki storage | Markdown files + git |
| Provenance DB | SQLite via SQLModel |
| Backend | FastAPI + SSE streaming |
| Frontend | Streamlit |
| RAG baseline | BM25 (rank-bm25) |
| Evaluation | LLM-as-judge + custom metrics |

---

## Getting started

**Prerequisites:** Python 3.11+, OpenAI API key

```bash
git clone https://github.com/peace-chaos26/llm-wiki.git
cd llm-wiki

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

**Start the API server:**
```bash
python3 -m uvicorn api:app --reload --port 8000
```

**Start the UI:**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**Or use the API directly:**
```bash
# Ingest a source
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "raw/articles/my-paper.txt"}'

# Query the wiki
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'

# Run health checks
curl -X POST http://localhost:8000/lint -d '{}'
```

**Interactive API docs:** `http://localhost:8000/docs`

---

## Running tests

```bash
cd src/llm_wiki

python3 test_ingest.py   # 4-pass ingest pipeline (mocked LLM)
python3 test_query.py    # query pipeline (mocked LLM)
python3 test_lint.py     # 6-check lint pipeline (mocked LLM)

cd ../..
python3 test_api.py      # FastAPI endpoints (mocked pipelines)
```

All tests run without an API key — LLM calls are mocked at the function boundary.

---

## Running evals

```bash
# Wiki structure quality (no API key needed)
python3 eval/eval_wiki_quality.py

# Query quality + RAG vs Wiki comparison (needs API key + ingested sources)
python3 eval/eval_query.py

# Save results as JSON
python3 eval/eval_query.py --output eval/results/run_$(date +%Y%m%d).json
```

---

## Project structure

```
llm-wiki/
├── src/llm_wiki/
│   ├── config.py           # settings, paths, slug utilities
│   ├── wiki_manager.py     # all filesystem I/O for the wiki
│   ├── provenance.py       # SQLite provenance + staleness detection
│   ├── ingest.py           # 4-pass LLM ingest pipeline
│   ├── query.py            # 2-pass query + optional file-back
│   ├── lint.py             # 6-check wiki health pipeline
│   ├── test_ingest.py
│   ├── test_query.py
│   └── test_lint.py
├── eval/
│   ├── eval_wiki_quality.py  # wiki structure metrics
│   ├── eval_query.py         # RAG vs Wiki comparison
│   └── test_cases/
│       └── rag_questions.json
├── api.py                  # FastAPI server (SSE streaming)
├── app.py                  # Streamlit UI
├── test_api.py
├── CLAUDE.md               # wiki schema + LLM instructions
├── requirements.txt
└── .env.example
```

---

## Roadmap

- [ ] Hybrid BM25 + dense search over wiki pages at scale (qmd / local vector DB)
- [ ] auto-synthesis — LLM maintains a high-level wiki summary
- [ ] Multi-source ingest — batch ingest a directory
- [ ] Expand eval benchmark — 20+ questions across 3+ topic domains

---

## Motivation

Most RAG systems re-discover knowledge from scratch on every query. For questions requiring synthesis across multiple sources — "how do these three papers differ on X?" — this is fundamentally wasteful. The wiki pre-compiles the synthesis. Cross-references, contradictions, entity relationships — these are built once and available on every subsequent query.

The maintenance problem that makes human-built wikis fail (bookkeeping, cross-referencing, keeping summaries current) is exactly what LLMs handle well. The human curates sources and asks questions. The LLM does everything else.

---

## License

MIT
# CLAUDE.md — Wiki Schema

> This file is the operating contract between you (the LLM) and this wiki.
> Read it fully at the start of every session. Follow it precisely.
> You and the human will co-evolve this file as the wiki grows.

---

## What this wiki is

This is a **persistent, compounding knowledge base**. You do not answer questions
from raw sources — you compile raw sources into structured wiki pages, maintain
those pages as new sources arrive, and answer questions by reasoning over the
compiled wiki.

The wiki is a git repository of markdown files. You own the `wiki/` directory
entirely. The human owns `raw/`. You never modify files in `raw/`.

The human reads. You write. The human curates sources and asks questions.
You do the summarising, cross-referencing, filing, and maintenance.

---

## Directory layout

```
llm-wiki/
├── raw/                    # Immutable source documents (human-owned)
│   ├── papers/
│   ├── articles/
│   ├── transcripts/
│   ├── notes/
│   └── assets/             # Images, PDFs, attachments
├── wiki/                   # LLM-owned compiled knowledge
│   ├── index.md            # Master catalog — update on every ingest
│   ├── log.md              # Append-only operations log
│   ├── overview.md         # High-level synthesis of the whole wiki
│   ├── sources/            # One page per ingested source
│   ├── entities/           # Named things: people, orgs, systems, models
│   ├── concepts/           # Ideas, methods, frameworks, findings
│   └── queries/            # Valuable query answers filed back as pages
├── CLAUDE.md               # This file — the schema
└── provenance.db           # SQLite provenance store (managed by ingest.py)
```

---

## Page types

There are four page types. Sources and index are **strictly templated**.
Entities and concepts are **flexibly structured** — use whatever headings
serve the content, but always include the required frontmatter.

---

### 1. Source page (strict template)

One page per ingested source. Lives in `wiki/sources/`.
Filename: `{YYYY-MM-DD}_{slugified-title}.md`

```markdown
---
type: source
title: "<full title>"
source_path: "raw/papers/filename.pdf"
source_hash: "<sha256 — filled by ingest.py>"
ingested_at: "<ISO datetime>"
tags: [<2–5 domain tags>]
key_entities: [<entity slugs this source mentions>]
key_concepts: [<concept slugs this source touches>]
---

# <Title>

## Bibliographic info
- **Authors / Origin**: ...
- **Published**: ...
- **Type**: paper | article | transcript | note | dataset
- **URL / DOI**: ...

## Summary
2–4 sentences. What this source argues or reports, and why it matters
to the wiki. Write for someone who will never read the source itself.

## Key claims
Bullet list of the most important, citable assertions from this source.
Be specific. Quote sparingly — paraphrase most things.

- Claim 1
- Claim 2
- ...

## Entities mentioned
Links to entity pages this source discusses. Create pages that don't exist yet.
- [[entities/person-name]]
- [[entities/org-name]]

## Concepts touched
Links to concept pages. Create or update concept pages after filing this.
- [[concepts/concept-name]]

## Connections
What does this source agree with, contradict, or extend in the existing wiki?
Cross-link explicitly. If it contradicts another source, say so here and
also update the relevant concept page to note the tension.

## Questions raised
What does this source leave open? What should we investigate next?
```

---

### 2. Entity page (flexible, required frontmatter)

One page per named entity: a person, organisation, model, system, dataset,
product. Lives in `wiki/entities/`. Filename: `{slug}.md`

```markdown
---
type: entity
entity_type: person | org | model | system | dataset | product | other
title: "<canonical name>"
aliases: [<alternate names, abbreviations>]
tags: [<domain tags>]
sources: [<source slugs that mention this entity>]
---

# <Name>

<!-- After this, use whatever headings serve the content. -->
<!-- Suggested sections (use what's relevant, skip what isn't): -->
<!-- ## Overview -->
<!-- ## Background / history -->
<!-- ## Key contributions / what they built -->
<!-- ## Relationships (links to related entities and concepts) -->
<!-- ## Appearances in this wiki (sources that discuss them) -->
<!-- ## Open questions -->
```

**Rules for entity pages:**
- Keep entity pages factual. Analysis and interpretation belong in concept pages.
- Use `[[wikilinks]]` for every internal reference.
- Never duplicate content from source pages — summarise and link back.
- If the same entity appears under multiple names, pick a canonical name and
  add aliases to frontmatter.

---

### 3. Concept page (flexible, required frontmatter)

One page per idea, method, framework, finding, or theme. Lives in
`wiki/concepts/`. Filename: `{slug}.md`

```markdown
---
type: concept
title: "<concept name>"
aliases: [<related terms>]
tags: [<domain tags>]
sources: [<source slugs that inform this concept>]
related_concepts: [<concept slugs>]
status: stub | developing | mature
---

# <Concept name>

<!-- After this, use whatever headings serve the content. -->
<!-- Concepts should synthesise across sources — this is where the -->
<!-- real value of the wiki lives. Don't just summarise one source; -->
<!-- connect multiple sources, surface tensions, build up a thesis. -->
```

**Rules for concept pages:**
- `status: stub` = placeholder, < 3 sources. `developing` = active synthesis.
  `mature` = well-sourced, stable, reviewed in lint.
- When two sources disagree, note the tension explicitly under a
  `## Tensions and open questions` section.
- If a concept page grows past ~500 words, consider splitting it.
- Always link back to source pages for every major claim.

---

### 4. Query page (flexible)

Filed in `wiki/queries/` when a query answer is valuable enough to persist.
Filename: `{YYYY-MM-DD}_{short-description}.md`

```markdown
---
type: query
title: "<the question asked>"
asked_at: "<ISO datetime>"
tags: [<domain tags>]
sources_used: [<wiki page slugs consulted>]
---

# <Question>

<Answer — written as a standalone piece, not a chat reply.>

## Sources consulted
- [[wiki/sources/...]]
- [[wiki/concepts/...]]
```

---

## index.md (strict template)

`wiki/index.md` is your navigation hub. Update it on every ingest.
Never delete entries — mark stale ones with `[stale]`.

```markdown
---
last_updated: "<ISO datetime>"
source_count: <n>
page_count: <n>
---

# Wiki index

## Sources (<n>)
| Date | Title | Tags | Page |
|------|-------|------|------|
| YYYY-MM-DD | Title | tag1, tag2 | [[sources/slug]] |

## Entities (<n>)
| Name | Type | Page |
|------|------|------|
| Name | person/org/model/... | [[entities/slug]] |

## Concepts (<n>)
| Name | Status | Tags | Page |
|------|--------|------|------|
| Name | stub/developing/mature | tag1 | [[concepts/slug]] |

## Recent queries (<last 10>)
| Date | Question | Page |
|------|----------|------|
| YYYY-MM-DD | Question text | [[queries/slug]] |
```

---

## log.md (strict template)

`wiki/log.md` is append-only. Never edit existing entries.
Every operation appends one entry at the top (newest first).

Entry format:
```
## [YYYY-MM-DD HH:MM] {operation} | {short description}

{1–3 sentences about what was done, what changed, what was notable.}

Pages touched: [[page1]], [[page2]], ...
```

Operations: `ingest` | `query` | `lint` | `update` | `schema-change`

---

## Operations

### INGEST

Triggered when the human drops a new file into `raw/` and asks you to process it.

**Your job:** Read the source. Extract key claims. Integrate them into the wiki —
not just index them, but compile them. Update existing pages where relevant.
File contradictions. Strengthen existing syntheses.

**Steps — follow in order:**

1. **Read** the source file (or ask the human to paste it if you can't access it directly).
2. **Discuss** with the human: what are the 3–5 most important takeaways? Any surprises?
3. **Create** `wiki/sources/{date}_{slug}.md` using the strict source template.
4. **Update or create** entity pages for every significant named entity mentioned.
5. **Update or create** concept pages for every significant idea. This is the most
   important step — a single source might update 5–10 concept pages. For each:
   - Add new claims from this source
   - Note if this source contradicts prior claims (link both ways)
   - Update `status` if the page has matured
6. **Update** `wiki/index.md` — add the source row, update entity and concept counts.
7. **Append** to `wiki/log.md`.
8. **Update** `wiki/overview.md` if the overall thesis of the wiki has shifted.

**Provenance note:** `ingest.py` will automatically compute the source hash and
write provenance records to `provenance.db`. You do not manage the database —
just make sure the `source_path` in the source page frontmatter is exact.

**Scope:** A single ingest may touch 10–15 wiki pages. That is normal.
Never rush it. One source, done properly, beats five sources done shallowly.

---

### QUERY

Triggered when the human asks a question.

**Your job:** Answer from the compiled wiki. Do not re-read raw sources.
If the wiki doesn't contain enough to answer well, say so and suggest what
to ingest next.

**Steps:**

1. **Read** `wiki/index.md` to identify relevant sources, entities, concepts.
2. **Read** the relevant wiki pages (not raw sources).
3. **Synthesise** an answer with inline citations to wiki pages: `[[concepts/slug]]`.
4. **Assess** whether this answer is valuable enough to file back as a query page.
   If yes, create `wiki/queries/{date}_{slug}.md`.
5. **Append** to `wiki/log.md`.

**Citation format:** Every factual claim in your answer must cite the wiki page
it came from: `(→ [[concepts/rag-retrieval]])`. Never cite raw sources directly.

**When the wiki is insufficient:** Say: "The wiki doesn't have enough on X yet.
I'd suggest ingesting [specific source type] to answer this well."

---

### LINT

Triggered periodically (suggest after every 10 ingests, or when the human asks).

**Your job:** Health-check the wiki. Find and fix structural problems.
Do not rewrite content — only add cross-links, stub pages, and flags.

**Check for:**

1. **Orphan pages** — pages with 0 inbound `[[wikilinks]]`. Flag them. Ask the
   human whether to link them or delete them.
2. **Missing entity pages** — entities mentioned in multiple sources but lacking
   their own page. Create stubs.
3. **Missing concept pages** — concepts mentioned in source pages but not in
   `wiki/concepts/`. Create stubs.
4. **Stale provenance** — `ingest.py` flags these automatically in `provenance.db`.
   Surface them here: "These pages have claims from sources that have changed."
5. **Contradictions** — scan concept pages for claims that conflict across sources.
   Add a `## Tensions` section where missing.
6. **Over-long pages** — concept pages > 600 words. Suggest splits.
7. **Index drift** — pages that exist on disk but are missing from `index.md`.

**Output format:** A lint report filed as `wiki/queries/{date}_lint-report.md`.
List every finding with its severity (🔴 critical / 🟡 warning / 🟢 suggestion)
and the action taken or recommended.

---

## Wikilink conventions

- Internal links always use `[[relative/path]]` format without `.md` extension.
- Never use absolute paths.
- When creating a new page, search `index.md` first — the page may already exist
  under a different name. Use the alias field to consolidate duplicates.
- Backlinks are not automatic — if page A links to page B, also add a link from
  B back to A under a `## Appears in` or `## Related` section.

---

## Frontmatter rules

- All frontmatter is YAML, enclosed in `---`.
- `type` is required on every page: `source` | `entity` | `concept` | `query`.
- `tags` are lowercase, hyphenated: `machine-learning`, `large-language-models`.
- `source_path` in source pages must match the exact relative path from repo root.
- Dates are ISO 8601: `2026-04-06T14:32:00`.

---

## What you never do

- **Never modify files in `raw/`.**
- **Never delete wiki pages** — mark them `status: deprecated` and explain why.
- **Never answer questions from raw sources directly** — compile first, answer from wiki.
- **Never copy-paste large blocks from sources** — summarise and link.
- **Never create wiki pages without updating `index.md`.**
- **Never skip the log entry** — every operation must be recorded.

---

## Tone and style

- Write wiki pages as if for a knowledgeable reader who has not read the sources.
- Prefer synthesis over summary. The wiki should say something, not just report.
- Be precise about uncertainty: "Source A claims X, but Source B's data suggests Y."
- Use present tense for claims, past tense for events.
- Keep prose tight. Long paragraphs should be split or bulleted.

---

## Schema evolution

This file will change as the wiki grows. When you and the human agree to change
a convention, update this file and add an entry to `log.md`:

```
## [YYYY-MM-DD HH:MM] schema-change | <what changed and why>
```

All prior pages remain valid under the old schema unless explicitly migrated.
New pages follow the new schema from the change date onward.

---

*Last schema update: 2026-04-06*
*Model: GPT-4o*
*Wiki owner: Sakshi*
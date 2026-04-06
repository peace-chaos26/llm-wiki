"""
eval/eval_query.py — query quality evaluation and RAG vs Wiki comparison.

Two evaluations:
  1. Wiki query quality  — faithfulness and citation accuracy of wiki answers
  2. RAG vs Wiki         — same questions answered two ways, LLM-as-judge scores

The RAG vs Wiki comparison is the headline result for the README.
Synthesis questions (requiring reasoning across multiple sources) should
favour the wiki. Lookup questions (single-hop factual) should be comparable.

Usage
-----
  cd src/llm_wiki

  # Run both evals on a golden test set
  python3 ../../eval/eval_query.py --test-set ../../eval/test_cases/rag_questions.json

  # RAG vs Wiki comparison only
  python3 ../../eval/eval_query.py --compare-only

  # JSON output for CI
  python3 ../../eval/eval_query.py --json

Golden test set format (eval/test_cases/*.json)
------------------------------------------------
  [
    {
      "question": "What is RAG?",
      "type": "lookup",           // lookup | synthesis | multi-hop
      "expected_concepts": ["retrieval-augmented-generation"],
      "reference_answer": "Optional reference for faithfulness scoring"
    }
  ]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

# Make src/llm_wiki importable — works whether run from repo root or eval/
_here = Path(__file__).resolve().parent
_repo_root = _here.parent
for _candidate in [
    _repo_root / "src" / "llm_wiki",
    _repo_root / "src",
    _repo_root,
]:
    if (_candidate / "config.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from config import settings
from wiki_manager import list_pages, parse_page

QuestionType = Literal["lookup", "synthesis", "multi-hop"]


# ── Test case dataclass ────────────────────────────────────────────────────────

@dataclass
class TestCase:
    question: str
    type: QuestionType = "lookup"
    expected_concepts: list[str] = field(default_factory=list)
    reference_answer: str = ""


@dataclass
class QueryEvalResult:
    question: str
    question_type: QuestionType
    wiki_answer: str
    rag_answer: str
    wiki_latency_s: float
    rag_latency_s: float

    # Scores (0–10 from LLM-as-judge)
    wiki_score: float = 0.0
    rag_score: float = 0.0

    # Citation metrics (wiki only)
    citation_count: int = 0
    citations_valid: int = 0
    citation_accuracy: float = 0.0

    # Faithfulness (wiki only)
    faithfulness_score: float = 0.0

    wiki_insufficient: bool = False


@dataclass
class EvalReport:
    total_questions: int = 0
    wiki_avg_score: float = 0.0
    rag_avg_score: float = 0.0
    wiki_win_rate: float = 0.0        # % questions where wiki score > rag score
    avg_citation_accuracy: float = 0.0
    avg_faithfulness: float = 0.0
    wiki_insufficient_rate: float = 0.0

    by_type: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        delta = self.wiki_avg_score - self.rag_avg_score
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        lines = [
            f"Query Evaluation Report",
            f"{'─'*42}",
            f"Questions evaluated : {self.total_questions}",
            f"",
            f"Wiki avg score      : {self.wiki_avg_score:.1f}/10",
            f"RAG  avg score      : {self.rag_avg_score:.1f}/10",
            f"Wiki vs RAG delta   : {delta_str}",
            f"Wiki win rate       : {self.wiki_win_rate*100:.1f}%",
            f"",
            f"Citation accuracy   : {self.avg_citation_accuracy*100:.1f}%",
            f"Faithfulness        : {self.avg_faithfulness*100:.1f}%",
            f"Insufficient rate   : {self.wiki_insufficient_rate*100:.1f}%",
        ]
        if self.by_type:
            lines += ["", "By question type:"]
            for qtype, stats in self.by_type.items():
                w = stats.get("wiki_avg", 0)
                r = stats.get("rag_avg", 0)
                n = stats.get("count", 0)
                lines.append(
                    f"  {qtype:<12} n={n}  wiki={w:.1f}  rag={r:.1f}  "
                    f"Δ={w-r:+.1f}"
                )
        return "\n".join(lines)


# ── RAG baseline ───────────────────────────────────────────────────────────────

def _build_rag_context(question: str, top_k: int = 3) -> str:
    """
    Simple BM25 retrieval over raw source files.
    This is the RAG baseline — no compiled wiki, just raw docs + retrieval.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return "[rank-bm25 not installed — pip install rank-bm25]"

    raw_docs = []
    doc_paths = []

    # Collect all raw source files
    for ext in [".txt", ".md"]:
        for p in settings.raw_path.rglob(f"*{ext}"):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                raw_docs.append(text)
                doc_paths.append(p)
            except Exception:
                continue

    if not raw_docs:
        return "[No raw source files found — ingest some sources first]"

    # Tokenise and build BM25 index
    tokenised = [doc.lower().split() for doc in raw_docs]
    bm25 = BM25Okapi(tokenised)

    query_tokens = question.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Get top-k docs
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    context_parts = []
    for i in top_indices:
        if scores[i] > 0:
            # Truncate each doc to 800 chars
            snippet = raw_docs[i][:800].strip()
            context_parts.append(f"[Source: {doc_paths[i].name}]\n{snippet}")

    return "\n\n---\n\n".join(context_parts) if context_parts else "[No relevant documents found]"


async def _get_rag_answer(question: str, client, model: str) -> tuple[str, float]:
    """Answer a question using BM25 RAG over raw sources."""
    t0 = time.time()
    context = _build_rag_context(question)

    prompt = f"""Answer the following question using only the provided source documents.
Be specific and cite which source each claim comes from.

Question: {question}

Source documents:
{context}

Answer:"""

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    answer = response.choices[0].message.content or ""
    return answer, time.time() - t0


async def _get_wiki_answer(question: str) -> tuple[str, float, bool]:
    """Answer a question using the compiled wiki pipeline."""
    import json as _json
    t0 = time.time()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "llm_wiki"))
    from query import query_wiki

    answer = ""
    wiki_insufficient = False

    async for chunk in query_wiki(question, file_back=False):
        if chunk.startswith("__RESULT__:"):
            result = _json.loads(chunk[11:])
            wiki_insufficient = result.get("wiki_insufficient", False)
        elif not chunk.startswith(("🔍", "📖", "💬", "✅", "⚠️", "   ")):
            answer += chunk

    return answer.strip(), time.time() - t0, wiki_insufficient


# ── Citation metrics ───────────────────────────────────────────────────────────

def _measure_citations(answer: str) -> tuple[int, int, float]:
    """
    Count [[wikilinks]] in the answer and check if the cited pages exist.
    Returns (total_citations, valid_citations, accuracy).
    """
    wikilink_re = re.compile(r"\[\[([^\[\]]+)\]\]")
    citations = wikilink_re.findall(answer)

    if not citations:
        return 0, 0, 0.0

    valid = 0
    for cite in citations:
        cite = cite.strip().removeprefix("wiki/")
        path = settings.wiki_path / f"{cite}.md"
        if path.exists():
            valid += 1

    accuracy = valid / len(citations)
    return len(citations), valid, accuracy


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

async def _judge_answer(
    question: str,
    answer: str,
    source_type: str,
    reference: str,
    client,
    model: str,
) -> float:
    """
    Score an answer 0-10 using LLM-as-judge.
    Evaluates: correctness, completeness, specificity.
    Returns float score.
    """
    ref_section = f"\nReference answer: {reference}" if reference else ""

    prompt = f"""You are evaluating the quality of an answer to a question.
Score the answer from 0 to 10 based on:
- Correctness (is the information accurate?)
- Completeness (does it fully address the question?)  
- Specificity (does it give concrete details vs vague generalities?)

Question: {question}
Answer ({source_type}): {answer[:1000]}{ref_section}

Respond with ONLY a single number from 0 to 10. No explanation."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        raw = (response.choices[0].message.content or "0").strip()
        score = float(re.search(r"\d+(?:\.\d+)?", raw).group())
        return min(10.0, max(0.0, score))
    except Exception:
        return 0.0


async def _judge_faithfulness(
    answer: str,
    wiki_pages: list[str],
    client,
    model: str,
) -> float:
    """
    Faithfulness: does the answer only use information present in the wiki pages?
    Returns 0.0-1.0.
    """
    if not wiki_pages:
        return 0.0

    # Fetch the wiki pages
    context = ""
    for slug in wiki_pages[:3]:
        slug = slug.removeprefix("wiki/")
        p = settings.wiki_path / f"{slug}.md"
        if p.exists():
            _, body = parse_page(p)
            context += f"\n[{slug}]\n{body[:600]}\n"

    if not context.strip():
        return 0.0

    prompt = f"""Does the following answer contain ONLY information that is present in the wiki pages below?
Answer YES (fully faithful), PARTIAL (mostly faithful, minor additions), or NO (contains information not in pages).

Answer: {answer[:600]}

Wiki pages:
{context}

Respond with only: YES, PARTIAL, or NO"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        if "YES" in raw:
            return 1.0
        elif "PARTIAL" in raw:
            return 0.5
        else:
            return 0.0
    except Exception:
        return 0.0


# ── Default test cases ─────────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
    TestCase(
        question="What is Retrieval-Augmented Generation?",
        type="lookup",
        expected_concepts=["retrieval-augmented-generation"],
    ),
    TestCase(
        question="What is the key advantage of RAG over standard language models?",
        type="lookup",
        expected_concepts=["retrieval-augmented-generation"],
    ),
    TestCase(
        question="How does RAG separate parametric and non-parametric memory?",
        type="synthesis",
        expected_concepts=["retrieval-augmented-generation"],
    ),
    TestCase(
        question="Who introduced RAG and when was it published?",
        type="lookup",
        expected_concepts=["retrieval-augmented-generation"],
    ),
    TestCase(
        question="What are the limitations of RAG systems?",
        type="synthesis",
        expected_concepts=["retrieval-augmented-generation"],
    ),
]


def load_test_cases(path: Path | None) -> list[TestCase]:
    if path is None or not path.exists():
        print(f"No test set found — using {len(DEFAULT_TEST_CASES)} built-in cases.")
        return DEFAULT_TEST_CASES

    with open(path) as f:
        raw = json.load(f)
    return [TestCase(**item) for item in raw]


# ── Main eval runner ───────────────────────────────────────────────────────────

async def run_eval(
    test_cases: list[TestCase],
    compare_rag: bool = True,
) -> EvalReport:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = settings.llm_model

    report = EvalReport(total_questions=len(test_cases))
    all_results = []

    print(f"\nEvaluating {len(test_cases)} questions...\n")

    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc.question[:60]}...")

        # Wiki answer
        wiki_answer, wiki_lat, wiki_insufficient = await _get_wiki_answer(tc.question)

        # RAG answer (baseline)
        if compare_rag:
            rag_answer, rag_lat = await _get_rag_answer(tc.question, client, model)
        else:
            rag_answer, rag_lat = "[RAG comparison skipped]", 0.0

        # Citation metrics (wiki only)
        n_cites, n_valid, cite_acc = _measure_citations(wiki_answer)

        # LLM-as-judge scores
        wiki_score = 0.0
        rag_score = 0.0
        faithfulness = 0.0

        if not wiki_insufficient and wiki_answer.strip():
            wiki_score = await _judge_answer(
                tc.question, wiki_answer, "wiki", tc.reference_answer, client, model
            )
            faithfulness = await _judge_faithfulness(
                wiki_answer, tc.expected_concepts, client, model
            )

        if compare_rag and rag_answer.strip():
            rag_score = await _judge_answer(
                tc.question, rag_answer, "rag", tc.reference_answer, client, model
            )

        result = QueryEvalResult(
            question=tc.question,
            question_type=tc.type,
            wiki_answer=wiki_answer[:500],
            rag_answer=rag_answer[:500],
            wiki_latency_s=round(wiki_lat, 2),
            rag_latency_s=round(rag_lat, 2),
            wiki_score=wiki_score,
            rag_score=rag_score,
            citation_count=n_cites,
            citations_valid=n_valid,
            citation_accuracy=cite_acc,
            faithfulness_score=faithfulness,
            wiki_insufficient=wiki_insufficient,
        )
        all_results.append(result)

        status = "✓" if not wiki_insufficient else "⚠"
        print(
            f"     {status} wiki={wiki_score:.1f}  rag={rag_score:.1f}  "
            f"cite_acc={cite_acc*100:.0f}%  faithful={faithfulness*100:.0f}%"
        )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    valid = [r for r in all_results if not r.wiki_insufficient]
    n_valid = len(valid)

    report.wiki_avg_score = (
        sum(r.wiki_score for r in valid) / n_valid if n_valid else 0.0
    )
    report.rag_avg_score = (
        sum(r.rag_score for r in valid) / n_valid if n_valid else 0.0
    )
    report.wiki_win_rate = (
        sum(1 for r in valid if r.wiki_score > r.rag_score) / n_valid
        if n_valid else 0.0
    )
    report.avg_citation_accuracy = (
        sum(r.citation_accuracy for r in valid) / n_valid if n_valid else 0.0
    )
    report.avg_faithfulness = (
        sum(r.faithfulness_score for r in valid) / n_valid if n_valid else 0.0
    )
    report.wiki_insufficient_rate = (
        sum(1 for r in all_results if r.wiki_insufficient) / len(all_results)
        if all_results else 0.0
    )

    # By question type
    for qtype in ("lookup", "synthesis", "multi-hop"):
        typed = [r for r in valid if r.question_type == qtype]
        if typed:
            report.by_type[qtype] = {
                "count": len(typed),
                "wiki_avg": round(sum(r.wiki_score for r in typed) / len(typed), 2),
                "rag_avg": round(sum(r.rag_score for r in typed) / len(typed), 2),
            }

    report.results = [asdict(r) for r in all_results]
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query quality evaluation")
    parser.add_argument("--test-set", type=Path, help="Path to JSON test case file")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip faithfulness/citation, only do RAG vs Wiki")
    parser.add_argument("--no-rag", action="store_true",
                        help="Skip RAG baseline comparison")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    args = parser.parse_args()

    test_cases = load_test_cases(args.test_set)
    report = asyncio.run(run_eval(test_cases, compare_rag=not args.no_rag))

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print("\n" + "="*52)
        print("  QUERY EVALUATION REPORT")
        print("="*52)
        print(report.summary())
        print("="*52 + "\n")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Results saved to {args.output}")
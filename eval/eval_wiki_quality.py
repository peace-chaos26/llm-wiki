"""
eval/eval_wiki_quality.py — wiki structure quality metrics.

Measures how well-compiled the wiki is, independent of any specific query.
Run after ingesting a batch of sources to get a health snapshot.

Metrics
-------
  1. entity_coverage      — % of named entities in sources that have wiki pages
  2. cross_link_density   — avg inbound [[wikilinks]] per non-index page
  3. orphan_rate          — % of pages with 0 real inbound links
  4. stub_rate            — % of concept pages still at stub status
  5. source_freshness     — % of wiki pages with no stale provenance flags
  6. avg_concept_length   — avg word count of concept pages (proxy for depth)

Usage
-----
  cd src/llm_wiki
  python3 ../../eval/eval_wiki_quality.py

  # or with --json for machine-readable output
  python3 ../../eval/eval_wiki_quality.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Make src/llm_wiki importable — works whether run from repo root or eval/
_here = Path(__file__).resolve().parent        # eval/
_repo_root = _here.parent                      # repo root
# Support both flat layout (dev) and src/llm_wiki layout (installed)
for _candidate in [
    _repo_root / "src" / "llm_wiki",
    _repo_root / "src",
    _repo_root,
]:
    if (_candidate / "config.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from config import settings
from provenance import get_stale_summary
from wiki_manager import (
    build_backlink_map,
    list_all_wiki_pages,
    list_pages,
    page_word_count,
    parse_page,
)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class WikiQualityReport:
    # Counts
    total_pages: int = 0
    total_sources: int = 0
    total_entities: int = 0
    total_concepts: int = 0

    # Metrics (0.0 – 1.0 or raw numbers)
    entity_coverage: float = 0.0       # % entities with wiki pages
    cross_link_density: float = 0.0    # avg inbound links per page
    orphan_rate: float = 0.0           # % pages with 0 inbound links
    stub_rate: float = 0.0             # % concept pages still stub
    source_freshness: float = 0.0      # % pages with no stale flags
    avg_concept_length: float = 0.0    # avg words in concept pages

    # Raw supporting data
    orphan_pages: list[str] = field(default_factory=list)
    stub_pages: list[str] = field(default_factory=list)
    stale_pages: int = 0
    longest_concepts: list[tuple[str, int]] = field(default_factory=list)

    def score(self) -> float:
        """
        Composite 0–100 wiki quality score.
        Weights reflect what matters most for a useful wiki.
        """
        return round(
            self.entity_coverage    * 25 +   # are key entities documented?
            (1 - self.orphan_rate)  * 20 +   # are pages connected?
            (1 - self.stub_rate)    * 20 +   # are concepts developed?
            self.source_freshness   * 20 +   # is knowledge current?
            min(self.cross_link_density / 3, 1.0) * 15,  # cap at 3 links avg
            1
        )

    def summary(self) -> str:
        lines = [
            f"Wiki Quality Score : {self.score()}/100",
            f"",
            f"Pages              : {self.total_pages} total  "
            f"({self.total_sources} sources, "
            f"{self.total_entities} entities, "
            f"{self.total_concepts} concepts)",
            f"",
            f"Entity coverage    : {self.entity_coverage*100:.1f}%  "
            f"({self.total_entities} entity pages)",
            f"Cross-link density : {self.cross_link_density:.2f} avg inbound links/page",
            f"Orphan rate        : {self.orphan_rate*100:.1f}%  "
            f"({len(self.orphan_pages)} orphan pages)",
            f"Stub rate          : {self.stub_rate*100:.1f}%  "
            f"({len(self.stub_pages)} stubs)",
            f"Source freshness   : {self.source_freshness*100:.1f}%  "
            f"({self.stale_pages} stale pages)",
            f"Avg concept length : {self.avg_concept_length:.0f} words",
        ]
        if self.orphan_pages:
            lines += ["", "Orphan pages:"]
            for p in self.orphan_pages[:5]:
                lines.append(f"  • {p}")
            if len(self.orphan_pages) > 5:
                lines.append(f"  ... and {len(self.orphan_pages)-5} more")
        if self.stub_pages:
            lines += ["", "Stub concept pages:"]
            for p in self.stub_pages[:5]:
                lines.append(f"  • {p}")
        return "\n".join(lines)


# ── Individual metric functions ────────────────────────────────────────────────

def measure_entity_coverage() -> tuple[float, int]:
    """
    Entity coverage: fraction of entities mentioned in source pages
    that have their own wiki page.

    Scans all source page frontmatter for key_entities, checks which
    have corresponding entity pages.
    """
    mentioned: set[str] = set()
    for p in list_pages("source"):
        fm, _ = parse_page(p)
        for slug in fm.get("key_entities", []):
            mentioned.add(slug.strip())

    if not mentioned:
        # Fall back to counting entity pages vs source mentions
        entity_pages = len(list_pages("entity"))
        return (1.0 if entity_pages > 0 else 0.0), entity_pages

    have_pages = sum(
        1 for slug in mentioned
        if (settings.wiki_path / "entities" / f"{slug}.md").exists()
    )
    return have_pages / len(mentioned), len(list_pages("entity"))


def measure_link_metrics() -> tuple[float, float, list[str]]:
    """
    Returns (cross_link_density, orphan_rate, orphan_page_list).

    cross_link_density: average number of inbound wikilinks per page
    orphan_rate: fraction of pages with 0 real inbound links
    """
    exempt = {"index", "log", "overview"}
    backlinks = build_backlink_map()

    all_pages = [
        p for p in list_all_wiki_pages()
        if p.stem not in exempt
    ]
    if not all_pages:
        return 0.0, 0.0, []

    total_inbound = 0
    orphans = []

    for page in all_pages:
        try:
            rel = str(page.relative_to(settings.wiki_path)).removesuffix(".md")
        except ValueError:
            continue
        inbound = [
            b for b in backlinks.get(rel, [])
            if b not in exempt
        ]
        total_inbound += len(inbound)
        if not inbound:
            orphans.append(rel)

    density = total_inbound / len(all_pages)
    orphan_rate = len(orphans) / len(all_pages)
    return density, orphan_rate, orphans


def measure_stub_rate() -> tuple[float, list[str]]:
    """Fraction of concept pages still at stub status."""
    concepts = list_pages("concept")
    if not concepts:
        return 0.0, []

    stubs = []
    for p in concepts:
        fm, _ = parse_page(p)
        if fm.get("status", "stub") == "stub":
            stubs.append(p.stem)

    return len(stubs) / len(concepts), stubs


def measure_source_freshness(total_pages: int) -> tuple[float, int]:
    """
    Fraction of wiki pages with no stale provenance flags.
    Uses provenance.db summary rather than re-scanning files.
    """
    summary = get_stale_summary()
    stale_pages = summary.get("stale_wiki_pages", 0)
    if total_pages == 0:
        return 1.0, 0
    fresh_pages = max(0, total_pages - stale_pages)
    return fresh_pages / total_pages, stale_pages


def measure_concept_depth() -> tuple[float, list[tuple[str, int]]]:
    """Average word count of concept pages. Returns (avg, top_5_longest)."""
    concepts = list_pages("concept")
    if not concepts:
        return 0.0, []

    lengths = []
    for p in concepts:
        wc = page_word_count(p)
        lengths.append((p.stem, wc))

    avg = sum(wc for _, wc in lengths) / len(lengths)
    top5 = sorted(lengths, key=lambda x: x[1], reverse=True)[:5]
    return avg, top5


# ── Main runner ────────────────────────────────────────────────────────────────

def run_eval() -> WikiQualityReport:
    report = WikiQualityReport(
        total_sources=len(list_pages("source")),
        total_entities=len(list_pages("entity")),
        total_concepts=len(list_pages("concept")),
    )
    report.total_pages = (
        report.total_sources + report.total_entities +
        report.total_concepts + len(list_pages("query"))
    )

    report.entity_coverage, _ = measure_entity_coverage()

    (report.cross_link_density,
     report.orphan_rate,
     report.orphan_pages) = measure_link_metrics()

    report.stub_rate, report.stub_pages = measure_stub_rate()

    report.source_freshness, report.stale_pages = measure_source_freshness(
        report.total_pages
    )

    report.avg_concept_length, report.longest_concepts = measure_concept_depth()

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wiki quality evaluation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = run_eval()

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print("\n" + "="*52)
        print("  WIKI QUALITY REPORT")
        print("="*52)
        print(report.summary())
        print("="*52 + "\n")
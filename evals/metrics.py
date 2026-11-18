"""Deterministic eval metrics (cheap, exact, run on every change)."""

from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+)\]")


def citation_check(draft: str, n_sources: int) -> dict[str, object]:
    cites = [int(n) for n in _CITE.findall(draft)]
    return {
        "n_citations": len(cites),
        "citations_valid": all(1 <= c <= n_sources for c in cites),
        "has_citations": bool(cites),
    }


def key_fact_recall(report: str, key_facts: list[list[str]]) -> float:
    """Each key fact is a list of acceptable phrasings; it counts if any appears."""
    if not key_facts:
        return 1.0
    text = report.lower()
    hits = sum(any(alt.lower() in text for alt in alts) for alts in key_facts)
    return hits / len(key_facts)


def source_hit(sources: list[dict], expected: list[str]) -> bool:
    urls = {s["url"] for s in sources}
    return all(f"kb://{e}" in urls for e in expected)

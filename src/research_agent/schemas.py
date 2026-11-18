"""Small data types shared across tools, graph and API."""

from __future__ import annotations

from typing import Literal, TypedDict


class Source(TypedDict):
    """A citable source. Tools attach these as ToolMessage *artifacts* (see tools/web.py)."""

    title: str
    url: str  # http(s) URL for web results, "kb://<file>" for knowledge-base chunks
    kind: Literal["web", "kb"]


def dedupe_sources(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            out.append(s)
    return out

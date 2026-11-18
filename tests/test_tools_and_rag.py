from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from research_agent.graph.graph import clean_plan, number_sources, sanitize_citations
from research_agent.llm import invoke_structured
from research_agent.rag.ingest import get_vectorstore, ingest_directory
from research_agent.tools import web
from research_agent.tools.kb import make_kb_tool
from tests.fakes import ScriptedChatModel

# ---------------------------------------- web ----------------------------------------


async def test_web_search_formats_results_and_artifacts(settings):
    def fake_search(query, n):
        return [{"title": "RAG", "href": "https://ex.com/rag", "body": "retrieval"}][:n]

    search, _ = web.make_web_tools(settings, search_fn=fake_search)
    msg = await search.ainvoke(
        {"type": "tool_call", "name": "web_search", "args": {"query": "rag"}, "id": "1"}
    )
    assert "https://ex.com/rag" in msg.content
    assert msg.artifact == [{"title": "RAG", "url": "https://ex.com/rag", "kind": "web"}]


async def test_web_search_failure_returns_message_not_exception(settings):
    def boom(query, n):
        raise RuntimeError("rate limited")

    search, _ = web.make_web_tools(settings, search_fn=boom)
    assert "Search failed" in await search.ainvoke({"query": "x"})


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1:11434/api/tags"])
async def test_fetch_url_blocks_non_public_urls(settings, url):
    _, fetch = web.make_web_tools(settings)
    assert (await fetch.ainvoke({"url": url})).startswith("Refused")


async def test_fetch_url_extracts_and_truncates(settings, monkeypatch):
    monkeypatch.setattr(web, "_is_public_http_url", lambda url: True)
    body = "<html><head><title>T</title></head><body><article>" + "<p>word " * 2000 + "</article>"
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
    settings.fetch_max_chars = 100
    _, fetch = web.make_web_tools(
        settings, http_client_factory=lambda: httpx.AsyncClient(transport=transport)
    )
    msg = await fetch.ainvoke(
        {"type": "tool_call", "name": "fetch_url", "args": {"url": "https://ex.com"}, "id": "1"}
    )
    assert msg.content.endswith("[truncated]") and len(msg.content) < 140
    assert msg.artifact[0]["url"] == "https://ex.com"


# ---------------------------------------- RAG ----------------------------------------


async def test_ingest_is_idempotent_and_searchable(settings, embeddings):
    (settings.docs_dir / "rag.md").write_text("# RAG\n\nRetrieval augmented generation. " * 40)
    (settings.docs_dir / "notes.txt").write_text("LangGraph checkpointers persist state.")
    (settings.docs_dir / "image.png").write_bytes(b"\x89PNG")
    vs = get_vectorstore(settings, embeddings)

    first = ingest_directory(vs, settings)
    assert sorted(first.files) == ["notes.txt", "rag.md"]
    assert first.chunks > 2 and any(s.endswith("image.png") for s in first.skipped)
    count = len(vs.get()["ids"])

    ingest_directory(vs, settings)  # re-ingest: replaces, doesn't duplicate
    assert len(vs.get()["ids"]) == count

    tool = make_kb_tool(vs, k=2)
    msg = await tool.ainvoke(
        {"type": "tool_call", "name": "search_knowledge_base", "args": {"query": "x"}, "id": "1"}
    )
    assert "source:" in msg.content and msg.artifact[0]["url"].startswith("kb://")


# -------------------------------------- helpers --------------------------------------


def test_clean_plan_dedupes_strips_and_falls_back():
    assert clean_plan(["1. A", "a", " - B ", ""], "Q", 5) == ["A", "B"]
    assert clean_plan([], "Q", 3) == ["Q"]
    assert clean_plan(["a", "b", "c"], "Q", 2) == ["a", "b"]


def test_number_sources_dedupes_across_findings():
    s1 = {"title": "a", "url": "u1", "kind": "web"}
    s2 = {"title": "b", "url": "u2", "kind": "kb"}
    sources, notes = number_sources(
        [
            {"sub_question": "q1", "summary": "x", "sources": [s1, s2]},
            {"sub_question": "q2", "summary": "y", "sources": [s2]},
        ]
    )
    assert [s["url"] for s in sources] == ["u1", "u2"]
    assert "### q2\nSources: [2]" in notes


def test_sanitize_citations():
    assert sanitize_citations("a [1] b [3] c [2][9]", 2) == "a [1] b  c [2]"


class _Out(BaseModel):
    value: int


async def test_invoke_structured_retries_then_succeeds_or_gives_up():
    ok = ScriptedChatModel(structured={"_Out": [ValueError("bad"), {"value": 7}]})
    assert (await invoke_structured(ok, _Out, [], retries=1)).value == 7
    bad = ScriptedChatModel(structured={"_Out": [ValueError("1"), ValueError("2")]})
    assert await invoke_structured(bad, _Out, [], retries=1) is None


async def test_kb_grader_filters_irrelevant_passages(embeddings):
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore

    vs = InMemoryVectorStore(embeddings)
    vs.add_documents(
        [
            Document("relevant", metadata={"source": "a.md"}),
            Document("junk", metadata={"source": "b.md"}),
        ]
    )

    def grade(msgs):  # keep only the passage whose text is "relevant"
        body = str(msgs[-1].content)
        return {"relevant": [i for i in (1, 2) if f"[{i}] relevant" in body]}

    tool = make_kb_tool(vs, k=2, grader=ScriptedChatModel(structured={"RelevantPassages": grade}))
    msg = await tool.ainvoke(
        {"type": "tool_call", "name": "search_knowledge_base", "args": {"query": "q"}, "id": "1"}
    )
    assert [s["url"] for s in msg.artifact] == ["kb://a.md"]

    none = make_kb_tool(
        vs, k=2, grader=ScriptedChatModel(structured={"RelevantPassages": [{"relevant": []}]})
    )
    assert "Use web_search" in await none.ainvoke({"query": "q"})


async def test_web_search_retries_transient_failure(settings):
    calls = []

    def flaky(query, n):
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("No results found.")
        return [{"title": "t", "href": "https://ex.com", "body": "b"}]

    search, _ = web.make_web_tools(settings, search_fn=flaky)
    assert "https://ex.com" in await search.ainvoke({"query": "x"})
    assert len(calls) == 2

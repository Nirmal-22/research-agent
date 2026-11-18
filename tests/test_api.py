"""API tests: real FastAPI app + real graph + fake model, in-memory checkpointer."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.api.main import create_app
from research_agent.deps import AppRuntime
from research_agent.graph.graph import build_graph
from tests.fakes import ScriptedChatModel


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        ev, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            events.append((ev, json.loads(data)))
    return events


@pytest.fixture
async def client(settings, kb_tool, embeddings):
    llm = ScriptedChatModel(
        structured={
            "Plan": lambda _: {"sub_questions": ["What is RAG?"]},
            "Review": lambda _: {"score": 5, "feedback": ""},
        }
    )

    @asynccontextmanager
    async def fake_runtime(s):
        graph = build_graph(llm, [kb_tool], s, checkpointer=InMemorySaver())
        yield AppRuntime(
            settings=s, graph=graph, vectorstore=InMemoryVectorStore(embeddings), llm=llm
        )

    app = create_app(settings, runtime_factory=fake_runtime)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_research_interrupt_resume_and_state(client):
    r = await client.post("/research", json={"question": "How does RAG work?"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "thread" and "interrupt" in kinds and kinds[-1] == "done"
    thread_id = events[0][1]["thread_id"]
    assert events[-1][1]["status"] == "awaiting_approval"

    r = await client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
    events = parse_sse(r.text)
    nodes = [d["node"] for e, d in events if e == "update"]
    assert {"research", "write", "review"} <= set(nodes)
    assert any(e == "token" for e, _ in events), "writer tokens are streamed"
    done = events[-1][1]
    assert done["status"] == "done" and "## Sources" in done["report"]

    state = (await client.get(f"/research/{thread_id}")).json()
    assert state["status"] == "done" and state["findings"][0]["sub_question"] == "What is RAG?"

    # resuming a finished thread is a client error
    assert (await client.post(f"/research/{thread_id}/resume", json={})).status_code == 409


async def test_auto_approve_and_errors(client):
    events = parse_sse(
        (
            await client.post("/research", json={"question": "What is RAG?", "auto_approve": True})
        ).text
    )
    assert "interrupt" not in [e for e, _ in events] and events[-1][1]["status"] == "done"
    assert (await client.get("/research/nope")).status_code == 404
    assert (await client.post("/research/nope/resume", json={})).status_code == 404
    assert (await client.post("/research", json={"question": "x"})).status_code == 422
    bad = await client.post("/research/nope/resume", json={"action": "explode"})
    assert bad.status_code == 422


async def test_health_and_index(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert "Research Agent" in (await client.get("/")).text

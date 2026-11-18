"""Turn LangGraph's stream into small, JSON-safe events (shared by the API's SSE and the CLI).

`graph.astream(stream_mode=["updates", "messages"], version="v2")` yields typed parts:
  {"type": "updates",  "ns": (...), "data": {node_name: state_update}}  after each node
  {"type": "messages", "ns": (...), "data": (AIMessageChunk, metadata)}  per LLM token
`updates` tells you WHAT happened; `messages` gives token-level streaming for UX.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from research_agent.config import Settings
from research_agent.graph.graph import run_config
from research_agent.graph.state import GraphContext

Event = dict[str, Any]  # {"event": str, "data": JSON-able}

TOKEN_NODES = {"write"}  # only stream tokens of the user-facing report


def summarize_update(node: str, update: Any) -> dict[str, Any]:
    """Keep stream events small: no full message lists, no big blobs."""
    if not isinstance(update, dict):
        return {}
    out: dict[str, Any] = {}
    if "plan" in update:
        out["plan"] = update["plan"]
    if "findings" in update:
        out["findings"] = [
            {
                "sub_question": f["sub_question"],
                "summary": f["summary"],
                "sources": [s["url"] for s in f["sources"]],
            }
            for f in update["findings"]
        ]
    if "review" in update:
        out["review"] = update["review"]
    if "revision" in update:
        out["revision"] = update["revision"]
    if "status" in update:
        out["status"] = update["status"]
    if "draft" in update:
        out["draft_chars"] = len(update["draft"])
    return out


def interrupt_payloads(interrupts: Any) -> list[dict[str, Any]]:
    out = []
    for it in interrupts or ():
        value = getattr(it, "value", None) if not isinstance(it, dict) else it.get("value")
        out.append({"id": getattr(it, "id", None), "value": value})
    return out


async def stream_events(
    graph: CompiledStateGraph,
    graph_input: dict[str, Any] | Command,
    thread_id: str,
    settings: Settings,
    *,
    auto_approve: bool = False,
) -> AsyncIterator[Event]:
    config = run_config(thread_id, settings)
    async for part in graph.astream(
        graph_input,
        config,
        context=GraphContext(auto_approve=auto_approve),
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if part["type"] == "messages":
            chunk, meta = part["data"]
            node = meta.get("langgraph_node")
            # Streaming models emit AIMessageChunk tokens; non-streaming ones one AIMessage.
            if node in TOKEN_NODES and isinstance(chunk, AIMessage) and chunk.text:
                yield {"event": "token", "data": {"node": node, "text": chunk.text}}
        elif part["type"] == "updates":
            for node, update in part["data"].items():
                if node == "__interrupt__":
                    yield {"event": "interrupt", "data": interrupt_payloads(update)}
                elif not node.startswith("__"):
                    yield {
                        "event": "update",
                        "data": {"node": node, **summarize_update(node, update)},
                    }

    yield {"event": "done", "data": await thread_summary(graph, thread_id, settings)}


async def thread_summary(
    graph: CompiledStateGraph, thread_id: str, settings: Settings
) -> dict[str, Any]:
    snap = await graph.aget_state(run_config(thread_id, settings))
    values = snap.values or {}
    interrupts = interrupt_payloads(snap.interrupts)
    if interrupts:
        status = "awaiting_approval"
    elif snap.next:
        status = "running"
    else:
        status = values.get("status") or ("done" if values.get("report") else "empty")
    return {
        "thread_id": thread_id,
        "status": status,
        "next": list(snap.next),
        "interrupts": interrupts,
        "question": values.get("question"),
        "plan": values.get("plan"),
        "findings": summarize_update("research", {"findings": values.get("findings", [])}).get(
            "findings", []
        ),
        "review": values.get("review"),
        "revision": values.get("revision", 0),
        "report": values.get("report"),
    }

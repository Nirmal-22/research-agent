"""Researcher subgraph: a tool-calling agent that answers one sub-question.

`create_agent` (LangChain v1) builds a ready-made LangGraph ReAct loop:

    model --(tool_calls?)--> tools --> model --> ... --> final answer

It's a compiled graph itself; invoked from inside a parent-graph node it runs as a
*subgraph* (its events stream through the parent, it shares the parent's checkpointer).
*Middleware* hooks into that loop. We use it to cap model calls, the #1 defence against a
small model looping on tool calls forever.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.graph.prompts import (
    RESEARCH_FALLBACK_PROMPT,
    RESEARCHER_SYSTEM,
    WEB_RULE,
    today,
)
from research_agent.graph.state import Finding, ResearchTask
from research_agent.logging import get_logger
from research_agent.schemas import Source, dedupe_sources

log = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def make_researcher(
    llm: BaseChatModel, tools: Sequence[BaseTool], settings: Settings
) -> CompiledStateGraph:
    has_web = any(t.name == "web_search" for t in tools)

    # dynamic_prompt middleware computes the system prompt on every model call, so values
    # like today's date stay correct in a long-running server (a static string would not).
    @dynamic_prompt
    def researcher_prompt(request: ModelRequest) -> str:
        return RESEARCHER_SYSTEM.format(
            today=today(),
            web_rule=WEB_RULE if has_web else "",
            max_tool_rounds=max(1, settings.researcher_max_model_calls - 1),
        )

    return create_agent(
        llm,
        tools=list(tools),
        middleware=[
            researcher_prompt,
            # Hard cap per sub-question. exit_behavior="end" stops the loop gracefully
            # (appends a notice message) instead of raising.
            ModelCallLimitMiddleware(
                run_limit=settings.researcher_max_model_calls, exit_behavior="end"
            ),
        ],
        name="researcher",
    )


def _collect(messages: Sequence[BaseMessage]) -> tuple[list[Source], list[str]]:
    """Pull citations from ToolMessage artifacts (never trust the model to copy URLs)."""
    sources: list[Source] = []
    tool_texts: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            if isinstance(m.artifact, list):
                sources.extend(m.artifact)
            tool_texts.append(f"[{m.name}] {str(m.content)[:1500]}")
    return dedupe_sources(sources), tool_texts


def _final_answer(messages: Sequence[BaseMessage]) -> str | None:
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None
    text = strip_think(last.text)
    if not text or text.startswith("Model call limits exceeded"):
        return None
    return text


async def run_researcher(
    agent: CompiledStateGraph, llm: BaseChatModel, task: ResearchTask
) -> Finding:
    sub_q = task["sub_question"]
    prompt = f"Overall question: {task['question']}\nYour sub-question: {sub_q}"
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(prompt)]})
    except GraphBubbleUp:
        raise  # interrupts / control flow must propagate
    except Exception as exc:
        log.error("researcher.failed", sub_question=sub_q, error=repr(exc)[:300])
        return Finding(
            sub_question=sub_q, summary=f"(research failed: {type(exc).__name__})", sources=[]
        )

    messages: list[BaseMessage] = result["messages"]
    sources, tool_texts = _collect(messages)
    summary = _final_answer(messages)
    if summary is None:
        # Hit the call cap (or ended on a tool call): salvage what the tools returned with
        # one plain, tool-free LLM call. Graceful degradation > crash.
        log.info("researcher.fallback_summary", sub_question=sub_q, tool_results=len(tool_texts))
        if tool_texts:
            chain = RESEARCH_FALLBACK_PROMPT | llm | StrOutputParser()  # LCEL pipeline
            summary = strip_think(
                await chain.ainvoke(
                    {
                        "sub_question": sub_q,
                        "tool_results": "\n\n".join(tool_texts)[:6000],
                        "today": today(),
                    }
                )
            )
        else:
            summary = "(no information found)"
    return Finding(sub_question=sub_q, summary=summary, sources=sources)

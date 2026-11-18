"""The main research workflow.

    START -> plan -> human_review --(interrupt: approve/edit)--> research x N  (Send fan-out)
                ^          |  (replan)                               |
                +----------+       (reject) -> END                   v
                                                   write <---------- (join)
                                                     |   ^
                                                     v   | score < 4 and revisions left
                                                   review (LLM-as-judge)
                                                     |
                                                     +--> END

Concepts on display: typed state + reducers, structured output, `interrupt` (human in the
loop), `Command` (update state AND choose the next node from inside a node), `Send` (dynamic
map-reduce fan-out), a subgraph (researcher), a conditional edge forming a cycle, and
checkpointing (required for interrupts; makes every run a resumable *thread*).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send, interrupt

from research_agent.config import Settings
from research_agent.graph.prompts import (
    PLAN_PROMPT,
    REVIEW_PROMPT,
    REVISION_BLOCK,
    WRITE_PROMPT,
    today,
)
from research_agent.graph.researcher import make_researcher, run_researcher, strip_think
from research_agent.graph.state import (
    Finding,
    GraphContext,
    Plan,
    PlanDecision,
    ResearchInput,
    ResearchState,
    ResearchTask,
    Review,
)
from research_agent.llm import invoke_structured
from research_agent.logging import get_logger, timed_node
from research_agent.schemas import Source

log = get_logger(__name__)

APPROVAL_SCORE = 4
_CITE_RE = re.compile(r"\[(\d+)\]")


# ----------------------------- pure helpers (unit-tested) -----------------------------


def clean_plan(sub_questions: Sequence[str], question: str, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in sub_questions:
        q = q.strip().lstrip("-*0123456789. ").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:limit] or [question]


def number_sources(findings: Sequence[Finding]) -> tuple[list[Source], str]:
    """Give every unique source a global number and render the notes with those numbers."""
    numbered: list[Source] = []
    index: dict[str, int] = {}
    blocks: list[str] = []
    for f in findings:
        refs: list[int] = []
        for s in f["sources"]:
            if s["url"] not in index:
                numbered.append(s)
                index[s["url"]] = len(numbered)
            refs.append(index[s["url"]])
        ref_txt = ", ".join(f"[{n}]" for n in refs) or "(none)"
        blocks.append(f"### {f['sub_question']}\nSources: {ref_txt}\n{f['summary']}")
    return numbered, "\n\n".join(blocks)


def render_source_list(sources: Sequence[Source]) -> str:
    return "\n".join(f"[{i}] {s['title']} - {s['url']}" for i, s in enumerate(sources, 1))


def sanitize_citations(text: str, n_sources: int) -> str:
    """Guardrail: drop citation markers that point at sources that don't exist."""
    return _CITE_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)


def append_sources_section(draft: str, sources: Sequence[Source]) -> str:
    if not sources:
        return draft
    return f"{draft.rstrip()}\n\n## Sources\n\n" + "\n".join(
        f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1)
    )


def run_config(thread_id: str, settings: Settings) -> RunnableConfig:
    # thread_id selects the checkpoint "thread": same id = same conversation/run to resume.
    return {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": settings.max_concurrency,
        "recursion_limit": 50,
    }


# ------------------------------------- the graph --------------------------------------


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    researcher = make_researcher(llm, tools, settings)

    @timed_node("plan")
    async def plan(state: ResearchState) -> dict[str, Any]:
        feedback = state.get("plan_feedback")
        prompt = await PLAN_PROMPT.ainvoke(
            {
                "question": state["question"],
                "max_subquestions": settings.max_subquestions,
                "feedback": f"Reviewer guidance for the plan: {feedback}" if feedback else "",
            }
        )
        result = await invoke_structured(
            llm, Plan, prompt.to_messages(), retries=settings.structured_output_retries
        )
        subs = result.sub_questions if result else []  # fallback: research the question as-is
        return {
            "plan": clean_plan(subs, state["question"], settings.max_subquestions),
            "plan_feedback": None,
        }

    @timed_node("human_review")
    async def human_review(state: ResearchState, runtime: Runtime[GraphContext]) -> Command:
        plan_ = state["plan"]
        if runtime.context is not None and runtime.context.auto_approve:
            decision = PlanDecision(action="approve")
        else:
            # interrupt() pauses the graph here and surfaces this payload to the client.
            # The state is checkpointed; the run resumes when the client sends
            # Command(resume=...), and *this node re-runs from the top*, with interrupt()
            # now returning the resume value (validated against response_schema).
            raw = interrupt(
                {"type": "plan_review", "question": state["question"], "plan": plan_},
                response_schema=PlanDecision,
            )
            decision = raw if isinstance(raw, PlanDecision) else PlanDecision.model_validate(raw)

        log.info("plan.decision", action=decision.action)
        if decision.action == "reject":
            return Command(goto=END, update={"status": "rejected"})
        if decision.action == "replan":
            return Command(goto="plan", update={"plan_feedback": decision.feedback or ""})
        if decision.action == "edit" and decision.sub_questions:
            plan_ = clean_plan(decision.sub_questions, state["question"], settings.max_subquestions)

        # Send = dynamic fan-out: one "research" task per sub-question, run in parallel
        # (bounded by max_concurrency), each with its OWN input. Their {"findings": [...]}
        # outputs are merged by the operator.add reducer before "write" runs.
        return Command(
            update={"plan": plan_},
            goto=[
                Send("research", ResearchTask(question=state["question"], sub_question=q))
                for q in plan_
            ],
        )

    @timed_node("research")
    async def research(task: ResearchTask) -> dict[str, Any]:
        finding = await run_researcher(researcher, llm, task)
        return {"findings": [finding]}

    @timed_node("write")
    async def write(state: ResearchState) -> dict[str, Any]:
        sources, notes = number_sources(state.get("findings", []))
        review = state.get("review")
        is_revision = bool(review and state.get("draft"))
        revision_block = (
            REVISION_BLOCK.format(feedback=review["feedback"], draft=state["draft"])
            if is_revision and review
            else ""
        )
        chain = WRITE_PROMPT | llm | StrOutputParser()  # LCEL: prompt -> model -> str
        draft = await chain.ainvoke(
            {
                "question": state["question"],
                "source_list": render_source_list(sources) or "(no sources)",
                "notes": notes or "(no research notes)",
                "revision_block": revision_block,
                "today": today(),
            }
        )
        draft = sanitize_citations(strip_think(draft), len(sources))
        return {
            "draft": draft,
            "report": append_sources_section(draft, sources),
            "sources": sources,
            "revision": state.get("revision", 0) + (1 if is_revision else 0),
        }

    @timed_node("review")
    async def review(state: ResearchState) -> dict[str, Any]:
        _, notes = number_sources(state.get("findings", []))
        prompt = await REVIEW_PROMPT.ainvoke(
            {
                "question": state["question"],
                "notes": notes,
                "draft": state.get("draft", ""),
                "today": today(),
            }
        )
        result = await invoke_structured(
            llm, Review, prompt.to_messages(), retries=settings.structured_output_retries
        )
        if result is None:  # judge failed to produce valid JSON: fail open, don't loop
            return {"review": {"score": APPROVAL_SCORE, "feedback": "(review unavailable)"}}
        return {"review": {"score": result.score, "feedback": result.feedback}}

    def route_after_review(state: ResearchState) -> Literal["write", "__end__"]:
        # Conditional edge: a plain function of state that picks the next node.
        # Routing back to "write" creates a cycle; max_revisions guarantees it terminates.
        rv = state.get("review") or {"score": APPROVAL_SCORE}
        if rv["score"] >= APPROVAL_SCORE or state.get("revision", 0) >= settings.max_revisions:
            return END
        return "write"

    builder = StateGraph(ResearchState, context_schema=GraphContext, input_schema=ResearchInput)
    builder.add_node("plan", plan)
    # destinations: declares where this node's Command may go (used for graph drawing).
    builder.add_node("human_review", human_review, destinations=("plan", "research", END))
    builder.add_node("research", research)
    builder.add_node("write", write)
    builder.add_node("review", review)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "human_review")
    builder.add_edge("research", "write")  # waits for ALL parallel research tasks
    builder.add_edge("write", "review")
    builder.add_conditional_edges("review", route_after_review, ["write", END])
    return builder.compile(checkpointer=checkpointer, name="research_agent")

"""End-to-end graph tests with a scripted fake model (no Ollama)."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research_agent.graph.graph import build_graph, run_config
from research_agent.graph.state import GraphContext
from tests.fakes import ScriptedChatModel, tool_call

PLAN = {"sub_questions": ["What is RAG?", "Why does RAG reduce hallucination?", "extra", "x4"]}


def _graph(llm, kb_tool, settings):
    return build_graph(llm, [kb_tool], settings, checkpointer=InMemorySaver())


async def test_interrupt_then_edit_and_revision_loop(settings, kb_tool):
    llm = ScriptedChatModel(
        structured={
            "Plan": [PLAN],
            # first draft rejected by the judge, second accepted -> exercises the cycle
            "Review": [{"score": 2, "feedback": "cite more"}, {"score": 5, "feedback": ""}],
        }
    )
    graph = _graph(llm, kb_tool, settings)
    cfg = run_config("t1", settings)

    first = await graph.ainvoke({"question": "How does RAG work?"}, cfg, version="v2")
    assert first.interrupts, "graph should pause for plan review"
    payload = first.interrupts[0].value
    assert payload["type"] == "plan_review"
    assert payload["plan"] == PLAN["sub_questions"][:3]  # capped at max_subquestions

    snapshot = await graph.aget_state(cfg)
    assert snapshot.next == ("human_review",)

    resumed = await graph.ainvoke(
        Command(resume={"action": "edit", "sub_questions": ["What is RAG?", "Why RAG?"]}),
        cfg,
        version="v2",
    )
    state = resumed.value
    assert [f["sub_question"] for f in state["findings"]] == ["What is RAG?", "Why RAG?"]
    assert all(f["sources"] for f in state["findings"]), "citations come from tool artifacts"
    assert state["revision"] == 1 and state["review"]["score"] == 5
    assert "[99]" not in state["draft"], "out-of-range citation stripped"
    assert "[1]" in state["draft"]
    assert "## Sources" in state["report"] and "kb://" in state["report"]


async def test_auto_approve_runs_without_interrupt(settings, kb_tool):
    llm = ScriptedChatModel(structured={"Plan": [PLAN], "Review": [{"score": 4, "feedback": ""}]})
    graph = _graph(llm, kb_tool, settings)
    out = await graph.ainvoke(
        {"question": "q"},
        run_config("t2", settings),
        context=GraphContext(auto_approve=True),
        version="v2",
    )
    assert not out.interrupts
    assert len(out.value["findings"]) == 3
    assert out.value["revision"] == 0  # approved first time


async def test_reject_ends_run(settings, kb_tool):
    graph = _graph(ScriptedChatModel(structured={"Plan": [PLAN]}), kb_tool, settings)
    cfg = run_config("t3", settings)
    await graph.ainvoke({"question": "q"}, cfg)
    out = await graph.ainvoke(Command(resume={"action": "reject"}), cfg, version="v2")
    assert out.value["status"] == "rejected"
    assert "report" not in out.value


async def test_replan_loops_back_to_planner(settings, kb_tool):
    seen_feedback: list[str] = []

    def planner(msgs):
        seen_feedback.append(str(msgs[-1].content))
        return {"sub_questions": [f"v{len(seen_feedback)}"]}

    graph = _graph(ScriptedChatModel(structured={"Plan": planner}), kb_tool, settings)
    cfg = run_config("t4", settings)
    await graph.ainvoke({"question": "q"}, cfg)
    out = await graph.ainvoke(
        Command(resume={"action": "replan", "feedback": "focus on costs"}), cfg, version="v2"
    )
    assert out.interrupts and out.interrupts[0].value["plan"] == ["v2"]
    assert "focus on costs" in seen_feedback[-1]


async def test_structured_output_failures_degrade_gracefully(settings, kb_tool):
    llm = ScriptedChatModel(
        structured={
            "Plan": [ValueError("bad json"), ValueError("bad json again")],
            "Review": [ValueError("bad"), ValueError("bad")],
        }
    )
    graph = _graph(llm, kb_tool, settings)
    out = await graph.ainvoke(
        {"question": "What is RAG?"},
        run_config("t5", settings),
        context=GraphContext(auto_approve=True),
        version="v2",
    )
    assert out.value["plan"] == ["What is RAG?"]  # fallback plan = the question itself
    assert out.value["review"]["feedback"] == "(review unavailable)"
    assert out.value["report"]


async def test_researcher_call_cap_triggers_fallback_summary(settings, kb_tool):
    def looping(messages):  # a small model that never stops calling tools
        sys = " ".join(str(m.content) for m in messages[:1])
        if "research assistant" in sys:
            return tool_call("search_knowledge_base", query="again")
        if "factual notes" in sys:
            return AIMessage("Fallback notes from tool results.")
        return AIMessage("Report [1]")

    llm = ScriptedChatModel(
        responder=looping,
        structured={"Plan": [{"sub_questions": ["one"]}], "Review": [{"score": 5, "feedback": ""}]},
    )
    graph = _graph(llm, kb_tool, settings)
    out = await graph.ainvoke(
        {"question": "q"},
        run_config("t6", settings),
        context=GraphContext(auto_approve=True),
        version="v2",
    )
    (finding,) = out.value["findings"]
    assert finding["summary"] == "Fallback notes from tool results."
    researcher_calls = [c for c in llm.calls if "research assistant" in str(c[0].content)]
    assert len(researcher_calls) == settings.researcher_max_model_calls

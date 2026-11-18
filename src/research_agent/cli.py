"""Terminal client.

research-agent ask "How does RAG reduce hallucinations?"   # interactive plan approval
research-agent ask --auto-approve "..."                     # no pause
research-agent ask --resume <thread_id>                    # continue a paused thread
research-agent ingest                                       # index data/docs
research-agent show <thread_id>                             # print a thread's state
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from typing import Any

from langgraph.types import Command

from research_agent.config import Settings, get_settings
from research_agent.deps import AppRuntime, open_runtime
from research_agent.graph.state import PlanDecision
from research_agent.logging import configure_logging
from research_agent.rag.ingest import ingest_directory
from research_agent.streaming import stream_events, thread_summary

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def ask_decision(plan: list[str]) -> PlanDecision:
    print(f"\n{BOLD}Proposed plan:{RESET}")
    for i, q in enumerate(plan, 1):
        print(f"  {i}. {q}")
    while True:
        choice = input("\n[a]pprove  [e]dit  [r]eplan  [x] reject > ").strip().lower() or "a"
        if choice.startswith("a"):
            return PlanDecision(action="approve")
        if choice.startswith("x"):
            return PlanDecision(action="reject")
        if choice.startswith("r"):
            return PlanDecision(action="replan", feedback=input("Feedback for the planner: "))
        if choice.startswith("e"):
            print("Enter sub-questions, one per line; empty line to finish:")
            lines: list[str] = []
            while line := input("  > ").strip():
                lines.append(line)
            if lines:
                return PlanDecision(action="edit", sub_questions=lines)


async def run_stream(rt: AppRuntime, graph_input: Any, thread_id: str, auto: bool) -> dict:
    done: dict = {}
    streaming = False
    async for ev in stream_events(rt.graph, graph_input, thread_id, rt.settings, auto_approve=auto):
        kind, data = ev["event"], ev["data"]
        if kind == "token":
            if not streaming:
                print(f"\n{BOLD}--- report (streaming) ---{RESET}\n")
                streaming = True
            print(data["text"], end="", flush=True)
        elif kind == "update":
            streaming = False
            node = data.pop("node")
            if node == "research":
                for f in data.get("findings", []):
                    print(
                        f"{DIM}✓ researched: {f['sub_question']} ({len(f['sources'])} sources)"
                        f"{RESET}"
                    )
            elif node == "review":
                rv = data.get("review", {})
                print(f"\n{DIM}judge: {rv.get('score')}/5 {rv.get('feedback', '')[:200]}{RESET}")
            else:
                print(f"{DIM}▸ {node}{RESET}")
        elif kind == "done":
            done = data
    return done


async def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    async with open_runtime(settings) as rt:
        thread_id = args.resume or args.thread or uuid.uuid4().hex
        print(f"{DIM}thread {thread_id} · model {settings.chat_model}{RESET}")
        if args.resume:
            summary = await thread_summary(rt.graph, thread_id, settings)
        else:
            if not args.question:
                print("question required", file=sys.stderr)
                return 2
            summary = await run_stream(
                rt, {"question": args.question}, thread_id, args.auto_approve
            )
        while summary.get("status") == "awaiting_approval":
            decision = ask_decision(summary["interrupts"][0]["value"]["plan"])
            graph_input = Command(resume=decision.model_dump())
            summary = await run_stream(rt, graph_input, thread_id, False)

        print(f"\n\n{BOLD}=== {summary.get('status')} ==={RESET}")
        if summary.get("report"):
            print(summary["report"])
        return 0


async def cmd_show(args: argparse.Namespace, settings: Settings) -> int:
    async with open_runtime(settings) as rt:
        print(json.dumps(await thread_summary(rt.graph, args.thread_id, settings), indent=2))
    return 0


def cmd_ingest(settings: Settings) -> int:
    from research_agent.llm import get_embeddings
    from research_agent.rag.ingest import get_vectorstore

    report = ingest_directory(get_vectorstore(settings, get_embeddings(settings)), settings)
    print(
        f"ingested {len(report.files)} files, {report.chunks} chunks; skipped {len(report.skipped)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="show structured logs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ask = sub.add_parser("ask", help="research a question")
    p_ask.add_argument("question", nargs="?")
    p_ask.add_argument("--auto-approve", action="store_true")
    p_ask.add_argument("--thread", help="thread id for a new run")
    p_ask.add_argument("--resume", metavar="THREAD_ID", help="continue a paused thread")
    sub.add_parser("ingest", help="index the documents in DOCS_DIR")
    p_show = sub.add_parser("show", help="print a thread's state")
    p_show.add_argument("thread_id")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("INFO" if args.verbose else "WARNING", json=False)
    if args.cmd == "ingest":
        return cmd_ingest(settings)
    if args.cmd == "show":
        return asyncio.run(cmd_show(args, settings))
    try:
        return asyncio.run(cmd_ask(args, settings))
    except KeyboardInterrupt:
        print("\ninterrupted; resume later with --resume <thread_id>")
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Run the agent over evals/dataset.jsonl and score each report.

    uv run python evals/run_evals.py                 # KB-only, plan auto-approved
    uv run python evals/run_evals.py --web --limit 2
    uv run python evals/run_evals.py --judge-model llama3.2:3b   # different judge family

Scores:
  deterministic   citations valid / present, key-fact recall, expected KB source retrieved
  LLM-as-judge    faithfulness (report supported by research notes?) and relevance, 1-5
Caveat: by default the judge is the same small model that wrote the report, so expect
self-preference and leniency. A different (ideally stronger) judge model is more honest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.metrics import citation_check, key_fact_recall, source_hit  # noqa: E402
from research_agent.config import Settings  # noqa: E402
from research_agent.deps import open_runtime  # noqa: E402
from research_agent.graph.graph import number_sources, run_config  # noqa: E402
from research_agent.graph.state import GraphContext  # noqa: E402
from research_agent.llm import get_chat_model, invoke_structured  # noqa: E402
from research_agent.logging import configure_logging  # noqa: E402


class Judgement(BaseModel):
    # Reasoning BEFORE the score: a cheap way to make judge scores more consistent.
    reasoning: str = Field(description="1-3 sentences of justification")
    score: int = Field(ge=1, le=5)


FAITHFULNESS = (
    "You check a report for faithfulness to its research notes. 5 = every factual claim is "
    "supported by the notes; 3 = some claims unsupported; 1 = mostly unsupported or "
    "contradicted. Ignore style. Reply in JSON."
)
RELEVANCE = (
    "You check whether a report answers the question. 5 = fully and directly answers it; "
    "3 = partially; 1 = off-topic. Reply in JSON."
)


async def judge(llm, system: str, body: str) -> Judgement | None:
    return await invoke_structured(
        llm, Judgement, [SystemMessage(system), HumanMessage(body)], retries=2
    )


async def run(args: argparse.Namespace) -> int:
    settings = Settings(
        enable_web=args.web,
        log_level="WARNING",
        checkpoint_db=ROOT / "evals/results/checkpoints.sqlite",
    )
    configure_logging(settings.log_level, json=False)
    examples = [
        json.loads(line)
        for line in (ROOT / "evals/dataset.jsonl").read_text().splitlines()
        if line.strip()
    ][args.start :][: args.limit]
    judge_llm = get_chat_model(
        settings.model_copy(
            update={"chat_model": args.judge_model or settings.chat_model, "temperature": 0.0}
        )
    )
    rows = []
    async with open_runtime(settings) as rt:
        for ex in examples:
            t0 = time.perf_counter()
            out = await rt.graph.ainvoke(
                {"question": ex["question"]},
                run_config(f"eval-{ex['id']}-{uuid.uuid4().hex[:6]}", settings),
                context=GraphContext(auto_approve=True),
                version="v2",
            )
            latency = time.perf_counter() - t0
            state = out.value
            report, draft = state.get("report", ""), state.get("draft", "")
            sources = state.get("sources", [])
            _, notes = number_sources(state.get("findings", []))
            faith = await judge(
                judge_llm, FAITHFULNESS, f"Research notes:\n{notes}\n\nReport:\n{draft}"
            )
            rel = await judge(
                judge_llm, RELEVANCE, f"Question: {ex['question']}\n\nReport:\n{draft}"
            )
            row = {
                "id": ex["id"],
                "latency_s": round(latency, 1),
                **citation_check(draft, len(sources)),
                "key_fact_recall": round(key_fact_recall(report, ex.get("key_facts", [])), 2),
                "source_hit": source_hit(sources, ex.get("expected_sources", [])),
                "faithfulness": faith.score if faith else None,
                "faithfulness_why": faith.reasoning if faith else None,
                "relevance": rel.score if rel else None,
                "in_graph_review": (state.get("review") or {}).get("score"),
                "revisions": state.get("revision", 0),
            }
            rows.append(row)
            print(
                f"{row['id']:22s} {row['latency_s']:6.1f}s  cites={row['n_citations']:<2} "
                f"valid={row['citations_valid']!s:5} recall={row['key_fact_recall']:.2f} "
                f"src={row['source_hit']!s:5} faith={row['faithfulness']} rel={row['relevance']}",
                flush=True,
            )

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r[key] is not None]
        return round(statistics.mean(vals), 2) if vals else None

    summary = {
        "model": settings.chat_model,
        "judge_model": args.judge_model or settings.chat_model,
        "web": args.web,
        "n": len(rows),
        "citations_valid_rate": mean("citations_valid"),
        "has_citations_rate": mean("has_citations"),
        "key_fact_recall": mean("key_fact_recall"),
        "source_hit_rate": mean("source_hit"),
        "faithfulness": mean("faithfulness"),
        "relevance": mean("relevance"),
        "latency_s": mean("latency_s"),
    }
    out_dir = ROOT / "evals/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\nSUMMARY", json.dumps(summary, indent=2))
    print(f"saved {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--web", action="store_true", help="enable web tools (slower, less stable)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0, help="skip the first N examples")
    p.add_argument("--judge-model", default=None)
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

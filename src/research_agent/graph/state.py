"""Graph state, context and structured-output schemas.

LangGraph state is a TypedDict. Each node returns a *partial* update; for every key the
graph merges it into the current state with that key's *reducer*. Default reducer =
overwrite. `Annotated[list, operator.add]` = append, which is what lets several parallel
researchers each return `{"findings": [one]}` without clobbering each other.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from research_agent.schemas import Source


class Finding(TypedDict):
    sub_question: str
    summary: str
    sources: list[Source]


class ReviewResult(TypedDict):
    score: int
    feedback: str


class ResearchInput(TypedDict):
    """What a caller sends to start a run (the graph's input schema)."""

    question: str


class ResearchState(TypedDict):
    question: str
    plan: NotRequired[list[str]]
    plan_feedback: NotRequired[str | None]
    findings: Annotated[list[Finding], operator.add]  # reducer: parallel-safe append
    sources: NotRequired[list[Source]]
    draft: NotRequired[str]
    report: NotRequired[str]
    review: NotRequired[ReviewResult | None]
    revision: NotRequired[int]
    status: NotRequired[Literal["rejected", "done"]]


class ResearchTask(TypedDict):
    """Payload of each `Send("research", ...)`: a researcher sees only its sub-question."""

    question: str
    sub_question: str


@dataclass
class GraphContext:
    """Per-run *context* (LangGraph runtime context): config that isn't state.

    Pass it as `graph.astream(..., context=GraphContext(auto_approve=True))`.
    """

    auto_approve: bool = False


# ---- Structured outputs (Pydantic -> JSON schema -> Ollama constrained decoding) ----


class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="2-4 focused, self-contained sub-questions that together answer the question"
    )


class Review(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor ... 5=excellent")
    feedback: str = Field(description="Concrete, actionable fixes. Empty if none needed.")


class PlanDecision(BaseModel):
    """The human's answer to the plan-review interrupt (validated on resume)."""

    action: Literal["approve", "edit", "replan", "reject"] = "approve"
    sub_questions: list[str] | None = Field(
        None, description="Replacement sub-questions when action='edit'"
    )
    feedback: str | None = Field(None, description="Guidance for the planner when action='replan'")

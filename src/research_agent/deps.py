"""Composition root: wires settings -> models -> vector store -> tools -> graph + checkpointer.

API, CLI, MCP server and evals all open the same runtime through `open_runtime()`, so there
is exactly one place that decides which concrete implementations are used.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings, get_settings
from research_agent.graph.graph import build_graph
from research_agent.llm import get_chat_model, get_embeddings
from research_agent.rag.ingest import get_vectorstore
from research_agent.tools.kb import make_kb_tool
from research_agent.tools.web import make_web_tools


@dataclass
class AppRuntime:
    settings: Settings
    graph: CompiledStateGraph
    vectorstore: VectorStore
    llm: BaseChatModel


def build_tools(settings: Settings, llm: BaseChatModel, vectorstore: VectorStore) -> list[BaseTool]:
    tools: list[BaseTool] = [
        make_kb_tool(
            vectorstore,
            k=settings.kb_top_k,
            grader=llm if settings.kb_grade else None,
            min_score=settings.kb_min_score,
        )
    ]
    if settings.enable_web:
        tools.extend(make_web_tools(settings))
    return tools


@asynccontextmanager
async def open_runtime(settings: Settings) -> AsyncIterator[AppRuntime]:
    llm = get_chat_model(settings)
    vectorstore = get_vectorstore(settings, get_embeddings(settings))
    tools = build_tools(settings, llm, vectorstore)

    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    # A checkpointer saves state after every super-step, keyed by thread_id. SQLite keeps
    # threads across restarts (resume a run tomorrow). Production swap: Postgres
    # (`langgraph-checkpoint-postgres`, AsyncPostgresSaver) for multi-instance deployments.
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as checkpointer:
        graph = build_graph(llm, tools, settings, checkpointer=checkpointer)
        yield AppRuntime(settings=settings, graph=graph, vectorstore=vectorstore, llm=llm)


def make_graph() -> CompiledStateGraph:
    """Factory for `langgraph dev` / Studio (see langgraph.json). The LangGraph server
    supplies its own checkpointer, so none is attached here."""
    settings = get_settings()
    llm = get_chat_model(settings)
    vectorstore = get_vectorstore(settings, get_embeddings(settings))
    return build_graph(llm, build_tools(settings, llm, vectorstore), settings)

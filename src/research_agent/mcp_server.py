"""MCP server: exposes the knowledge base and the research agent as MCP tools.

MCP (Model Context Protocol) is a standard way for AI apps (Claude Desktop/Code, IDEs,
other agents) to discover and call tools. Here our LangGraph agent *becomes* a tool.

Run (stdio transport, what desktop clients spawn):  uv run python -m research_agent.mcp_server
Register with Claude Code:  claude mcp add research-agent -- uv --directory <repo> run \
    python -m research_agent.mcp_server

Uses the mcp 2.x API (`MCPServer`; v1 called it `FastMCP`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.mcpserver import Context, MCPServer

from research_agent.config import get_settings
from research_agent.deps import AppRuntime, open_runtime
from research_agent.graph.graph import run_config
from research_agent.graph.state import GraphContext
from research_agent.logging import configure_logging
from research_agent.tools.kb import format_docs


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[AppRuntime]:
    settings = get_settings()
    # stdio transport: stdout carries the protocol, so logs must go to stderr (they do).
    configure_logging(settings.log_level, settings.log_json)
    async with open_runtime(settings) as runtime:
        yield runtime


mcp = MCPServer(
    "research-agent",
    instructions="Local research agent: knowledge-base search and multi-step web research.",
    lifespan=lifespan,
)


def _runtime(ctx: Context) -> AppRuntime:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def search_knowledge_base(query: str, ctx: Context, k: int = 4) -> str:
    """Semantic search over the local document knowledge base. Returns cited passages."""
    docs = await _runtime(ctx).vectorstore.asimilarity_search(query, k=max(1, min(k, 10)))
    return format_docs(docs)


@mcp.tool()
async def research(question: str, ctx: Context) -> str:
    """Run the full research agent (plan -> research -> write -> review) and return a
    Markdown report with citations. Slow (minutes) on a local model. Plan auto-approved."""
    rt = _runtime(ctx)
    thread_id = f"mcp-{uuid.uuid4().hex[:12]}"
    await ctx.info(f"research started (thread {thread_id})")
    result = await rt.graph.ainvoke(
        {"question": question},
        run_config(thread_id, rt.settings),
        context=GraphContext(auto_approve=True),
        version="v2",
    )
    return result.value.get("report") or "No report was produced."


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

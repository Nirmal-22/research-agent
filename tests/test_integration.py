"""Integration tests against a real Ollama server. Run with `make test-integration`."""

from __future__ import annotations

import sys

import pytest

from research_agent.config import Settings

pytestmark = pytest.mark.integration


async def test_ollama_tool_calling_and_structured_output():
    from langchain_core.tools import tool

    from research_agent.graph.state import Plan
    from research_agent.llm import get_chat_model, invoke_structured

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    llm = get_chat_model(Settings())
    msg = await llm.bind_tools([add]).ainvoke("What is 17 + 25? Use the tool.")
    assert msg.tool_calls and msg.tool_calls[0]["name"] == "add"

    from langchain_core.messages import HumanMessage

    plan = await invoke_structured(
        llm, Plan, [HumanMessage("Split into 2 sub-questions: What is RAG?")]
    )
    assert plan is not None and plan.sub_questions


async def test_mcp_server_over_stdio():
    """Spawns the MCP server as a subprocess and calls a tool, like a desktop client would."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "research_agent.mcp_server"], env=None
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        assert {"search_knowledge_base", "research"} <= names
        result = await session.call_tool("search_knowledge_base", {"query": "What is RAG?"})
        text = result.content[0].text
        assert "source:" in text

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore

from research_agent.config import Settings
from research_agent.tools.kb import make_kb_tool


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    return Settings(
        _env_file=None,  # type: ignore[call-arg]  # ignore any developer .env
        docs_dir=docs,
        chroma_dir=tmp_path / "chroma",
        checkpoint_db=tmp_path / "cp.sqlite",
        enable_web=False,
        max_subquestions=3,
        researcher_max_model_calls=3,
        max_revisions=1,
        structured_output_retries=1,
        log_json=False,
        search_retry_delay_s=0.0,
    )


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def kb_tool(embeddings: DeterministicFakeEmbedding) -> BaseTool:
    vs = InMemoryVectorStore(embeddings)
    vs.add_documents(
        [
            Document("RAG retrieves passages before generating.", metadata={"source": "rag.md"}),
            Document("LangGraph models agents as graphs.", metadata={"source": "langgraph.md"}),
        ]
    )
    return make_kb_tool(vs, k=2)

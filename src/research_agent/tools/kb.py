"""Knowledge-base tool: semantic search over the local vector store, with relevance grading.

Lesson learned on real data: a vector store ALWAYS returns top-k, even for questions the
documents don't cover, and similarity scores of relevant vs irrelevant chunks overlap too
much for a fixed threshold. So after retrieval we let the LLM *grade* the passages (the
"corrective RAG" pattern) and drop the irrelevant ones. That keeps junk out of the notes and
out of the citations, and tells the agent to go to the web instead.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from research_agent.llm import invoke_structured
from research_agent.logging import get_logger
from research_agent.schemas import Source

log = get_logger(__name__)

NO_RESULTS = "No relevant passages in the knowledge base. Use web_search instead."


class RelevantPassages(BaseModel):
    relevant: list[int] = Field(description="Numbers of the passages that help answer the query")


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return NO_RESULTS
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


def docs_to_sources(docs: list[Document]) -> list[Source]:
    return [
        Source(
            title=str(d.metadata.get("source", "kb")),
            url=f"kb://{d.metadata.get('source')}",
            kind="kb",
        )
        for d in docs
    ]


async def grade_docs(grader: BaseChatModel, query: str, docs: list[Document]) -> list[Document]:
    """One structured call grades all passages at once (cheaper than one call per doc)."""
    if not docs:
        return docs
    passages = "\n\n".join(f"[{i}] {d.page_content[:600]}" for i, d in enumerate(docs, 1))
    result = await invoke_structured(
        grader,
        RelevantPassages,
        [
            SystemMessage(
                "You grade search results. List the numbers of passages that contain "
                "information directly useful for answering the query. Return an empty list "
                "if none are. Reply in JSON."
            ),
            HumanMessage(f"Query: {query}\n\nPassages:\n{passages}"),
        ],
        retries=1,
    )
    if result is None:  # grader failed: fail open, keep everything
        return docs
    keep = [docs[i - 1] for i in dict.fromkeys(result.relevant) if 1 <= i <= len(docs)]
    log.info("kb.graded", query=query[:80], retrieved=len(docs), kept=len(keep))
    return keep


def make_kb_tool(
    vectorstore: VectorStore,
    k: int = 4,
    *,
    grader: BaseChatModel | None = None,
    min_score: float | None = None,
) -> BaseTool:
    """`grader=None` disables LLM grading; `min_score` (if set) drops obvious junk first."""

    @tool(response_format="content_and_artifact")
    async def search_knowledge_base(query: str) -> tuple[str, list[Source]]:
        """Search the local knowledge base of documents. Use it first for any topic."""
        # A *retriever* / similarity search: query -> nearest chunks by embedding similarity.
        # Exposing it as a tool = "agentic RAG": the agent decides when and what to retrieve.
        if min_score is None:
            docs = await vectorstore.asimilarity_search(query, k=k)
        else:
            scored = await vectorstore.asimilarity_search_with_relevance_scores(query, k=k)
            docs = [d for d, score in scored if score >= min_score]
        if grader is not None:
            docs = await grade_docs(grader, query, docs)
        return format_docs(docs), docs_to_sources(docs)

    return search_knowledge_base

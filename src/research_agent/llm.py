"""Model factories + a robust structured-output helper.

Everything that needs a model receives it as an argument (dependency injection), so tests
can pass fake chat models / embeddings and never touch Ollama.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

from research_agent.config import Settings
from research_agent.logging import get_logger

log = get_logger(__name__)


def get_chat_model(settings: Settings) -> BaseChatModel:
    # A LangChain *chat model* is a Runnable: messages in -> AIMessage out. Same interface
    # for every provider, so swapping Ollama for another backend is a one-line change.
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        # reasoning=False asks Ollama not to "think" (no <think> block): faster, cleaner tool
        # calls/JSON. Caveat learned the hard way: `qwen3:4b` now points at the
        # thinking-ONLY 2507 build, which ignores this flag and thinks for thousands of
        # tokens. That's why the default is the non-thinking `qwen3:4b-instruct`.
        reasoning=False,
        client_kwargs={"timeout": settings.request_timeout_s},
    )


class PrefixedEmbeddings(Embeddings):
    """Adds task prefixes for *asymmetric* embedding models.

    nomic-embed-text was trained with "search_query: " / "search_document: " prefixes;
    using them measurably improves retrieval (see the README's RAG notes). Changing them
    changes every vector, so re-run `make ingest` afterwards.
    """

    def __init__(self, inner: Embeddings, query_prefix: str, document_prefix: str) -> None:
        self.inner, self.qp, self.dp = inner, query_prefix, document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.dp + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.qp + text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.inner.aembed_documents([self.dp + t for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await self.inner.aembed_query(self.qp + text)


def get_embeddings(settings: Settings) -> Embeddings:
    # Embeddings map text -> vector. RAG retrieval is nearest-neighbour search over these.
    base = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    if "nomic-embed" in settings.embed_model:
        return PrefixedEmbeddings(base, "search_query: ", "search_document: ")
    return base


async def invoke_structured[T: BaseModel](
    llm: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    *,
    retries: int = 2,
) -> T | None:
    """Ask the model for a `schema` instance; retry on bad output; return None on failure.

    `with_structured_output(method="json_schema")` passes the JSON schema to Ollama's
    constrained decoding (`format=`), which is far more reliable for small models than
    asking for JSON in the prompt or using tool calls. Still, it can fail (truncation,
    validation errors), so callers must handle `None` with a sensible fallback.
    """
    structured = llm.with_structured_output(schema, method="json_schema")
    for attempt in range(retries + 1):
        try:
            result = await structured.ainvoke(list(messages))
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            log.warning("structured.empty", schema=schema.__name__, attempt=attempt)
        except Exception as exc:
            log.warning(
                "structured.failed", schema=schema.__name__, attempt=attempt, error=repr(exc)[:300]
            )
    return None


async def ping_ollama(settings: Settings) -> dict[str, object]:
    """Readiness check: Ollama reachable and the configured models are pulled."""
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=3) as client:
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        names = {m["name"] for m in resp.json().get("models", [])}

    def present(model: str) -> bool:
        return model in names or f"{model}:latest" in names

    return {
        "ollama": "ok",
        "chat_model": settings.chat_model,
        "chat_model_present": present(settings.chat_model),
        "embed_model": settings.embed_model,
        "embed_model_present": present(settings.embed_model),
    }

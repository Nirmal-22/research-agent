# LEARNING.md: concept → where it lives

Read these roughly in order. Each file has a short comment where the concept first appears.

## LangChain building blocks

| Concept | Where | What to look at |
|---|---|---|
| Chat models as Runnables | [llm.py](src/research_agent/llm.py) | `get_chat_model`: one interface for every provider; `reasoning=False` and the thinking-model caveat |
| Prompts | [graph/prompts.py](src/research_agent/graph/prompts.py) | `ChatPromptTemplate`, short explicit rules for small models, `GROUNDING` (temporal grounding) |
| LCEL / runnables | [graph/graph.py](src/research_agent/graph/graph.py) `write`, [graph/researcher.py](src/research_agent/graph/researcher.py) fallback | `WRITE_PROMPT \| llm \| StrOutputParser()` |
| Structured output | [llm.py](src/research_agent/llm.py) `invoke_structured`, [graph/state.py](src/research_agent/graph/state.py) `Plan` / `Review` | `with_structured_output(method="json_schema")` → Ollama constrained decoding; capped retries; `None` → fallback |
| Tool calling | [tools/web.py](src/research_agent/tools/web.py), [tools/kb.py](src/research_agent/tools/kb.py) | `@tool`; the docstring is the description; tools never raise; `response_format="content_and_artifact"` carries citations without sending them to the LLM |
| Agents + middleware | [graph/researcher.py](src/research_agent/graph/researcher.py) | `create_agent` (ReAct loop), `ModelCallLimitMiddleware` (iteration cap), `@dynamic_prompt` |

## RAG

| Concept | Where |
|---|---|
| Loaders (md/txt/pdf) | [rag/ingest.py](src/research_agent/rag/ingest.py) `load_file` |
| Text splitters | `split_documents`: `RecursiveCharacterTextSplitter`, chunk size and overlap |
| Embeddings (asymmetric, task prefixes) | [llm.py](src/research_agent/llm.py) `PrefixedEmbeddings` |
| Vector store (Chroma, persisted) | `get_vectorstore`; idempotent re-ingest with stable chunk ids |
| Retriever as a tool (agentic RAG) | [tools/kb.py](src/research_agent/tools/kb.py) `make_kb_tool` |
| Retrieval grading (corrective RAG) | [tools/kb.py](src/research_agent/tools/kb.py) `grade_docs`; why a score threshold alone fails |

## LangGraph

| Concept | Where |
|---|---|
| State as TypedDict, reducers (`operator.add`) | [graph/state.py](src/research_agent/graph/state.py) `ResearchState.findings` |
| Input schema / runtime context | `ResearchInput`, `GraphContext` (`auto_approve`), `Runtime[GraphContext]` in `human_review` |
| Nodes, edges, conditional edges | [graph/graph.py](src/research_agent/graph/graph.py) `build_graph`, `route_after_review` |
| Cycles (bounded) | review → write loop, capped by `MAX_REVISIONS` |
| `Command` (update + goto from a node) | `human_review` returns `Command(goto=...)` |
| `Send` fan-out (map-reduce) | `human_review` → `Send("research", ResearchTask(...))` per sub-question; joined at `write` |
| Subgraphs | the `create_agent` researcher, invoked inside the `research` node |
| Checkpointing and threads | [deps.py](src/research_agent/deps.py) `AsyncSqliteSaver`; `run_config(thread_id)`; Postgres is the production swap |
| Interrupt / human-in-the-loop | `interrupt(..., response_schema=PlanDecision)` + `Command(resume=...)` in [api/main.py](src/research_agent/api/main.py) and [cli.py](src/research_agent/cli.py) |
| Streaming | [streaming.py](src/research_agent/streaming.py): `astream(stream_mode=["updates","messages"], version="v2")` → SSE |

## Serving, evaluation, ops

| Concept | Where |
|---|---|
| FastAPI + SSE, per-thread locking, readiness probe | [api/main.py](src/research_agent/api/main.py) |
| Dependency injection / composition root | [deps.py](src/research_agent/deps.py), `create_app(runtime_factory=...)` |
| MCP server (mcp 2.x `MCPServer`, lifespan context) | [mcp_server.py](src/research_agent/mcp_server.py) |
| Evals: deterministic checks + LLM-as-judge | [evals/run_evals.py](evals/run_evals.py), [evals/metrics.py](evals/metrics.py) |
| In-graph LLM-as-judge | `review` node in [graph/graph.py](src/research_agent/graph/graph.py) |
| Guardrails | `sanitize_citations`, SSRF guard `_is_public_http_url`, output caps in `fetch_url` |
| Observability | [logging.py](src/research_agent/logging.py): structlog JSON, `timed_node`, `thread_id` bound per request |
| Testing LLM apps without an LLM | [tests/fakes.py](tests/fakes.py) `ScriptedChatModel` (tool calls, structured output, token streaming) |

## Exercises

1. Set `CHAT_MODEL=llama3.2:3b`, run `make evals`, and compare the results with qwen.
2. Run `uv run python evals/run_evals.py --judge-model llama3.2:3b` (after `ollama pull llama3.2:3b`) and see how the scores move when the judge isn't grading its own model family.
3. Set `KB_GRADE=false` and ask an off-topic question. Watch irrelevant `kb://` citations come back.
4. Add a `"custom"` stream event from the researcher with `get_stream_writer()` that shows each tool call live in the UI.

## Next modules

- **Tracing**: LangSmith (set `LANGSMITH_TRACING=true`), or self-hosted **Langfuse** through its LangChain callback handler
- **Vector DBs**: pgvector (Postgres) or Qdrant instead of Chroma; move the checkpointer to Postgres in the same step
- **Redis**: an LLM response cache (`set_llm_cache`) and embedding cache
- **Guardrails**: input/output validation, prompt-injection defence for fetched pages, `PIIMiddleware`
- **Reranking**: a cross-encoder (e.g. `bge-reranker`) after retrieval
- **Hybrid search**: BM25 + vectors with reciprocal rank fusion
- **Prompt versioning**: prompts in files or a registry, with each eval run tagged by prompt version
- **vLLM serving**: OpenAI-compatible endpoint, continuous batching; compare throughput with Ollama
- **DSPy**: optimize the planner/writer prompts against the eval set
- **OpenTelemetry**: spans per node, exported to Jaeger or Grafana Tempo
- **CI with GitHub Actions**: `make lint test` on every push; evals on a self-hosted runner that has Ollama

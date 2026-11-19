# research-agent — CLAUDE.md

## Purpose
A learning project for **AI engineering**: LangChain, LangGraph, RAG, agents, evals, serving.
The user is learning the agentic / AI-engineering side. **Don't make it a Linux or DevOps lesson.**
Handle environment and setup chores directly, and keep explanations focused on the AI concepts.

Write production-style code: typed, configurable, tested, observable, with clear module boundaries.
Add a short comment where a LangChain/LangGraph concept first appears, so the code also teaches.

## Hard constraints
- **No paid API keys.** Everything must run locally and free.
  - LLM and embeddings come from **Ollama** (local).
  - Web search uses **DuckDuckGo (`ddgs`)**, which needs no key.
  - LangSmith tracing is optional. Leave it off by default and don't require it.
- **Use a light model.** Learning matters more than quality. Default chat model: a ~4B model with
  decent tool calling: **`qwen3:4b-instruct`** (set `reasoning=False` on ChatOllama). Do NOT use
  plain `qwen3:4b`: that tag is now the thinking-only 2507 build, which ignores `reasoning=False`
  (~3k thinking tokens, 200 s per tool call). Fallback: `llama3.2:3b`. Embeddings: `nomic-embed-text`.
  The model is configurable through env vars (`CHAT_MODEL`, `EMBED_MODEL`).
  Small models are flaky at tool calling and JSON output. Treat that as a lesson: use
  `with_structured_output(method="json_schema")`, cap retries and iterations, and fail gracefully.
- Docker Desktop's WSL integration is **off**, so `docker` doesn't work in WSL. The app must run
  without Docker. You can still write a Dockerfile and docker-compose.yml, but label them as untested.

## Environment facts (WSL Ubuntu-24.04)
- Python 3.12.10 (pyenv), Node 24. `sudo` needs a password, so install everything user-level.
- GPU: RTX 5070 Laptop, 8 GB VRAM. WSL RAM is about 7 GB.
- `gh` is logged in as **Nirmal-22**. Remote: https://github.com/Nirmal-22/research-agent (private).
- **Ollama runs as the Windows app** (v0.40.16, installed with `winget install Ollama.Ollama`),
  with `qwen3:4b-instruct` + `nomic-embed-text` pulled there. WSL reaches it on
  `localhost:11434` (mirrored networking). **Never run `ollama serve` inside WSL**: the model
  server took 5.4 GB of WSL's 7 GB RAM and the OOM killer crashed the session repeatedly. On
  Windows it uses about 2 GB. `make ollama` (→ `scripts/ollama-up.sh`) starts the Windows
  server if it's down. Starting the tray app from WSL does not bring the server up; the script
  uses `Start-Process ollama.exe serve`. The old Linux copy (`~/.local/ollama`, `~/.ollama`)
  was deleted on 2026-11-19; don't reinstall it.
- The PC has 14.7 GB RAM in total and Windows is often down to 1-2 GB free, so keep runs
  modest: evals in chunks (`--start N --limit 2`), no parallel heavy jobs.
- A dead localhost port in WSL *hangs* rather than refusing, so always pass `-m` to curl.
  Never `pkill -f <pattern>` that also matches your own shell command. Background processes
  started from a Claude Code session (uvicorn) die when the session ends.
- **uv** is installed with `pip install --user uv` (→ `~/.local/bin/uv`). Use `uv sync` and
  `uv run`. If uv is missing, reinstall it the same way.

## Pinned stack (latest versions as of 2026-11, already in pyproject.toml)
langchain 1.5, langchain-core 1.8, langgraph 1.2, langgraph-checkpoint-sqlite 3.1,
langchain-ollama 1.1, langchain-chroma 1.1 / chromadb 1.5, ddgs 9.16, trafilatura 2.3,
fastapi 0.162, sse-starlette 3.7, pydantic-settings 2.16, structlog, mcp 2.3, pytest 9 / pytest-asyncio 1.5, ruff.
These are newer than the model's training data. **After `uv sync`, check the real APIs before
writing code that uses them** (for example `python -c "import inspect, langchain.agents as a; print(inspect.signature(a.create_agent))"`).
This applies especially to `langchain.agents.create_agent`, `langgraph.types.interrupt/Command/Send`,
the AsyncSqliteSaver import path, and the mcp 2.x server API.

## Architecture (planned)
```
src/research_agent/
  config.py           pydantic-settings (env / .env)
  logging.py          structlog JSON logs
  llm.py              ChatOllama / OllamaEmbeddings factory (dependency-injected so tests can use fakes)
  tools/web.py        web_search (ddgs), fetch_url (httpx + trafilatura), with timeouts and size caps
  tools/kb.py         search_knowledge_base retriever tool
  rag/ingest.py       load (md/txt/pdf) → split → embed → Chroma (persisted in ./data/chroma)
  graph/state.py      TypedDict state with reducers (operator.add for findings)
  graph/prompts.py
  graph/researcher.py subgraph: tool-calling agent per sub-question, iteration-capped
  graph/graph.py      plan → human_review(interrupt) → fan-out research via Send → write → review
                      (LLM-as-judge) → conditional loop back to write (max N revisions) → END
  api/main.py         FastAPI: POST /research (SSE stream), POST /research/{thread_id}/resume,
                      GET /research/{thread_id}, POST /kb/ingest, /health, /ready (pings Ollama);
                      also serves a minimal static HTML UI to watch the stream and approve the plan
  mcp_server.py       exposes search_knowledge_base + research as MCP tools
  cli.py              terminal client
data/docs/            sample documents for RAG
evals/                small dataset + run_evals.py (LLM-as-judge faithfulness/relevance, citation checks)
tests/                unit tests with fake chat models (no Ollama needed);
                      `@pytest.mark.integration` tests use real Ollama
langgraph.json        for `langgraph dev` / Studio
Makefile, .env.example, Dockerfile + docker-compose.yml (untested), README.md, LEARNING.md
```
Checkpointer: AsyncSqliteSaver locally, which keeps threads across restarts. Note Postgres as the production swap.
Streaming: `graph.astream(stream_mode=["updates", "messages"])` → SSE events.

## LEARNING.md should map each concept to the file that teaches it
LCEL/runnables, prompts, structured output, tool calling, RAG (loaders, splitters, embeddings, vector store,
retriever), LangGraph state/reducers, conditional edges, cycles, Send fan-out, subgraphs, checkpointing and
threads, interrupt/human-in-the-loop, streaming, evals/LLM-as-judge, MCP, observability.
Then a "next modules" list: LangSmith or self-hosted Langfuse tracing, pgvector or Qdrant, Redis caching,
guardrails, reranking, hybrid search, prompt versioning, vLLM serving, DSPy, OpenTelemetry, CI with GitHub Actions.

## Status (2026-11-19)
Built and verified against real Ollama: every module in the architecture above, plus README.md
(including "lessons learned") and LEARNING.md. `make test` = 24 unit tests (no Ollama), `make
test-integration` = 2 (real Ollama + MCP over stdio). Verified end to end: CLI (KB-only and web),
API SSE interrupt → server restart → resume with an edited plan, and MCP stdio.
Evals ran to completion on 2026-11-18 (6/6, results table in README; self-judged, so lenient).
Ollama moved from WSL to the Windows app on 2026-11-19 after repeated OOM crashes; one CLI run
verified against it.
API changes found while building (they differ from older docs): mcp 2.x uses
`mcp.server.mcpserver.MCPServer` (not FastMCP); `interrupt(value, response_schema=Model)`
validates resume values; `astream(..., version="v2")` yields typed `{type, ns, data}` parts;
`ainvoke(version="v2")` returns `GraphOutput(.value, .interrupts)`; no langchain-community, so
loaders are hand-written.
Remaining ideas: judge the evals with a different model family; harder eval questions; optional
LangGraph Studio (`langgraph-cli[inmem]`).

## Commands
`make install` · `make ollama` · `make ingest` · `make dev` (API on :8000) · `make test` · `make evals` · `make lint`

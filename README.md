# research-agent

A local, free, production-style **research agent** built with LangChain v1 + LangGraph:
it plans sub-questions, **pauses for human approval**, fans out parallel tool-using
researchers (local RAG knowledge base + DuckDuckGo web search), writes a cited Markdown
report, and has an **LLM judge** send it back for revision when it's weak. It's served over
a streaming FastAPI API, a CLI and an MCP server.

Everything runs on your machine: **Ollama** (`qwen3:4b-instruct` + `nomic-embed-text`),
Chroma, SQLite. No API keys.

> It's a learning project. [LEARNING.md](LEARNING.md) maps each AI-engineering concept to the
> file that demonstrates it.

```
START → plan ─→ human_review ──(interrupt: approve / edit / replan / reject)
          ▲          │ replan                         │ approve/edit
          └──────────┘                                ▼
                               research × N   (Send fan-out; each = create_agent subgraph
                                    │          with search_knowledge_base / web_search / fetch_url)
                                    ▼
                                  write  ◄──────────┐  score < 4 and revisions left
                                    │               │
                                    ▼               │
                                  review (LLM-as-judge) ──→ END
```

## Quickstart

```bash
make install          # uv sync
make ollama           # start Ollama if needed + pull qwen3:4b-instruct and nomic-embed-text
make ingest           # index data/docs/*.md|txt|pdf into Chroma (data/chroma)
uv run research-agent ask "How does RAG reduce hallucinations?"     # interactive plan approval
uv run research-agent ask --auto-approve "What is the Model Context Protocol?"
make dev              # API + UI → http://localhost:8000
make test             # 24 unit tests, no Ollama needed
make test-integration # real Ollama: tool calling, structured output, MCP over stdio
make evals            # eval suite (LLM-as-judge + deterministic checks)
```

Ollama without sudo: the binary lives in `~/.local/ollama/bin/ollama` (see CLAUDE.md).
`make ollama` starts it detached if it isn't already running.

## API

| Method | Path | What |
|---|---|---|
| POST | `/research` `{question, auto_approve?, thread_id?}` | Starts a run. **SSE stream**: `thread`, `update` (per node), `token` (report tokens), `interrupt` (plan to approve), `done`, `error` |
| POST | `/research/{thread_id}/resume` `{action: approve\|edit\|replan\|reject, sub_questions?, feedback?}` | Answers the plan review. SSE stream continues the run |
| GET | `/research/{thread_id}` | Thread state from the checkpointer (works across restarts) |
| POST | `/kb/ingest` | Re-index `DOCS_DIR` |
| GET | `/health`, `/ready` | Liveness / readiness (pings Ollama, checks the models are pulled) |
| GET | `/` | Minimal UI: watch the stream, edit/approve the plan |

```bash
curl -N -X POST localhost:8000/research -H 'content-type: application/json' \
  -d '{"question": "What is LangGraph?"}'            # ... event: interrupt ... status awaiting_approval
curl -N -X POST localhost:8000/research/<thread_id>/resume -H 'content-type: application/json' \
  -d '{"action": "approve"}'
```

Threads are checkpointed in SQLite (`data/checkpoints.sqlite`), so you can stop the server
while a run is waiting for approval and resume it after a restart.

## MCP

`uv run python -m research_agent.mcp_server` serves two tools over stdio:
`search_knowledge_base(query, k)` and `research(question)`. To add it to Claude Code:

```bash
claude mcp add research-agent -- uv --directory "$(pwd)" run python -m research_agent.mcp_server
```

## Configuration

All settings are env vars or `.env` entries (see [.env.example](.env.example) and
[config.py](src/research_agent/config.py)). The main ones:

| Var | Default | Notes |
|---|---|---|
| `CHAT_MODEL` | `qwen3:4b-instruct` | Fallback: `llama3.2:3b` |
| `EMBED_MODEL` | `nomic-embed-text` | Task prefixes added automatically |
| `ENABLE_WEB` | `true` | `false` = knowledge base only (deterministic, offline) |
| `MAX_SUBQUESTIONS` / `RESEARCHER_MAX_MODEL_CALLS` / `MAX_REVISIONS` | 3 / 4 / 1 | Budgets: small models loop |
| `KB_GRADE` | `true` | LLM grades retrieved chunks (corrective RAG) |

## Evals

`make evals` runs [evals/dataset.jsonl](evals/dataset.jsonl) end to end with the plan
auto-approved and the web off, then scores each report:
- **deterministic**: citation markers valid and present, key-fact recall, expected KB file retrieved
- **LLM-as-judge**: faithfulness to the research notes, and relevance to the question (1-5, reasoning-before-score)

Results land in `evals/results/<timestamp>.json`. Use `--judge-model` to judge with a
different model family (this reduces self-preference), and `--web` to include web tools.
On a small machine, run it in chunks with `--start N --limit 2` (each example takes ~30-60 s).

## Lessons learned building this with a 4B model

These are real failures found while running against Ollama. Each one is fixed in the code:

1. **The `qwen3:4b` tag is now thinking-only.** It ignores `reasoning=False` and spent about
   3,000 tokens deliberating over "say hi in 5 words" (206 s for one tool call). Switching to
   `qwen3:4b-instruct` brought that to about 2 s. → [llm.py](src/research_agent/llm.py)
2. **Vector search always returns top-k.** Irrelevant chunks became fake citations, and their
   similarity scores overlapped with relevant ones (0.61 vs 0.62), so no fixed threshold could
   separate them. Fix: an LLM relevance grader after retrieval. → [tools/kb.py](src/research_agent/tools/kb.py)
3. **The agent gave up instead of searching the web.** It treated "no KB results" as the
   answer. Fix: the KB tool's empty result now says "Use web_search instead", and the prompt
   has an explicit MUST rule.
4. **Parallel researchers got rate-limited by DuckDuckGo.** Every backend returned "No results
   found." Fix: a semaphore that serializes searches, plus retry with backoff. → [tools/web.py](src/research_agent/tools/web.py)
5. **Knowledge-cutoff skepticism.** The model called real 2025-26 sources "speculative" and
   flagged URLs as "from the future". Fix: put today's date in the prompts with a "trust the
   sources" rule, using `dynamic_prompt` middleware. → [graph/prompts.py](src/research_agent/graph/prompts.py)
6. **The self-judge is lenient.** It scored a report with an invented "real-world benchmarks"
   section 5/5. That's why the evals add deterministic checks and support a separate judge.
7. **The model invents citation numbers.** Out-of-range `[n]` markers are stripped, and the
   Sources section is built from tool artifacts instead of being written by the model.

## Project layout

```
src/research_agent/
  config.py  logging.py  llm.py  schemas.py  deps.py (composition root)  streaming.py
  tools/web.py  tools/kb.py  rag/ingest.py
  graph/state.py  graph/prompts.py  graph/researcher.py  graph/graph.py
  api/main.py  api/static/index.html  mcp_server.py  cli.py
data/docs/   evals/   tests/   langgraph.json   Makefile
Dockerfile, docker-compose.yml   ← UNTESTED (no Docker in this WSL setup)
```

`langgraph.json` points LangGraph Studio at `research_agent.deps:make_graph`. It's optional
and needs `uv add --dev "langgraph-cli[inmem]"`, then `uv run langgraph dev`.

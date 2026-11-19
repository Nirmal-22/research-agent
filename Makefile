# All targets run without Docker. On WSL, Ollama runs as the Windows app (see README).
CHAT_MODEL ?= qwen3:4b-instruct
EMBED_MODEL ?= nomic-embed-text

.PHONY: install ollama ingest dev test test-integration evals lint format mcp clean

install:            ## install Python deps into .venv
	uv sync

ollama:             ## make sure an Ollama server is up, then pull the models
	@bash scripts/ollama-up.sh $(CHAT_MODEL) $(EMBED_MODEL)

ingest:             ## index data/docs into Chroma (data/chroma)
	uv run research-agent ingest

dev:                ## API + UI on http://localhost:8000
	uv run uvicorn --factory research_agent.api.main:create_app --port 8000 --reload

test:               ## unit tests, no Ollama needed
	uv run pytest

test-integration:   ## tests against real Ollama
	uv run pytest -m integration

evals:              ## run the eval suite against real Ollama
	uv run python evals/run_evals.py

lint:
	uv run ruff check src tests evals
	uv run ruff format --check src tests evals

format:
	uv run ruff check --fix src tests evals
	uv run ruff format src tests evals

mcp:                ## run the MCP server on stdio
	uv run python -m research_agent.mcp_server

clean:
	rm -rf data/chroma data/*.sqlite* .pytest_cache .ruff_cache evals/results

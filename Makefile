# All targets run without Docker. Ollama binary lives in ~/.local/ollama (see README).
OLLAMA ?= $(shell command -v ollama 2>/dev/null || echo $(HOME)/.local/ollama/bin/ollama)
CHAT_MODEL ?= qwen3:4b-instruct
EMBED_MODEL ?= nomic-embed-text

.PHONY: install ollama ingest dev test test-integration evals lint format mcp clean

install:            ## install Python deps into .venv
	uv sync

ollama:             ## start ollama (if not running) and pull the models
	@curl -sf -m 2 localhost:11434/api/version >/dev/null || \
	  (setsid nohup $(OLLAMA) serve > $(HOME)/.local/ollama/serve.log 2>&1 < /dev/null & sleep 3)
	$(OLLAMA) pull $(CHAT_MODEL)
	$(OLLAMA) pull $(EMBED_MODEL)

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

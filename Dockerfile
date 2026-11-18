# UNTESTED: Docker is not available in the dev environment (WSL integration off).
# Kept as a reference for how this service would be containerised.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data/docs ./data/docs
RUN uv sync --frozen --no-dev

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uv", "run", "--no-dev", "uvicorn", "--factory", "research_agent.api.main:create_app", "--host", "0.0.0.0", "--port", "8000"]

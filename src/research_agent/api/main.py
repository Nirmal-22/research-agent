"""HTTP API: run research as a Server-Sent-Events stream, approve plans, inspect threads.

    POST /research                    start a run (SSE stream; pauses at plan review)
    POST /research/{thread_id}/resume answer the plan review (SSE stream continues the run)
    GET  /research/{thread_id}        current state of a thread (from the checkpointer)
    POST /kb/ingest                   (re)index data/docs into the vector store
    GET  /health  /ready              liveness / readiness (pings Ollama, checks models)
    GET  /                            minimal HTML UI

Run: `make dev` -> `uvicorn --factory research_agent.api.main:create_app` on :8000.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from research_agent.config import Settings, get_settings
from research_agent.deps import AppRuntime, open_runtime
from research_agent.graph.graph import run_config
from research_agent.graph.state import PlanDecision
from research_agent.llm import ping_ollama
from research_agent.logging import configure_logging, get_logger
from research_agent.rag.ingest import ingest_directory
from research_agent.streaming import Event, stream_events, thread_summary

log = get_logger(__name__)
STATIC = Path(__file__).parent / "static"

RuntimeFactory = Callable[[Settings], AbstractAsyncContextManager[AppRuntime]]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    auto_approve: bool = False
    thread_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


def create_app(
    settings: Settings | None = None, runtime_factory: RuntimeFactory = open_runtime
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with runtime_factory(settings) as runtime:
            app.state.runtime = runtime
            app.state.locks = {}
            log.info("api.started", chat_model=settings.chat_model, web=settings.enable_web)
            yield

    app = FastAPI(title="research-agent", version="0.1.0", lifespan=lifespan)

    def rt(request: Request) -> AppRuntime:
        return request.app.state.runtime

    def lock_for(request: Request, thread_id: str) -> asyncio.Lock:
        # One active run per thread: concurrent writes to the same checkpoint thread race.
        return request.app.state.locks.setdefault(thread_id, asyncio.Lock())

    def sse(request: Request, thread_id: str, events: AsyncIterator[Event]) -> EventSourceResponse:
        lock = lock_for(request, thread_id)
        if lock.locked():
            raise HTTPException(409, f"thread {thread_id} already has an active run")

        async def body() -> AsyncIterator[dict[str, str]]:
            async with lock:
                structlog.contextvars.bind_contextvars(thread_id=thread_id)
                try:
                    yield {"event": "thread", "data": json.dumps({"thread_id": thread_id})}
                    async for ev in events:
                        yield {"event": ev["event"], "data": json.dumps(ev["data"], default=str)}
                except Exception as exc:
                    log.exception("run.failed")
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": type(exc).__name__, "detail": str(exc)[:500]}),
                    }
                finally:
                    structlog.contextvars.unbind_contextvars("thread_id")

        return EventSourceResponse(body(), ping=15)

    @app.post("/research")
    async def start_research(req: ResearchRequest, request: Request) -> EventSourceResponse:
        runtime = rt(request)
        thread_id = req.thread_id or uuid.uuid4().hex
        snap = await runtime.graph.aget_state(run_config(thread_id, settings))
        if snap.values:
            raise HTTPException(409, "thread_id already used; omit it to start a new thread")
        events = stream_events(
            runtime.graph,
            {"question": req.question},
            thread_id,
            settings,
            auto_approve=req.auto_approve,
        )
        return sse(request, thread_id, events)

    @app.post("/research/{thread_id}/resume")
    async def resume_research(
        thread_id: str, decision: PlanDecision, request: Request
    ) -> EventSourceResponse:
        runtime = rt(request)
        snap = await runtime.graph.aget_state(run_config(thread_id, settings))
        if not snap.values:
            raise HTTPException(404, "unknown thread")
        if not snap.interrupts:
            raise HTTPException(409, "thread is not waiting for approval")
        # Command(resume=...) is how a client answers an interrupt().
        events = stream_events(
            runtime.graph, Command(resume=decision.model_dump()), thread_id, settings
        )
        return sse(request, thread_id, events)

    @app.get("/research/{thread_id}")
    async def get_research(thread_id: str, request: Request) -> dict[str, Any]:
        summary = await thread_summary(rt(request).graph, thread_id, settings)
        if summary["status"] == "empty" and not summary["question"]:
            raise HTTPException(404, "unknown thread")
        return summary

    @app.post("/kb/ingest")
    async def kb_ingest(request: Request) -> dict[str, Any]:
        report = await asyncio.to_thread(ingest_directory, rt(request).vectorstore, settings)
        return {"files": report.files, "chunks": report.chunks, "skipped": report.skipped}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        try:
            info = await ping_ollama(settings)
        except Exception as exc:
            return JSONResponse({"ollama": "unreachable", "error": repr(exc)[:200]}, 503)
        ok = info["chat_model_present"] and info["embed_model_present"]
        return JSONResponse(info, 200 if ok else 503)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app

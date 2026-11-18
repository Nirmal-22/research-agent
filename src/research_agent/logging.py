"""Structured logging with structlog: JSON lines in prod, pretty console when LOG_JSON=false."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    renderer: Any = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # e.g. thread_id bound per request
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def timed_node[**P, R](
    name: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator for async graph nodes: logs start/end/duration/errors per node.

    Poor-man's tracing. LangSmith/Langfuse/OpenTelemetry give you this (and more) for free.
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        log = get_logger("node")

        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            log.info("node.start", node=name)
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:
                # GraphInterrupt is raised on purpose by interrupt(); don't log it as an error.
                level = "info" if type(exc).__name__ == "GraphInterrupt" else "error"
                getattr(log, level)(
                    "node.raised",
                    node=name,
                    error=type(exc).__name__,
                    ms=round((time.perf_counter() - start) * 1000),
                )
                raise
            log.info("node.end", node=name, ms=round((time.perf_counter() - start) * 1000))
            return result

        return wrapper

    return decorator

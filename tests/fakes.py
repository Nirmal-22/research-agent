"""Test doubles: a scripted chat model that supports tool binding and structured output."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field

Responder = Callable[[list[BaseMessage]], AIMessage]
StructuredSpec = Callable[[list[BaseMessage]], Any] | list[Any]

_ids = itertools.count()


def tool_call(name: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def system_text(messages: Sequence[BaseMessage]) -> str:
    return " ".join(str(m.content) for m in messages if isinstance(m, SystemMessage))


def default_responder(messages: list[BaseMessage]) -> AIMessage:
    """Behaves like a well-mannered model for every prompt in the graph."""
    sys = system_text(messages)
    if "research assistant" in sys:  # researcher agent loop
        if not any(isinstance(m, ToolMessage) for m in messages):
            return tool_call("search_knowledge_base", query=str(messages[-1].content)[-80:])
        return AIMessage("RAG grounds answers in retrieved passages (source: rag.md).")
    if "research reports" in sys:  # writer
        return AIMessage("Summary: RAG retrieves documents first [1]. Bogus cite [99].\n\n## How")
    if "factual notes" in sys:  # researcher fallback summarizer
        return AIMessage("Fallback notes from tool results.")
    return AIMessage("ok")


class ScriptedChatModel(BaseChatModel):
    """A fake chat model.

    - free-text / tool-calling replies come from `responder(messages)`
    - `with_structured_output(Schema)` replies come from `structured[Schema.__name__]`:
      a list consumed in order (items may be Exceptions to simulate bad JSON) or a callable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Responder = default_responder
    structured: dict[str, StructuredSpec] = Field(default_factory=dict)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responder(list(messages)))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Token-ish streaming (word by word) so `stream_mode="messages"` is exercised."""
        msg = self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        if getattr(msg, "tool_calls", None):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tc["name"],
                            "args": json.dumps(tc["args"]),
                            "id": tc["id"],
                            "index": i,
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ],
                )
            )
            return
        for word in re.split(r"(\s+)", str(msg.content)):
            if word:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
                if run_manager:
                    run_manager.on_llm_new_token(word, chunk=chunk)
                yield chunk

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:  # type: ignore[override]
        return self

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], **kwargs: Any
    ) -> Runnable:
        spec = self.structured.get(schema.__name__)

        def run(messages: Any) -> Any:
            msgs = messages.to_messages() if hasattr(messages, "to_messages") else messages
            if spec is None:
                raise ValueError(f"no structured response scripted for {schema.__name__}")
            value = spec(msgs) if callable(spec) else (spec.pop(0) if spec else None)
            if isinstance(value, Exception):
                raise value
            return schema.model_validate(value) if isinstance(value, dict) else value

        return RunnableLambda(run)

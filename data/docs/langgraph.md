# LangGraph Overview

LangGraph is an open-source library from LangChain Inc. for building stateful, multi-step
agent applications as graphs. It is lower level than a prebuilt agent: you define the state,
the nodes and the edges explicitly, which gives fine control over loops, branching and
human involvement.

## Core concepts

**State.** A graph has a shared state, usually a TypedDict. Each node receives the state and
returns a partial update. Each key can have a *reducer* that decides how updates are merged;
for example `Annotated[list, operator.add]` appends instead of overwriting, which allows
parallel nodes to write to the same key safely.

**Nodes and edges.** Nodes are Python functions (sync or async). Normal edges always go from
one node to another. *Conditional edges* call a routing function that inspects the state and
returns the name of the next node. Pointing a conditional edge back to an earlier node creates
a cycle, which is how agent loops and revise-until-good loops are built.

**Send.** The `Send` primitive lets a routing function or node launch a dynamic number of
parallel tasks, each with its own input. This is the map-reduce pattern: for example, one
research task per sub-question, with the results merged by a reducer.

**Command.** A node can return `Command(update=..., goto=...)` to update state and choose the
next node in a single step.

**Subgraphs.** A compiled graph can be used inside another graph, either as a node or by
calling it from inside a node. This keeps complex agents modular.

## Persistence and human-in-the-loop

When a graph is compiled with a *checkpointer* (in memory, SQLite or Postgres), the state is
saved after every step under a `thread_id`. This enables resuming a conversation, inspecting
history ("time travel") and fault tolerance.

Checkpointing also enables *interrupts*. Calling `interrupt(value)` inside a node pauses the
graph and returns `value` to the caller. The caller later resumes the same thread with
`Command(resume=answer)`; the node re-runs from its beginning and `interrupt` returns the
answer. This is the standard way to add human approval steps, for example approving a plan
before an agent spends time executing it.

## Streaming

`graph.stream()` / `astream()` support several modes: `values` (full state after each step),
`updates` (only each node's update), `messages` (LLM tokens as they are generated), `custom`
(arbitrary data written by nodes) and `debug`. Several modes can be combined.

## Deployment

LangGraph graphs can be served with the LangGraph server / LangGraph Platform, or wrapped in a
regular web framework such as FastAPI. LangGraph Studio is a visual IDE for inspecting and
debugging graphs.

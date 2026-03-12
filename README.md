# langgraph-vertex-memorybank

LangGraph `BaseStore` backed by [Vertex AI Agent Engine Memory Bank](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview) — LLM-powered long-term memory with automatic extraction, consolidation, and semantic recall.

## Why Memory Bank?

Unlike `InMemoryStore` which loses data on restart, or vector stores that just embed raw text, Memory Bank uses an LLM to extract meaningful facts and consolidates them over time:

- **LLM-powered extraction** — not just embedding, but understanding: "I moved to Portland last year" becomes the fact "User lives in Portland"
- **Automatic consolidation** — duplicate or updated facts are merged, not duplicated
- **Managed infrastructure** — no vector DB to run, scale, or maintain
- **Per-user scoping** — memories are isolated per user, session, or any scope you define

## Install

```bash
pip install langgraph-vertex-memorybank
```

Requires Python 3.10+ and a Google Cloud project with [Memory Bank enabled](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview).

## Quick Start

### 1. Create the Store

```python
from langgraph_vertex_memorybank import VertexMemoryBankStore

store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
)
```

Or pass an existing `vertexai.Client`:

```python
import vertexai

client = vertexai.Client(project="my-project", location="us-central1")
store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
    client=client,
)
```

### 2. Use as a LangGraph BaseStore

```python
# Search memories (semantic similarity)
results = store.search(
    ("memories", "user_id", "alice"),
    query="What does the user prefer?",
    limit=5,
)

# Create a memory directly
store.put(
    ("memories", "user_id", "alice"),
    "key",
    {"fact": "User prefers dark mode"},
)

# Get a memory by ID
item = store.get(("memories", "user_id", "alice"), "memory-id")

# Delete a memory
store.delete(("memories", "user_id", "alice"), "memory-id")
```

### 3. Wire into a LangGraph Agent (Pre-built Nodes)

The fastest way — use `create_recall_node` and `create_capture_node`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph_vertex_memorybank import (
    VertexMemoryBankStore,
    create_recall_node,
    create_capture_node,
)

store = VertexMemoryBankStore(...)

# recall: searches Memory Bank, injects facts as a system message
recall = create_recall_node(store, recall_mode="system", top_k=5)

# capture: extracts memories from conversation (fire-and-forget)
capture = create_capture_node(store, topics=["preferences", "facts"])

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", your_agent_fn)
builder.add_node("capture", capture)

builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "capture")
builder.add_edge("capture", END)

graph = builder.compile()

# Invoke with user_id in config
result = graph.invoke(
    {"messages": [HumanMessage(content="Hi!")]},
    {"configurable": {"user_id": "alice"}},
)
```

The recall node:
- Gets `user_id` from `config["configurable"]["user_id"]`
- Searches memories using the last user message as the query
- Injects matching facts as a `SystemMessage` (or into `state["memory_context"]` with `recall_mode="state"`)

The capture node:
- Converts recent messages to Memory Bank event format
- Calls `generate_memories()` for LLM-powered fact extraction
- Runs in a background thread by default (doesn't block the response)

### 4. Generate Memories from Conversations

Memory Bank's core feature — LLM-powered fact extraction:

```python
events = [
    {"content": {"role": "user", "parts": [{"text": "I just moved to Portland. I love hiking and coffee."}]}},
    {"content": {"role": "model", "parts": [{"text": "Welcome to Portland! Great city for both."}]}},
]

op = store.generate_memories(
    scope={"user_id": "alice"},
    events=events,
    topics=["preferences", "facts"],
)
# Memory Bank extracts: "User lives in Portland", "User loves hiking", "User loves coffee"
```

## Namespace Mapping

LangGraph uses namespace tuples. This store maps them to Memory Bank scopes:

| Namespace | Scope | Topic Filter |
|-----------|-------|-------------|
| `("memories", "user_id", "alice")` | `{"user_id": "alice"}` | None (all topics) |
| `("memories", "user_id", "alice", "topic", "preferences")` | `{"user_id": "alice"}` | `"preferences"` |
| `("memories", "user_id", "alice", "agent", "bot1")` | `{"user_id": "alice", "agent": "bot1"}` | None |

Format: `(prefix, key1, value1, key2, value2, ..., ["topic", topic_name])`

## Configuration

```python
store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",

    # Optional
    client=vertexai.Client(...),       # Use existing client
    default_top_k=10,                  # Search result count
    min_similarity_score=0.3,          # Filter low-relevance results
    topics=["preferences", "facts"],   # Default extraction topics
    disable_consolidation=False,       # Skip dedup/merge
    revision_labels={"source": "app"}, # Labels on generated memories
    namespace_prefix="memories",       # Namespace prefix
)
```

## Authentication

Uses standard Google Cloud authentication via the `google-cloud-aiplatform` SDK:

- **Application Default Credentials (ADC)** — `gcloud auth application-default login`
- **Service Account** — `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`
- **Explicit credentials** — pass to `vertexai.Client(credentials=...)`

## Advanced

### Similarity Scoring

Memory Bank returns Euclidean distance. We convert to similarity: `score = 1 / (1 + distance)`. Memories below `min_similarity_score` (default 0.3) are filtered out.

### Revision Labels

Optional labels attached to generated memories for tracking provenance:

```python
store = VertexMemoryBankStore(
    ...,
    revision_labels={"source": "langgraph", "version": "2.0"},
)
```

### Custom Topics

Topics scope what kind of facts Memory Bank extracts:

```python
store.generate_memories(
    scope={"user_id": "alice"},
    events=events,
    topics=["dietary_preferences", "travel_history", "technical_skills"],
)
```

### Async Support

All BaseStore methods have async counterparts, and `generate_memories` has `agenerate_memories`:

```python
results = await store.asearch(namespace, query="...", limit=5)
await store.aput(namespace, "key", {"fact": "..."})
op = await store.agenerate_memories(scope, events)
```

## Sample App

See [`sample/chatbot.py`](sample/chatbot.py) for a complete interactive chatbot using Gemini + Memory Bank:

```bash
pip install "langgraph-vertex-memorybank[sample]"
python sample/chatbot.py --user alice
```

## Development

```bash
git clone https://github.com/zeroasterisk/langgraph-store-vertex-memorybank
cd langgraph-store-vertex-memorybank
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run unit tests (no GCP needed)
pytest -m "not integration" -v

# Run integration tests (needs GCP auth)
pytest -m integration -v
```

## License

Apache 2.0

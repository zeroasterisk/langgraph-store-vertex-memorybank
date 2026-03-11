# langgraph-store-vertex-memorybank

LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/concepts/memory/) implementation backed by [Vertex AI Agent Engine Memory Bank](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview).

## What is this?

This package provides a drop-in LangGraph store that gives your agents **persistent, semantic long-term memory** powered by Google Cloud's managed Memory Bank service.

Unlike simple vector stores, Memory Bank uses an **LLM to extract meaningful facts** from conversations and **consolidates** them with existing memories over time — deduplicating, updating, and even deleting contradicted information automatically.

## Key Features

- **LangGraph-native**: Implements `BaseStore` — works with `graph.compile(store=...)` and LangGraph's memory patterns
- **Semantic search**: `store.search(namespace, query="...")` uses Memory Bank's similarity search
- **LLM-powered extraction**: `store.generate_memories()` extracts facts from conversation turns (Memory Bank's killer feature)
- **Scoped isolation**: Memories are isolated per user/session via namespace-to-scope mapping
- **Managed infrastructure**: No vector DB to manage — Memory Bank handles storage, embeddings, and consolidation

## Installation

```bash
pip install langgraph-store-vertex-memorybank
```

Or from source:

```bash
git clone https://github.com/zeroasterisk/langgraph-store-vertex-memorybank
cd langgraph-store-vertex-memorybank
pip install -e ".[dev,sample]"
```

## Prerequisites

1. **Google Cloud project** with billing enabled
2. **Vertex AI API** enabled
3. **Agent Engine instance** with Memory Bank — [set up guide](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/set-up)
4. **Authentication** via Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```

## Quick Start

```python
from langgraph_store_vertex_memorybank import VertexMemoryBankStore

store = VertexMemoryBankStore(
    project_id="your-project",
    location="us-central1",
    reasoning_engine_id="your-engine-id",
)

# Store a memory
store.put(
    ("memories", "user_id", "alice"),
    "pref-1",
    {"fact": "Alice prefers dark mode"},
)

# Search memories (semantic similarity)
results = store.search(
    ("memories", "user_id", "alice"),
    query="What UI preferences does the user have?",
    limit=5,
)
for r in results:
    print(f"  {r.value['fact']} (score: {r.score})")
```

## Namespace Mapping

LangGraph uses namespace tuples; Memory Bank uses scope dicts. This store maps between them:

| LangGraph Namespace | Memory Bank Scope |
|---|---|
| `("memories", "user_id", "alice")` | `{"user_id": "alice"}` |
| `("memories", "user_id", "alice", "session", "s1")` | `{"user_id": "alice", "session": "s1"}` |

The first element is the prefix (default: `"memories"`), remaining elements are alternating key-value pairs.

## Memory Generation (The Killer Feature)

Memory Bank doesn't just store text — it uses an LLM to extract facts:

```python
events = [
    {"content": {"role": "user", "parts": [{"text": "I just moved to Portland. I'm a Python developer."}]}},
    {"content": {"role": "model", "parts": [{"text": "Welcome to Portland! Great city for tech."}]}},
]

results = store.generate_memories(
    scope={"user_id": "alice"},
    events=events,
)
# Memory Bank extracts: "Lives in Portland", "Is a Python developer"
# and consolidates with existing memories (dedup, update, delete contradictions)
```

## Using with LangGraph

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

store = VertexMemoryBankStore(...)

def recall_node(state, *, store):
    results = store.search(
        ("memories", "user_id", state["user_id"]),
        query=state["messages"][-1].content,
        limit=5,
    )
    facts = [r.value["fact"] for r in results]
    return {"context": "\n".join(facts)}

graph = StateGraph(...)
graph.add_node("recall", recall_node)
# ...
compiled = graph.compile(store=store, checkpointer=MemorySaver())
```

## Sample Chatbot

A full interactive chatbot demo is included:

```bash
# Install sample dependencies
pip install -e ".[sample]"

# Set config
export GCP_PROJECT_ID=your-project
export REASONING_ENGINE_ID=your-engine-id

# Run
python sample/chatbot.py --user alice
```

The sample demonstrates:
- Memory recall before each response
- Memory extraction after each conversation turn
- Cross-session continuity (type `new` to start a new session)
- Memory inspection (type `memories` to list stored facts)

## API Reference

### `VertexMemoryBankStore`

| Method | Description |
|---|---|
| `get(namespace, key)` | Get a single memory by ID |
| `search(namespace, query=..., filter=..., limit=...)` | Semantic search or list memories |
| `put(namespace, key, value)` | Create a memory directly |
| `delete(namespace, key)` | Delete a memory |
| `list_namespaces(prefix=..., suffix=...)` | List known memory scopes |
| `generate_memories(scope, events)` | **Extract facts from conversation** (Memory Bank feature) |

All methods have async counterparts (`aget`, `asearch`, `aput`, etc.).

## Testing

```bash
# Unit tests (mocked, no GCP needed)
pytest -m "not integration"

# Integration tests (requires GCP credentials)
export GCP_PROJECT_ID=your-project
export REASONING_ENGINE_ID=your-engine-id
pytest -m integration
```

## Design Decisions

1. **BaseStore, not BaseCheckpointSaver**: Memory Bank is a semantic memory store, not a conversation state checkpoint. LangGraph's `BaseStore` (cross-thread memory) is the right abstraction. Use `MemorySaver` or `PostgresSaver` for conversation checkpointing alongside this store.

2. **Namespace-to-scope mapping**: We map LangGraph's hierarchical namespace tuples to Memory Bank's flat scope dicts using alternating key-value pairs. This preserves both models' semantics.

3. **generate_memories() as an extension**: Memory Bank's LLM-powered extraction doesn't fit neatly into BaseStore's `put()`. We expose it as an additional method that LangGraph nodes can call explicitly.

4. **urllib for HTTP**: We use stdlib `urllib` instead of adding `httpx`/`requests` as a dependency, keeping the package lightweight. The Memory Bank API is simple REST.

5. **Client-side filtering**: Memory Bank's filter API uses metadata filters. We apply LangGraph-style `filter={"key": "value"}` client-side against memory metadata.

## License

Apache 2.0

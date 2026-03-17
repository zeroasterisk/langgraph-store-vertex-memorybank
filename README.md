# langgraph-store-vertex-memorybank

LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/) backed by [Vertex AI Agent Engine Memory Bank](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview) — fully managed, LLM-powered long-term memory.

Swap one line to give any LangGraph agent persistent memory. Works with [langmem](https://langchain-ai.github.io/langmem/) tools out of the box.

```bash
pip install langgraph-store-vertex-memorybank
```

## Why

| Store | Persists? | Managed? | LLM extraction? |
|---|---|---|---|
| `InMemoryStore` | ❌ Lost on restart | — | ❌ |
| `AsyncPostgresStore` | ✅ | ❌ You run it | ❌ |
| **`VertexMemoryBankStore`** | ✅ | ✅ Google Cloud | ✅ Extracts, consolidates, deduplicates |

## Two Memory Patterns

This package supports two complementary patterns for managing long-term memory:

| Pattern | How it works | Best for |
|---|---|---|
| **A: langmem tools** | Agent decides what to store via `create_manage_memory_tool` | Explicit, agent-driven memory |
| **B: GenerateMemories + nodes** | Memory Bank's LLM extracts facts automatically after each conversation | Automatic, implicit, smarter extraction |

**They're complementary.** Use langmem tools when the agent should decide (e.g., "remember this"). Use GenerateMemories when you want automatic extraction without the agent needing to think about memory.

## Quick Start

### Pattern A: langmem tools (agent-driven)

```python
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph_store_vertex_memorybank import VertexMemoryBankStore

store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
)

agent = create_react_agent(
    "google_genai:gemini-2.5-flash",
    tools=[
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)
```

### Pattern B: GenerateMemories + nodes (automatic)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langgraph_store_vertex_memorybank import (
    VertexMemoryBankStore,
    create_recall_node,
    create_capture_node,
)

store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
)

# Recall injects relevant memories as a SystemMessage before the agent responds.
# Capture runs generate_memories() in a background thread after the agent responds.
recall = create_recall_node(store, recall_mode="system", top_k=5)
capture = create_capture_node(store, fire_and_forget=True)

def agent(state):
    # your agent logic here
    ...

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", agent)
builder.add_node("capture", capture)
builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "capture")
builder.add_edge("capture", END)

graph = builder.compile()
graph.invoke(
    {"messages": [("user", "I just moved to Portland")]},
    {"configurable": {"user_id": "alice"}},
)
```

### Direct usage

```python
from langgraph_store_vertex_memorybank import VertexMemoryBankStore

store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
)

ns = ("memories", "user_id", "alice")

store.put(ns, "pref-1", {"fact": "Prefers dark mode"})

results = store.search(ns, query="What theme?", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.value['fact']}")
```

## Memory Generation

`generate_memories()` is Memory Bank's killer feature. Unlike `put()` (which stores exactly what you give it), `generate_memories()` uses an LLM to:

- **Extract** meaningful facts from conversation turns
- **Consolidate** with existing memories — deduplicating and updating
- **Delete** contradicted information automatically

```python
# Direct usage
events = [
    {"content": {"role": "user", "parts": [{"text": "I just moved to Portland from NYC"}]}},
    {"content": {"role": "model", "parts": [{"text": "Welcome to Portland!"}]}},
]
store.generate_memories(scope={"user_id": "alice"}, events=events)
```

The `create_capture_node()` helper wraps this into a LangGraph node that automatically converts messages and runs generation in a background thread (fire-and-forget) so it doesn't block the response.

The `create_recall_node()` helper searches memories using the last user message and injects them as a `SystemMessage` (or state field) before the agent responds.

## Architecture

```
┌─ Pattern A ──────────────────────┐  ┌─ Pattern B ───────────────────────┐
│ langmem tools                    │  │ recall_node → agent → capture_node│
│ (agent decides what to store)    │  │ (automatic LLM extraction)        │
└──────────────┬───────────────────┘  └──────────────┬────────────────────┘
               ↓                                     ↓
        LangGraph BaseStore                  generate_memories()
        batch([GetOp, SearchOp, ...])        (Memory Bank extension)
               ↓                                     ↓
         VertexMemoryBankStore  ←── this package ──────┘
               ↓
    google-cloud-aiplatform SDK  ← handles auth, retries, pagination
               ↓
      Vertex AI Memory Bank  ← managed service
```

## Namespace ↔ Scope Mapping

LangGraph uses namespace tuples. Memory Bank uses scope dicts. This package maps between them:

```python
# Basic: user-scoped
("memories", "user_id", "alice")
  → scope: {"user_id": "alice"}

# With topic filter
("memories", "user_id", "alice", "topic", "preferences")
  → scope: {"user_id": "alice"}, filter: topic="preferences"

# Multi-key
("memories", "user_id", "alice", "session", "s1")
  → scope: {"user_id": "alice", "session": "s1"}
```

Namespaces are parsed as alternating key-value pairs after the prefix. The special key `"topic"` adds a server-side filter instead of a scope entry.

## BaseStore Interface

Implements the full [LangGraph BaseStore](https://langchain-ai.github.io/langgraph/reference/store/) contract:

| Method | Memory Bank SDK call |
|---|---|
| `search(ns, query)` | `memories.retrieve(similarity_search_params)` |
| `put(ns, key, value)` | `memories.create(fact, scope)` |
| `get(ns, key)` | `memories.get(name)` |
| `put(ns, key, None)` | `memories.delete(name)` |
| `list_namespaces()` | `memories.list()` → extract unique scopes |
| `batch(ops)` / `abatch(ops)` | Dispatches to above |
| `generate_memories(scope, events)` | `memories.generate(...)` *(extension)* |

Distance is converted to similarity: `score = 1 / (1 + distance)`.

Results below `min_similarity_score` (default 0.3) are filtered out.

## Configuration

```python
store = VertexMemoryBankStore(
    project_id="my-project",        # required
    location="us-central1",         # required
    reasoning_engine_id="123456",   # required
    client=None,                    # optional pre-configured vertexai.Client
    namespace_prefix="memories",    # prefix for namespace tuples
    default_top_k=10,               # default search result count
    min_similarity_score=0.3,       # similarity floor
)
```

Pass an existing `vertexai.Client` to share connections across stores.

## Prerequisites

1. Python ≥ 3.10
2. A Google Cloud project with [Vertex AI API](https://console.cloud.google.com/apis/api/aiplatform.googleapis.com) enabled
3. An Agent Engine instance for Memory Bank — [setup guide](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/set-up)
4. Authentication via [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials):
   ```bash
   gcloud auth application-default login
   ```

## Development

```bash
git clone https://github.com/zeroasterisk/langgraph-store-vertex-memorybank.git
cd langgraph-store-vertex-memorybank
uv sync --extra dev

# Unit tests (mocked, no GCP needed)
uv run pytest -m "not integration" -v

# Integration tests (needs ADC + Memory Bank instance)
uv run pytest -m integration -v
```

## References

- **Memory Bank overview**: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview
- **Memory Bank quickstart (ADK)**: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/quickstart-adk
- **Memory Bank quickstart (API)**: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/quickstart-api
- **Official LangGraph notebook**: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/memory/get_started_with_memory_bank_langgraph.ipynb
- **Official CrewAI notebook**: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/memory/get_started_with_memory_bank_crewai.ipynb
- **LangGraph long-term memory concepts**: https://langchain-ai.github.io/langgraph/concepts/memory/
- **langmem documentation**: https://langchain-ai.github.io/langmem/
- **LangGraph BaseStore reference**: https://langchain-ai.github.io/langgraph/reference/store/

## License

Apache 2.0

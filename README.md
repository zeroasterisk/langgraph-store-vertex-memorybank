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

## Quick Start

### With langmem (recommended)

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

## Architecture

```
langmem tools          ← create_manage_memory_tool / create_search_memory_tool
    ↕
LangGraph BaseStore    ← batch([GetOp, SearchOp, PutOp, ...])
    ↕
VertexMemoryBankStore  ← this package (501 lines)
    ↕
google-cloud-aiplatform SDK  ← handles auth, retries, pagination
    ↕
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

# langgraph-store-vertex-memorybank

**LangGraph `BaseStore` backed by [Vertex AI Agent Engine Memory Bank](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank) — fully managed, LLM-powered long-term memory for your agents.**

Unlike `InMemoryStore` (lost on restart) or `AsyncPostgresStore` (DIY infrastructure), Memory Bank is a fully managed service with LLM-powered extraction, consolidation, and semantic recall.

## Quick Start

```bash
pip install langgraph-store-vertex-memorybank
```

### With langmem (recommended)

Swap one line to give any LangGraph agent persistent, managed memory:

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

# The agent manages memory automatically
response = agent.invoke(
    {"messages": [{"role": "user", "content": "I prefer dark mode."}]}
)
```

### Direct usage (no langmem)

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

## Namespace ↔ Scope Mapping

LangGraph namespaces map to Memory Bank scopes:

| LangGraph namespace | Memory Bank scope | Topic filter |
|---|---|---|
| `("memories", "user_id", "alice")` | `{"user_id": "alice"}` | — |
| `("memories", "user_id", "alice", "topic", "prefs")` | `{"user_id": "alice"}` | `"prefs"` |
| `("memories", "user_id", "alice", "session", "s1")` | `{"user_id": "alice", "session": "s1"}` | — |

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `project_id` | *required* | Google Cloud project ID |
| `location` | *required* | Region (e.g. `us-central1`) |
| `reasoning_engine_id` | *required* | Agent Engine instance ID |
| `client` | `None` | Optional pre-configured `vertexai.Client` |
| `namespace_prefix` | `"memories"` | Prefix for namespace tuples |
| `default_top_k` | `10` | Default search result count |
| `min_similarity_score` | `0.3` | Minimum similarity threshold |

## BaseStore Contract

Implements the full [LangGraph BaseStore](https://langchain-ai.github.io/langgraph/reference/store/) interface:

- **`get(namespace, key)`** → fetch a single memory by ID
- **`search(namespace, query, ...)`** → semantic similarity search
- **`put(namespace, key, value)`** → create a memory
- **`delete(namespace, key)`** → remove a memory
- **`list_namespaces(...)`** → discover scopes
- **`batch(ops)` / `abatch(ops)`** → execute multiple operations

Both sync and async paths are supported.

## Prerequisites

- Python ≥ 3.10
- A Google Cloud project with [Agent Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview) enabled
- A Memory Bank instance (Reasoning Engine) — see the [official notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/memory/get_started_with_memory_bank_langgraph.ipynb)
- Authentication via ADC or service account

## Development

```bash
git clone https://github.com/zeroasterisk/langgraph-store-vertex-memorybank.git
cd langgraph-store-vertex-memorybank
uv sync --extra dev

# Unit tests (mocked, no GCP needed)
uv run pytest -m "not integration"

# Integration tests (needs ADC + Memory Bank instance)
uv run pytest -m integration
```

## License

Apache 2.0

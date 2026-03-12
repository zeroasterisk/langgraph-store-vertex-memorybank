"""Chatbot using the store directly (no langmem dependency).

Shows how to use VertexMemoryBankStore's BaseStore API directly
for manual memory management.

Usage:
    pip install langgraph-store-vertex-memorybank
    python chatbot_direct.py
"""

from langgraph_store_vertex_memorybank import VertexMemoryBankStore

store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
)

namespace = ("memories", "user_id", "alice")

# Store a memory
store.put(namespace, "pref-1", {"fact": "Prefers dark mode"})

# Search semantically
results = store.search(namespace, query="What UI theme does the user like?", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.value['fact']}")

# Get by key
item = store.get(namespace, "pref-1")
if item:
    print(f"Got: {item.value['fact']}")

# List known namespaces
namespaces = store.list_namespaces()
print(f"Namespaces: {namespaces}")

# Delete
store.delete(namespace, "pref-1")

"""LangGraph BaseStore backed by Vertex AI Agent Engine Memory Bank.

Provides persistent, semantic long-term memory for LangGraph agents using
Google Cloud's managed Memory Bank service. Memory Bank uses an LLM to extract
meaningful facts from conversations and consolidates them over time.

Quick start::

    from langgraph_vertex_memorybank import VertexMemoryBankStore

    store = VertexMemoryBankStore(
        project_id="my-project",
        location="us-central1",
        reasoning_engine_id="123456",
    )

    # Search memories
    results = store.search(
        ("memories", "user_id", "alice"),
        query="What does the user prefer?",
    )

Pre-built graph nodes::

    from langgraph_vertex_memorybank import create_recall_node, create_capture_node

    recall = create_recall_node(store, recall_mode="system", top_k=5)
    capture = create_capture_node(store, topics=["preferences", "facts"])
"""

from langgraph_vertex_memorybank.nodes import create_capture_node, create_recall_node
from langgraph_vertex_memorybank.store import VertexMemoryBankStore

# Alias for discoverability
LangGraphVertexAIMemoryBank = VertexMemoryBankStore

__all__ = [
    "VertexMemoryBankStore",
    "LangGraphVertexAIMemoryBank",
    "create_recall_node",
    "create_capture_node",
]

__version__ = "0.2.0"

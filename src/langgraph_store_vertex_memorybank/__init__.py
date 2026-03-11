"""LangGraph BaseStore backed by Vertex AI Agent Engine Memory Bank.

Provides persistent, semantic long-term memory for LangGraph agents using
Google Cloud's managed Memory Bank service, which features LLM-powered
memory extraction and consolidation.

Usage:
    from langgraph_store_vertex_memorybank import VertexMemoryBankStore

    store = VertexMemoryBankStore(
        project_id="my-project",
        location="us-central1",
        reasoning_engine_id="1234567890",
    )
"""

from langgraph_store_vertex_memorybank.store import VertexMemoryBankStore

__all__ = ["VertexMemoryBankStore"]
__version__ = "0.1.0"

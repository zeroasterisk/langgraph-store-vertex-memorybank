"""LangGraph BaseStore backed by Vertex AI Agent Engine Memory Bank."""

from langgraph_store_vertex_memorybank.store import VertexMemoryBankStore

LangGraphVertexAIMemoryBank = VertexMemoryBankStore

__all__ = ["VertexMemoryBankStore", "LangGraphVertexAIMemoryBank"]

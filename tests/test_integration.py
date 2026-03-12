"""Integration tests — hit the real Vertex AI Memory Bank API.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

from __future__ import annotations

import os
import uuid

import pytest

from langgraph_store_vertex_memorybank import VertexMemoryBankStore

pytestmark = pytest.mark.integration

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "zaf-sandbox")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
ENGINE_ID = os.environ.get("REASONING_ENGINE_ID", "8505178738172887040")


@pytest.fixture
def store() -> VertexMemoryBankStore:
    return VertexMemoryBankStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        reasoning_engine_id=ENGINE_ID,
    )


@pytest.fixture
def test_scope() -> dict[str, str]:
    return {"user_id": f"test-{uuid.uuid4().hex[:8]}"}


class TestIntegrationSearch:
    def test_search_empty_scope(self, store: VertexMemoryBankStore) -> None:
        scope = {"user_id": f"empty-{uuid.uuid4().hex[:8]}"}
        ns = store.namespace_for_scope(scope)
        assert store.search(ns, query="anything") == []

    def test_search_existing_scope(self, store: VertexMemoryBankStore) -> None:
        ns = ("memories", "user_id", "alan")
        results = store.search(ns, query="What does the user work on?", limit=3)
        assert isinstance(results, list)
        for r in results:
            assert "fact" in r.value


class TestIntegrationPutAndGet:
    def test_create_and_search(self, store: VertexMemoryBankStore, test_scope: dict[str, str]) -> None:
        ns = store.namespace_for_scope(test_scope)
        store.put(ns, "test-key", {"fact": "Integration test memory"})
        results = store.search(ns, query="integration test")
        assert len(results) > 0
        # Cleanup
        for r in results:
            store.delete(ns, r.key)

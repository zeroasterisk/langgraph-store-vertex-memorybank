"""Integration tests that hit the real Vertex AI Memory Bank API.

Run with: pytest -m integration
Skip with: pytest -m "not integration"

Requires:
    - GOOGLE_APPLICATION_CREDENTIALS or ADC configured
    - GCP project with Memory Bank enabled
"""

from __future__ import annotations

import os
import uuid

import pytest

from langgraph_vertex_memorybank import VertexMemoryBankStore

pytestmark = pytest.mark.integration

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "zaf-sandbox")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
ENGINE_ID = os.environ.get("REASONING_ENGINE_ID", "8505178738172887040")


@pytest.fixture
def store() -> VertexMemoryBankStore:
    """Create a store connected to the real API."""
    return VertexMemoryBankStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        reasoning_engine_id=ENGINE_ID,
    )


@pytest.fixture
def test_scope() -> dict[str, str]:
    """Create a unique scope for test isolation."""
    return {"user_id": f"test-{uuid.uuid4().hex[:8]}"}


class TestIntegrationSearch:
    def test_search_empty_scope(self, store: VertexMemoryBankStore) -> None:
        """Searching an empty scope should return no results."""
        scope = {"user_id": f"empty-{uuid.uuid4().hex[:8]}"}
        ns = store.namespace_for_scope(scope)
        results = store.search(ns, query="anything")
        assert results == []

    def test_search_existing_scope(self, store: VertexMemoryBankStore) -> None:
        """Search against the existing alan scope (has real memories)."""
        ns = ("memories", "user_id", "alan")
        results = store.search(ns, query="What does the user work on?", limit=3)
        assert isinstance(results, list)
        for r in results:
            assert hasattr(r, "value")
            assert "fact" in r.value


class TestIntegrationPutAndGet:
    def test_create_and_retrieve_memory(
        self, store: VertexMemoryBankStore, test_scope: dict[str, str]
    ) -> None:
        """Create a memory and verify it can be retrieved."""
        ns = store.namespace_for_scope(test_scope)

        # Create
        store.put(ns, "test-key", {"fact": "Integration test memory"})

        # Search for it
        results = store.search(ns, query="integration test")
        assert len(results) > 0
        facts = [r.value["fact"] for r in results]
        assert any("ntegration" in f for f in facts)

        # Cleanup
        for r in results:
            store.delete(ns, r.key)


class TestIntegrationGenerateMemories:
    def test_generate_from_events(
        self, store: VertexMemoryBankStore, test_scope: dict[str, str]
    ) -> None:
        """Generate memories from conversation events."""
        events = [
            {
                "content": {
                    "role": "user",
                    "parts": [{"text": "I just moved to Austin, Texas. I'm a software engineer who loves tacos."}],
                }
            },
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Welcome to Austin! Great city for tech and tacos."}],
                }
            },
        ]

        op = store.generate_memories(
            scope=test_scope,
            events=events,
        )

        # The SDK returns an operation object
        assert op is not None

        # Verify memories are searchable
        ns = store.namespace_for_scope(test_scope)
        search_results = store.search(ns, query="where does the user live")
        assert len(search_results) > 0

        # Cleanup
        for r in search_results:
            store.delete(ns, r.key)

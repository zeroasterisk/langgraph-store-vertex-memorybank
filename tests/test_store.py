"""Unit tests for VertexMemoryBankStore.

All API calls are mocked — no real GCP credentials needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langgraph_store_vertex_memorybank.store import (
    VertexMemoryBankStore,
    _extract_memory_id,
    _memory_resource_name,
    _memory_to_item,
    _memory_to_search_item,
    _namespace_to_scope,
    _scope_to_namespace,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_store() -> VertexMemoryBankStore:
    """Create a store with mocked credentials."""
    with patch("langgraph_store_vertex_memorybank.store.google_auth_default") as mock_auth:
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "fake-token"
        mock_auth.return_value = (mock_creds, "test-project")
        store = VertexMemoryBankStore(
            project_id="test-project",
            location="us-central1",
            reasoning_engine_id="123456",
            credentials=mock_creds,
        )
    return store


def _mock_memory(
    memory_id: str = "mem-1",
    fact: str = "User likes Python",
    scope: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a mock memory response."""
    return {
        "name": f"projects/test-project/locations/us-central1/reasoningEngines/123456/memories/{memory_id}",
        "scope": scope or {"user_id": "alice"},
        "fact": fact,
        "metadata": {},
        "createTime": "2026-01-15T10:00:00Z",
        "updateTime": "2026-01-15T10:30:00Z",
    }


# ── Namespace/Scope mapping tests ──────────────────────────────────────


class TestNamespaceMapping:
    def test_namespace_to_scope_basic(self) -> None:
        ns = ("memories", "user_id", "alice")
        assert _namespace_to_scope(ns) == {"user_id": "alice"}

    def test_namespace_to_scope_multiple_keys(self) -> None:
        ns = ("memories", "user_id", "alice", "session_id", "s1")
        assert _namespace_to_scope(ns) == {"user_id": "alice", "session_id": "s1"}

    def test_namespace_to_scope_too_short(self) -> None:
        with pytest.raises(ValueError, match="at least 3 elements"):
            _namespace_to_scope(("memories",))

    def test_namespace_to_scope_odd_pairs(self) -> None:
        with pytest.raises(ValueError):
            _namespace_to_scope(("memories", "user_id"))

    def test_scope_to_namespace_basic(self) -> None:
        scope = {"user_id": "alice"}
        result = _scope_to_namespace(scope)
        assert result == ("memories", "user_id", "alice")

    def test_scope_to_namespace_sorted_keys(self) -> None:
        scope = {"z_key": "z_val", "a_key": "a_val"}
        result = _scope_to_namespace(scope)
        assert result == ("memories", "a_key", "a_val", "z_key", "z_val")

    def test_scope_to_namespace_custom_prefix(self) -> None:
        scope = {"user_id": "bob"}
        result = _scope_to_namespace(scope, prefix="user_memories")
        assert result == ("user_memories", "user_id", "bob")

    def test_roundtrip(self) -> None:
        scope = {"user_id": "alice", "agent": "chatbot"}
        ns = _scope_to_namespace(scope)
        recovered = _namespace_to_scope(ns)
        assert recovered == scope


# ── Resource name tests ─────────────────────────────────────────────────


class TestResourceNames:
    def test_memory_resource_name(self) -> None:
        name = _memory_resource_name("proj", "us-c1", "eng1", "mem1")
        assert name == "projects/proj/locations/us-c1/reasoningEngines/eng1/memories/mem1"

    def test_extract_memory_id(self) -> None:
        name = "projects/p/locations/l/reasoningEngines/e/memories/abc123"
        assert _extract_memory_id(name) == "abc123"

    def test_extract_memory_id_short(self) -> None:
        assert _extract_memory_id("abc123") == "abc123"


# ── Memory to Item conversion tests ─────────────────────────────────────


class TestMemoryConversion:
    def test_memory_to_item(self) -> None:
        mem = _mock_memory("m1", "Likes hiking")
        ns = ("memories", "user_id", "alice")
        item = _memory_to_item(mem, ns)

        assert item.key == "m1"
        assert item.namespace == ns
        assert item.value["fact"] == "Likes hiking"
        assert item.created_at.year == 2026

    def test_memory_to_search_item_with_distance(self) -> None:
        mem_result = {
            "memory": _mock_memory("m2", "Prefers dark mode"),
            "distance": 0.5,
        }
        ns = ("memories", "user_id", "alice")
        item = _memory_to_search_item(mem_result, ns)

        assert item.key == "m2"
        assert item.value["fact"] == "Prefers dark mode"
        # score = 1 / (1 + 0.5) = 0.666...
        assert item.score is not None
        assert abs(item.score - 0.6667) < 0.001

    def test_memory_to_search_item_no_distance(self) -> None:
        mem_result = {"memory": _mock_memory("m3", "Lives in Portland")}
        ns = ("memories", "user_id", "alice")
        item = _memory_to_search_item(mem_result, ns)

        assert item.score is None


# ── Store operation tests ───────────────────────────────────────────────


class TestStoreGet:
    def test_get_found(self) -> None:
        store = _make_store()
        mem = _mock_memory("mem-1", "Likes Python")
        with patch.object(store, "_make_request", return_value=mem):
            result = store.get(("memories", "user_id", "alice"), "mem-1")

        assert result is not None
        assert result.key == "mem-1"
        assert result.value["fact"] == "Likes Python"

    def test_get_not_found(self) -> None:
        store = _make_store()
        with patch.object(
            store, "_make_request", side_effect=RuntimeError("API error 404: Not Found")
        ):
            result = store.get(("memories", "user_id", "alice"), "nonexistent")

        assert result is None

    def test_get_other_error_raises(self) -> None:
        store = _make_store()
        with patch.object(
            store, "_make_request", side_effect=RuntimeError("API error 500: Internal")
        ):
            with pytest.raises(RuntimeError, match="500"):
                store.get(("memories", "user_id", "alice"), "mem-1")


class TestStoreSearch:
    def test_search_with_query(self) -> None:
        store = _make_store()
        response = {
            "retrievedMemories": [
                {"memory": _mock_memory("m1", "Lives in Portland"), "distance": 0.3},
                {"memory": _mock_memory("m2", "Likes coffee"), "distance": 0.7},
            ]
        }
        with patch.object(store, "_make_request", return_value=response) as mock_req:
            results = store.search(
                ("memories", "user_id", "alice"),
                query="Where does the user live?",
                limit=5,
            )

        assert len(results) == 2
        assert results[0].value["fact"] == "Lives in Portland"
        assert results[0].score is not None
        assert results[0].score > results[1].score  # type: ignore[operator]

        # Verify API call
        mock_req.assert_called_once()
        body = _get_body(mock_req)
        assert body["scope"] == {"user_id": "alice"}
        assert body["similaritySearchParams"]["searchQuery"] == "Where does the user live?"

    def test_search_without_query(self) -> None:
        store = _make_store()
        response = {
            "retrievedMemories": [
                {"memory": _mock_memory("m1", "Fact 1")},
            ]
        }
        with patch.object(store, "_make_request", return_value=response):
            results = store.search(("memories", "user_id", "alice"))

        assert len(results) == 1

    def test_search_with_filter(self) -> None:
        store = _make_store()
        response = {
            "retrievedMemories": [
                {
                    "memory": {
                        **_mock_memory("m1", "Pref 1"),
                        "metadata": {"category": "preference"},
                    }
                },
                {
                    "memory": {
                        **_mock_memory("m2", "Info 1"),
                        "metadata": {"category": "info"},
                    }
                },
            ]
        }
        with patch.object(store, "_make_request", return_value=response):
            results = store.search(
                ("memories", "user_id", "alice"),
                filter={"category": "preference"},
            )

        # Client-side filter should only return the preference
        assert len(results) == 1
        assert results[0].value["fact"] == "Pref 1"

    def test_search_bad_namespace_returns_empty(self) -> None:
        store = _make_store()
        results = store.search(("too_short",))
        assert results == []

    def test_search_api_error_returns_empty(self) -> None:
        store = _make_store()
        with patch.object(
            store, "_make_request", side_effect=RuntimeError("API error 503")
        ):
            results = store.search(("memories", "user_id", "alice"))

        assert results == []


def _get_body(mock_req: MagicMock) -> dict[str, Any]:
    """Extract the json_body from a mock _make_request call."""
    call_args = mock_req.call_args
    if call_args[1] and "json_body" in call_args[1]:
        return call_args[1]["json_body"]
    # Positional args: method, path, json_body
    return call_args[0][2]


class TestStorePut:
    def test_put_creates_memory(self) -> None:
        store = _make_store()
        with patch.object(store, "_make_request", return_value=_mock_memory()) as mock_req:
            store.put(
                ("memories", "user_id", "alice"),
                "new-mem",
                {"fact": "Likes hiking"},
            )

        mock_req.assert_called_once()
        assert mock_req.call_args[0][0] == "POST"  # method
        body = _get_body(mock_req)
        assert body["scope"] == {"user_id": "alice"}
        assert body["fact"] == "Likes hiking"

    def test_put_with_metadata(self) -> None:
        store = _make_store()
        with patch.object(store, "_make_request", return_value=_mock_memory()) as mock_req:
            store.put(
                ("memories", "user_id", "alice"),
                "new-mem",
                {"fact": "Likes tea", "metadata": {"source": "chat"}},
            )

        body = _get_body(mock_req)
        assert body["metadata"] == {"source": "chat"}

    def test_put_without_fact_serializes_value(self) -> None:
        store = _make_store()
        with patch.object(store, "_make_request", return_value={}) as mock_req:
            store.put(
                ("memories", "user_id", "alice"),
                "key1",
                {"name": "Alice", "age": 30},
            )

        body = _get_body(mock_req)
        # fact should be JSON-serialized value since no "fact" key
        parsed = json.loads(body["fact"])
        assert parsed["name"] == "Alice"


class TestStoreDelete:
    def test_delete_calls_api(self) -> None:
        store = _make_store()
        with patch.object(store, "_make_request", return_value={}) as mock_req:
            store.delete(("memories", "user_id", "alice"), "mem-to-delete")

        mock_req.assert_called_once_with("DELETE", "/memories/mem-to-delete")

    def test_delete_not_found_is_silent(self) -> None:
        store = _make_store()
        with patch.object(
            store, "_make_request", side_effect=RuntimeError("API error 404: Not Found")
        ):
            # Should not raise
            store.delete(("memories", "user_id", "alice"), "nonexistent")


class TestStoreListNamespaces:
    def test_list_from_known_scopes(self) -> None:
        store = _make_store()
        store._known_scopes.add((("user_id", "alice"),))
        store._known_scopes.add((("user_id", "bob"),))

        with patch.object(store, "_make_request", return_value={"memories": []}):
            result = store.list_namespaces()

        assert len(result) == 2
        assert ("memories", "user_id", "alice") in result
        assert ("memories", "user_id", "bob") in result

    def test_list_discovers_from_api(self) -> None:
        store = _make_store()
        api_response = {
            "memories": [
                _mock_memory("m1", scope={"user_id": "carol"}),
            ]
        }
        with patch.object(store, "_make_request", return_value=api_response):
            result = store.list_namespaces()

        assert ("memories", "user_id", "carol") in result

    def test_list_with_prefix_filter(self) -> None:
        store = _make_store()
        store._known_scopes.add((("user_id", "alice"),))
        store._known_scopes.add((("agent_id", "bot1"),))

        with patch.object(store, "_make_request", return_value={"memories": []}):
            result = store.list_namespaces(prefix=("memories", "user_id"))

        # Only the user_id namespace should match
        assert len(result) == 1
        assert ("memories", "user_id", "alice") in result

    def test_list_with_max_depth(self) -> None:
        store = _make_store()
        store._known_scopes.add((("agent", "bot1"), ("user_id", "alice")))

        with patch.object(store, "_make_request", return_value={"memories": []}):
            result = store.list_namespaces(max_depth=2)

        # Namespace ("memories", "agent", "bot1", "user_id", "alice") truncated to depth 2
        assert all(len(ns) <= 2 for ns in result)


# ── Generate memories tests ─────────────────────────────────────────────


class TestGenerateMemories:
    def test_generate_memories_sync(self) -> None:
        store = _make_store()
        api_response = {
            "done": True,
            "response": {
                "generatedMemories": [
                    {"memory": _mock_memory("g1", "Lives in Portland"), "action": "CREATED"},
                    {"memory": _mock_memory("g2", "Likes hiking"), "action": "CREATED"},
                ]
            },
        }
        events = [
            {"content": {"role": "user", "parts": [{"text": "I live in Portland and love hiking"}]}},
        ]

        with patch.object(store, "_make_request", return_value=api_response) as mock_req:
            results = store.generate_memories(
                scope={"user_id": "alice"},
                events=events,
            )

        assert len(results) == 2
        assert results[0]["action"] == "CREATED"

        # Verify API call
        mock_req.assert_called_once()
        body = _get_body(mock_req)
        assert body["scope"] == {"user_id": "alice"}
        assert "directContentsSource" in body

    def test_generate_memories_background(self) -> None:
        store = _make_store()
        api_response = {
            "name": "projects/p/locations/l/operations/op1",
            "done": False,
        }
        events = [
            {"content": {"role": "user", "parts": [{"text": "Hello"}]}},
        ]

        with patch.object(store, "_make_request", return_value=api_response):
            results = store.generate_memories(
                scope={"user_id": "alice"},
                events=events,
                wait_for_completion=False,
            )

        # Should return operation info
        assert len(results) == 1
        assert "operation" in results[0]

    def test_generate_with_disable_consolidation(self) -> None:
        store = _make_store()
        with patch.object(store, "_make_request", return_value={"done": True, "response": {"generatedMemories": []}}) as mock_req:
            store.generate_memories(
                scope={"user_id": "alice"},
                events=[{"content": {"role": "user", "parts": [{"text": "test"}]}}],
                disable_consolidation=True,
            )

        body = _get_body(mock_req)
        assert body["disableConsolidation"] is True


# ── Utility method tests ────────────────────────────────────────────────


class TestUtilities:
    def test_scope_for_namespace(self) -> None:
        store = _make_store()
        ns = ("memories", "user_id", "alice", "session", "s1")
        scope = store.scope_for_namespace(ns)
        assert scope == {"user_id": "alice", "session": "s1"}

    def test_namespace_for_scope(self) -> None:
        store = _make_store()
        scope = {"user_id": "bob"}
        ns = store.namespace_for_scope(scope)
        assert ns == ("memories", "user_id", "bob")


# ── Auth tests ──────────────────────────────────────────────────────────


class TestAuth:
    def test_get_headers_refreshes_expired_credentials(self) -> None:
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.token = "refreshed-token"

        store = VertexMemoryBankStore(
            project_id="p",
            location="l",
            reasoning_engine_id="e",
            credentials=mock_creds,
        )
        headers = store._get_headers()
        mock_creds.refresh.assert_called_once()
        assert headers["Authorization"] == "Bearer refreshed-token"

    def test_get_headers_skips_refresh_when_valid(self) -> None:
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "valid-token"

        store = VertexMemoryBankStore(
            project_id="p",
            location="l",
            reasoning_engine_id="e",
            credentials=mock_creds,
        )
        headers = store._get_headers()
        mock_creds.refresh.assert_not_called()
        assert headers["Authorization"] == "Bearer valid-token"


# ── URL construction tests ──────────────────────────────────────────────


class TestURLConstruction:
    def test_base_url(self) -> None:
        store = _make_store()
        assert "test-project" in store._base_url
        assert "us-central1" in store._base_url
        assert "123456" in store._base_url
        assert store._base_url.startswith("https://")

    def test_different_location(self) -> None:
        mock_creds = MagicMock()
        mock_creds.valid = True
        store = VertexMemoryBankStore(
            project_id="proj",
            location="europe-west1",
            reasoning_engine_id="789",
            credentials=mock_creds,
        )
        assert "europe-west1-aiplatform.googleapis.com" in store._base_url

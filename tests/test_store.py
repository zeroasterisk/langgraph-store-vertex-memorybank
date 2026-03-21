"""Unit tests for VertexMemoryBankStore (v3 — thin BaseStore).

All SDK calls are mocked — no GCP credentials needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langgraph.store.base import MatchCondition

from langgraph_store_vertex_memorybank.store import (
    VertexMemoryBankStore,
    _distance_to_score,
    _extract_memory_id,
    _filter_namespaces,
    _parse_namespace,
    _scope_to_namespace,
    _sdk_memory_to_item,
    _sdk_retrieved_to_search_item,
)

# ── Helpers ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_memories():
    return MagicMock()


@pytest.fixture
def store(mock_memories):
    with patch("langgraph_store_vertex_memorybank.store.vertexai.Client"):
        s = VertexMemoryBankStore(
            project_id="test-project",
            location="us-central1",
            reasoning_engine_id="123456",
        )
    s._client = MagicMock()
    s._client.agent_engines.memories = mock_memories
    s._client.aio.agent_engines.memories = mock_memories
    return s


def _make_store(**kwargs: Any) -> tuple[MagicMock, VertexMemoryBankStore]:
    mock_memories = MagicMock()
    with patch("langgraph_store_vertex_memorybank.store.vertexai.Client"):
        defaults = dict(project_id="test-project", location="us-central1", reasoning_engine_id="123456")
        defaults.update(kwargs)
        s = VertexMemoryBankStore(**defaults)
    s._client = MagicMock()
    s._client.agent_engines.memories = mock_memories
    s._client.aio.agent_engines.memories = mock_memories
    return mock_memories, s


def _mock_sdk_memory(
    memory_id: str = "mem-1",
    fact: str = "User likes Python",
    scope: dict[str, str] | None = None,
) -> MagicMock:
    mem = MagicMock()
    mem.name = f"projects/test-project/locations/us-central1/reasoningEngines/123456/memories/{memory_id}"
    mem.fact = fact
    mem.scope = scope or {"user_id": "alice"}
    mem.metadata = None
    mem.create_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    mem.update_time = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    return mem


def _mock_sdk_retrieved(
    memory_id: str = "mem-1",
    fact: str = "User likes Python",
    scope: dict[str, str] | None = None,
    distance: float | None = None,
) -> MagicMock:
    retrieved = MagicMock()
    retrieved.memory = _mock_sdk_memory(memory_id, fact, scope)
    retrieved.distance = distance
    return retrieved


# ── Namespace / Scope mapping ───────────────────────────────────────────


class TestParseNamespace:
    def test_basic(self) -> None:
        scope, topic = _parse_namespace(("memories", "user_id", "alice"))
        assert scope == {"user_id": "alice"}
        assert topic is None

    def test_multiple_keys(self) -> None:
        scope, topic = _parse_namespace(("memories", "user_id", "alice", "session_id", "s1"))
        assert scope == {"user_id": "alice", "session_id": "s1"}
        assert topic is None

    def test_with_topic(self) -> None:
        scope, topic = _parse_namespace(("memories", "user_id", "alice", "topic", "preferences"))
        assert scope == {"user_id": "alice"}
        assert topic == "preferences"

    def test_multiple_keys_with_topic(self) -> None:
        scope, topic = _parse_namespace(
            ("memories", "user_id", "alice", "agent", "bot1", "topic", "facts")
        )
        assert scope == {"user_id": "alice", "agent": "bot1"}
        assert topic == "facts"

    def test_too_short(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            _parse_namespace(("memories",))

    def test_odd_pairs(self) -> None:
        with pytest.raises(ValueError):
            _parse_namespace(("memories", "user_id"))

    def test_only_topic_raises(self) -> None:
        with pytest.raises(ValueError, match="non-topic"):
            _parse_namespace(("memories", "topic", "preferences"))


class TestScopeToNamespace:
    def test_basic(self) -> None:
        assert _scope_to_namespace({"user_id": "alice"}) == ("memories", "user_id", "alice")

    def test_sorted_keys(self) -> None:
        result = _scope_to_namespace({"z_key": "z", "a_key": "a"})
        assert result == ("memories", "a_key", "a", "z_key", "z")

    def test_custom_prefix(self) -> None:
        result = _scope_to_namespace({"user_id": "bob"}, prefix="custom")
        assert result == ("custom", "user_id", "bob")

    def test_with_topic(self) -> None:
        result = _scope_to_namespace({"user_id": "alice"}, topic="prefs")
        assert result == ("memories", "user_id", "alice", "topic", "prefs")

    def test_roundtrip(self) -> None:
        scope = {"user_id": "alice", "agent": "chatbot"}
        ns = _scope_to_namespace(scope)
        recovered, topic = _parse_namespace(ns)
        assert recovered == scope
        assert topic is None

    def test_roundtrip_with_topic(self) -> None:
        scope = {"user_id": "alice"}
        ns = _scope_to_namespace(scope, topic="facts")
        recovered, topic = _parse_namespace(ns)
        assert recovered == scope
        assert topic == "facts"


# ── Distance / Score ────────────────────────────────────────────────────


class TestDistanceToScore:
    def test_zero_distance(self) -> None:
        assert _distance_to_score(0.0) == 1.0

    def test_positive_distance(self) -> None:
        assert abs(_distance_to_score(0.5) - 0.6667) < 0.001  # type: ignore

    def test_none(self) -> None:
        assert _distance_to_score(None) is None


# ── Resource name helpers ───────────────────────────────────────────────


class TestExtractMemoryId:
    def test_full_name(self) -> None:
        assert _extract_memory_id("projects/p/locations/l/reasoningEngines/e/memories/abc") == "abc"

    def test_short_name(self) -> None:
        assert _extract_memory_id("abc123") == "abc123"


# ── SDK Memory conversion ──────────────────────────────────────────────


class TestSDKMemoryConversion:
    def test_memory_to_item(self) -> None:
        item = _sdk_memory_to_item(_mock_sdk_memory("m1", "Likes hiking"), ("memories", "user_id", "alice"))
        assert item.key == "m1"
        assert item.value["fact"] == "Likes hiking"
        assert item.created_at.year == 2026

    def test_retrieved_to_search_item_with_distance(self) -> None:
        item = _sdk_retrieved_to_search_item(
            _mock_sdk_retrieved("m2", "Prefers dark mode", distance=0.5),
            ("memories", "user_id", "alice"),
        )
        assert item.key == "m2"
        assert item.value["fact"] == "Prefers dark mode"
        assert abs(item.score - 0.6667) < 0.001  # type: ignore

    def test_retrieved_to_search_item_no_distance(self) -> None:
        item = _sdk_retrieved_to_search_item(
            _mock_sdk_retrieved("m3", "Lives in Portland", distance=None),
            ("memories", "user_id", "alice"),
        )
        assert item.score is None

    def test_memory_with_metadata(self) -> None:
        mem = _mock_sdk_memory("m4", "Has a dog")
        meta_val = MagicMock()
        meta_val.string_value = "pet_info"
        meta_val.double_value = None
        meta_val.bool_value = None
        mem.metadata = {"category": meta_val}
        item = _sdk_memory_to_item(mem, ("memories", "user_id", "alice"))
        assert item.value["metadata"]["category"] == "pet_info"


# ── Store: Get ──────────────────────────────────────────────────────────


class TestStoreGet:
    def test_get_found(self, store, mock_memories) -> None:
        mock_memories.get.return_value = _mock_sdk_memory("mem-1", "Likes Python")
        result = store.get(("memories", "user_id", "alice"), "mem-1")
        assert result is not None
        assert result.key == "mem-1"
        assert result.value["fact"] == "Likes Python"

    def test_get_not_found(self, store, mock_memories) -> None:
        mock_memories.get.side_effect = Exception("404 NOT_FOUND")
        assert store.get(("memories", "user_id", "alice"), "nonexistent") is None

    def test_get_other_error_raises(self, store, mock_memories) -> None:
        mock_memories.get.side_effect = Exception("500 Internal Server Error")
        with pytest.raises(Exception, match="500"):
            store.get(("memories", "user_id", "alice"), "mem-1")


# ── Store: Search ───────────────────────────────────────────────────────


class TestStoreSearch:
    def test_search_with_query(self, store, mock_memories) -> None:
        mock_memories.retrieve.return_value = [
            _mock_sdk_retrieved("m1", "Lives in Portland", distance=0.3),
            _mock_sdk_retrieved("m2", "Likes coffee", distance=0.7),
        ]
        results = store.search(("memories", "user_id", "alice"), query="Where does the user live?", limit=5)
        assert len(results) == 2
        assert results[0].value["fact"] == "Lives in Portland"
        assert results[0].score > results[1].score  # type: ignore

    def test_search_without_query(self, store, mock_memories) -> None:
        mock_memories.retrieve.return_value = [_mock_sdk_retrieved("m1", "Fact 1")]
        assert len(store.search(("memories", "user_id", "alice"))) == 1

    def test_search_with_topic_namespace(self, store, mock_memories) -> None:
        mock_memories.retrieve.return_value = [_mock_sdk_retrieved("m1", "Dark mode", distance=0.2)]
        results = store.search(("memories", "user_id", "alice", "topic", "preferences"), query="theme")
        assert len(results) == 1
        call_kwargs = mock_memories.retrieve.call_args.kwargs
        assert "preferences" in call_kwargs.get("config", {}).get("filter", "")

    def test_search_with_filter(self, store, mock_memories) -> None:
        mock_memories.retrieve.return_value = [
            _mock_sdk_retrieved("m1", "Pref 1"),
            _mock_sdk_retrieved("m2", "Info 1"),
        ]
        results = store.search(("memories", "user_id", "alice"), filter={"fact": "Pref 1"})
        assert len(results) == 1

    def test_search_min_similarity_filter(self) -> None:
        mock_memories, s = _make_store(min_similarity_score=0.5)
        mock_memories.retrieve.return_value = [
            _mock_sdk_retrieved("m1", "Good match", distance=0.3),
            _mock_sdk_retrieved("m2", "Bad match", distance=5.0),
        ]
        results = s.search(("memories", "user_id", "alice"), query="test")
        assert len(results) == 1
        assert results[0].value["fact"] == "Good match"

    def test_search_bad_namespace_returns_empty(self, store) -> None:
        assert store.search(("too_short",)) == []

    def test_search_api_error_returns_empty(self, store, mock_memories) -> None:
        mock_memories.retrieve.side_effect = Exception("503 Unavailable")
        assert store.search(("memories", "user_id", "alice")) == []

    def test_search_respects_limit_and_offset(self, store, mock_memories) -> None:
        mock_memories.retrieve.return_value = [_mock_sdk_retrieved(f"m{i}", f"Fact {i}") for i in range(10)]
        results = store.search(("memories", "user_id", "alice"), limit=3, offset=2)
        assert len(results) == 3
        assert results[0].value["fact"] == "Fact 2"


# ── Store: Put ──────────────────────────────────────────────────────────


class TestStorePut:
    def test_put_creates_memory(self, store, mock_memories) -> None:
        store.put(("memories", "user_id", "alice"), "new", {"fact": "Likes hiking"})
        mock_memories.create.assert_called_once_with(
            name="projects/test-project/locations/us-central1/reasoningEngines/123456",
            fact="Likes hiking",
            scope={"user_id": "alice"},
        )

    def test_put_without_fact_serializes_value(self, store, mock_memories) -> None:
        store.put(("memories", "user_id", "alice"), "k", {"name": "Alice", "age": 30})
        parsed = json.loads(mock_memories.create.call_args.kwargs["fact"])
        assert parsed["name"] == "Alice"

    def test_put_tracks_scope(self, store, mock_memories) -> None:
        store.put(("memories", "user_id", "alice"), "k", {"fact": "test"})
        assert (("user_id", "alice"),) in store._known_scopes

    def test_put_none_deletes(self, store, mock_memories) -> None:
        store.put(("memories", "user_id", "alice"), "mem-to-delete", None)
        mock_memories.delete.assert_called_once()

    def test_put_invalid_namespace_raises(self, store) -> None:
        with pytest.raises(ValueError, match="valid scope"):
            store.put(("too_short",), "k", {"fact": "test"})


# ── Store: Delete ───────────────────────────────────────────────────────


class TestStoreDelete:
    def test_delete_calls_sdk(self, store, mock_memories) -> None:
        store.delete(("memories", "user_id", "alice"), "mem-to-delete")
        mock_memories.delete.assert_called_once()

    def test_delete_not_found_is_silent(self, store, mock_memories) -> None:
        mock_memories.delete.side_effect = Exception("404 NOT_FOUND")
        store.delete(("memories", "user_id", "alice"), "nonexistent")  # no raise


# ── Store: List Namespaces ──────────────────────────────────────────────


class TestStoreListNamespaces:
    def test_list_from_known_scopes(self, store, mock_memories) -> None:
        store._known_scopes.add((("user_id", "alice"),))
        store._known_scopes.add((("user_id", "bob"),))
        mock_memories.list.return_value = iter([])
        result = store.list_namespaces()
        assert len(result) == 2

    def test_list_discovers_from_sdk(self, store, mock_memories) -> None:
        # memories.list() is deprecated — list_namespaces now relies on _known_scopes only.
        # A scope only appears if it was previously used via search/put.
        # Verify that the old list() mock has no effect and scopes don't magically appear.
        mock_memories.list.return_value = iter([_mock_sdk_memory("m1", scope={"user_id": "carol"})])
        result = store.list_namespaces()
        # carol was never added to _known_scopes, so she should NOT appear
        assert ("memories", "user_id", "carol") not in result
        assert len(result) == 0

    def test_list_discovers_via_known_scopes(self, store, mock_memories) -> None:
        # The new way: scopes are tracked in _known_scopes after search/put
        store._known_scopes.add((("user_id", "carol"),))
        result = store.list_namespaces()
        assert ("memories", "user_id", "carol") in result

    def test_list_with_prefix_filter(self, store, mock_memories) -> None:
        store._known_scopes.add((("user_id", "alice"),))
        store._known_scopes.add((("agent_id", "bot1"),))
        mock_memories.list.return_value = iter([])
        result = store.list_namespaces(prefix=("memories", "user_id"))
        assert len(result) == 1

    def test_list_with_max_depth(self, store, mock_memories) -> None:
        store._known_scopes.add((("agent", "bot1"), ("user_id", "alice")))
        mock_memories.list.return_value = iter([])
        result = store.list_namespaces(max_depth=2)
        assert all(len(ns) <= 2 for ns in result)


# ── Batch ───────────────────────────────────────────────────────────────


class TestBatch:
    def test_batch_mixed_ops(self, store, mock_memories) -> None:
        mock_memories.get.return_value = _mock_sdk_memory("m1", "Fact")
        mock_memories.retrieve.return_value = [_mock_sdk_retrieved("m2", "Found")]
        mock_memories.list.return_value = iter([])

        from langgraph.store.base import GetOp, PutOp, SearchOp

        results = store.batch([
            GetOp(namespace=("memories", "user_id", "alice"), key="m1"),
            SearchOp(namespace_prefix=("memories", "user_id", "alice"), filter=None, limit=10, offset=0, query="test"),
            PutOp(namespace=("memories", "user_id", "alice"), key="k1", value={"fact": "new"}),
        ])
        assert len(results) == 3
        assert results[0] is not None  # GetOp result
        assert isinstance(results[1], list)  # SearchOp result
        assert results[2] is None  # PutOp result

    def test_batch_unsupported_op(self, store) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            store.batch(["not_an_op"])  # type: ignore


# ── Utilities ───────────────────────────────────────────────────────────


class TestUtilities:
    def test_scope_for_namespace(self, store) -> None:
        assert store.scope_for_namespace(("memories", "user_id", "alice", "session", "s1")) == {
            "user_id": "alice",
            "session": "s1",
        }

    def test_namespace_for_scope(self, store) -> None:
        assert store.namespace_for_scope({"user_id": "bob"}) == ("memories", "user_id", "bob")

    def test_namespace_for_scope_with_topic(self, store) -> None:
        assert store.namespace_for_scope({"user_id": "bob"}, topic="prefs") == (
            "memories", "user_id", "bob", "topic", "prefs"
        )


# ── Constructor config ──────────────────────────────────────────────────


class TestConstructorConfig:
    def test_defaults(self, store) -> None:
        assert store.default_top_k == 10
        assert store.min_similarity_score == 0.3

    def test_custom_config(self) -> None:
        _, s = _make_store(default_top_k=20, min_similarity_score=0.5)
        assert s.default_top_k == 20
        assert s.min_similarity_score == 0.5

    def test_engine_name_format(self, store) -> None:
        assert store._engine_name == "projects/test-project/locations/us-central1/reasoningEngines/123456"

    def test_uses_provided_client(self) -> None:
        mock_client = MagicMock()
        s = VertexMemoryBankStore(project_id="p", location="l", reasoning_engine_id="e", client=mock_client)
        assert s._client is mock_client

    @patch("langgraph_store_vertex_memorybank.store.vertexai.Client")
    def test_creates_client_from_params(self, mock_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        s = VertexMemoryBankStore(project_id="my-proj", location="eu-west1", reasoning_engine_id="789")
        mock_cls.assert_called_once_with(project="my-proj", location="eu-west1")
        assert s._client is mock_instance


# ── Alias ───────────────────────────────────────────────────────────────


class TestAlias:
    def test_alias_is_same_class(self) -> None:
        from langgraph_store_vertex_memorybank import LangGraphVertexAIMemoryBank
        assert LangGraphVertexAIMemoryBank is VertexMemoryBankStore


# ── Filter namespaces ───────────────────────────────────────────────────


class TestFilterNamespaces:
    def test_prefix_filter(self) -> None:
        ns_list = [("a", "b", "c"), ("a", "b", "d"), ("x", "y", "z")]
        result = _filter_namespaces(ns_list, (MatchCondition(match_type="prefix", path=("a", "b")),))
        assert len(result) == 2

    def test_suffix_filter(self) -> None:
        ns_list = [("a", "b", "c"), ("x", "y", "c")]
        result = _filter_namespaces(ns_list, (MatchCondition(match_type="suffix", path=("c",)),))
        assert len(result) == 2

    def test_wildcard_prefix(self) -> None:
        ns_list = [("a", "b", "c"), ("x", "b", "d")]
        result = _filter_namespaces(ns_list, (MatchCondition(match_type="prefix", path=("*", "b")),))
        assert len(result) == 2


class TestGenerateMemories:
    @patch("anthropic.Anthropic")
    def test_generate_memories(self, mock_anthropic, store, mock_memories):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content[0].text = '["fact 1", "fact 2"]'
        mock_client.messages.create.return_value = mock_response

        conversation = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        namespace = ("memories", "user_id", "alice")
        store.extract_memories_anthropic(conversation, namespace)

        mock_client.messages.create.assert_called_once()
        assert mock_memories.create.call_count == 2
        mock_memories.create.assert_any_call(
            name=store._engine_name, fact="fact 1", scope={"user_id": "alice"}
        )
        mock_memories.create.assert_any_call(
            name=store._engine_name, fact="fact 2", scope={"user_id": "alice"}
        )


class TestCreateCaptureNode:
    def test_create_capture_node(self, store):
        from langgraph_store_vertex_memorybank.store import create_capture_node

        namespace = ("memories", "user_id", "alice")
        capture_node = create_capture_node(store, namespace)
        assert callable(capture_node)

        with patch.object(store, "extract_memories_anthropic") as mock_generate_memories:
            state = {"messages": [{"role": "user", "content": "Hello"}]}
            capture_node(state)
            mock_generate_memories.assert_called_once_with(state["messages"], namespace)

def test_put_ttl():
    from langgraph.store.base import PutOp
    mock_client = MagicMock()
    # No default ttl mapping
    store = VertexMemoryBankStore("project", "location", "engine", mock_client)
    
    op = PutOp(namespace=("memories", "user_id", "123"), key="key1", value={"fact": "foo"}, ttl=3600)
    store.batch([op])
    
    mock_client.agent_engines.memories.create.assert_called_once()
    kwargs = mock_client.agent_engines.memories.create.call_args.kwargs
    assert kwargs["config"]["ttl"] == "3600s"

def test_namespace_ttl():
    from langgraph.store.base import PutOp
    mock_client = MagicMock()
    # With namespace ttl mapping
    store = VertexMemoryBankStore(
        "project", "location", "engine", mock_client,
        namespace_ttl={("memories", "user_id"): 86400, ("memories", "user_id", "VIP"): 99999}
    )
    
    # Matches ("memories", "user_id")
    op1 = PutOp(namespace=("memories", "user_id", "123", "topic", "x"), key="key1", value={"fact": "foo"})
    # Matches ("memories", "user_id", "VIP") -> longer prefix
    op2 = PutOp(namespace=("memories", "user_id", "VIP", "topic", "x"), key="key2", value={"fact": "bar"})
    # Overridden by op.ttl
    op3 = PutOp(namespace=("memories", "user_id", "123"), key="key3", value={"fact": "baz"}, ttl=123)
    
    store.batch([op1, op2, op3])
    
    calls = mock_client.agent_engines.memories.create.call_args_list
    assert len(calls) == 3
    assert calls[0].kwargs["config"]["ttl"] == "86400s"
    assert calls[1].kwargs["config"]["ttl"] == "99999s"
    assert calls[2].kwargs["config"]["ttl"] == "123s"



def test_aput_ttl():
    import asyncio
    from langgraph.store.base import PutOp
    mock_client = AsyncMock()
    store = VertexMemoryBankStore("project", "location", "engine", mock_client, namespace_ttl={("memories",): 100})
    
    op = PutOp(namespace=("memories", "user_id", "abc"), key="key1", value={"fact": "foo"})
    asyncio.run(store.abatch([op]))
    
    mock_client.aio.agent_engines.memories.create.assert_called_once()
    kwargs = mock_client.aio.agent_engines.memories.create.call_args.kwargs
    assert kwargs["config"]["ttl"] == "100s"

class TestRevisions:
    def test_list_revisions(self):
        mock_memories, store = _make_store()
        store._client.agent_engines.memories.revisions.list.return_value = ["rev1", "rev2"]
        revs = store.list_revisions(("memories", "user", "alice"), "my-key")
        assert revs == ["rev1", "rev2"]
        store._client.agent_engines.memories.revisions.list.assert_called_once_with(name="projects/test-project/locations/us-central1/reasoningEngines/123456/memories/my-key")

    def test_get_revision(self):
        mock_memories, store = _make_store()
        store._client.agent_engines.memories.revisions.get.return_value = "rev1"
        rev = store.get_revision(("memories", "user", "alice"), "my-key", "my-rev")
        assert rev == "rev1"
        store._client.agent_engines.memories.revisions.get.assert_called_once_with(name="projects/test-project/locations/us-central1/reasoningEngines/123456/memories/my-key/revisions/my-rev")

    def test_rollback(self):
        mock_memories, store = _make_store()
        store._client.agent_engines.memories.rollback.return_value = "op1"
        res = store.rollback(("memories", "user", "alice"), "my-key", "my-rev")
        assert res == "op1"
        store._client.agent_engines.memories.rollback.assert_called_once_with(name="projects/test-project/locations/us-central1/reasoningEngines/123456/memories/my-key", target_revision_id="my-rev")

    @pytest.mark.asyncio
    async def test_alist_revisions(self):
        mock_memories, store = _make_store()
        async def mock_list(*args, **kwargs):
            async def pager():
                yield "rev1"
                yield "rev2"
            return pager()
        store._client.aio.agent_engines.memories.revisions.list = mock_list
        revs = await store.alist_revisions(("memories", "user", "alice"), "my-key")
        assert revs == ["rev1", "rev2"]

    @pytest.mark.asyncio
    async def test_aget_revision(self):
        mock_memories, store = _make_store()
        # Mock as async function
        async def mock_get(name):
            return "rev1"
        store._client.aio.agent_engines.memories.revisions.get = mock_get
        rev = await store.aget_revision(("memories", "user", "alice"), "my-key", "my-rev")
        assert rev == "rev1"

    @pytest.mark.asyncio
    async def test_arollback(self):
        mock_memories, store = _make_store()
        async def mock_rollback(name, target_revision_id):
            return "op1"
        store._client.aio.agent_engines.memories.rollback = mock_rollback
        res = await store.arollback(("memories", "user", "alice"), "my-key", "my-rev")
        assert res == "op1"

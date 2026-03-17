"""Unit tests for nodes.py — create_recall_node, create_capture_node, helpers."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_store_vertex_memorybank.nodes import (
    _get_user_id,
    _last_human_text,
    create_capture_node,
    create_recall_node,
    messages_to_events,
)
from langgraph_store_vertex_memorybank.store import VertexMemoryBankStore

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_store() -> VertexMemoryBankStore:
    with patch("langgraph_store_vertex_memorybank.store.vertexai.Client"):
        s = VertexMemoryBankStore(
            project_id="test-project",
            location="us-central1",
            reasoning_engine_id="123456",
        )
    s._client = MagicMock()
    s._client.agent_engines.memories = MagicMock()
    s._client.aio.agent_engines.memories = MagicMock()
    return s


def _config(user_id: str = "alice") -> dict[str, Any]:
    return {"configurable": {"user_id": user_id}}


# ── _get_user_id ────────────────────────────────────────────────────────


def test_get_user_id_success():
    assert _get_user_id({"configurable": {"user_id": "bob"}}) == "bob"


def test_get_user_id_missing():
    with pytest.raises(ValueError, match="user_id not found"):
        _get_user_id({"configurable": {}})


def test_get_user_id_no_configurable():
    with pytest.raises(ValueError, match="user_id not found"):
        _get_user_id({})


# ── _last_human_text ────────────────────────────────────────────────────


def test_last_human_text_finds_last():
    msgs = [
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
    ]
    assert _last_human_text(msgs) == "second"


def test_last_human_text_no_human():
    msgs = [AIMessage(content="reply"), SystemMessage(content="sys")]
    assert _last_human_text(msgs) is None


def test_last_human_text_empty():
    assert _last_human_text([]) is None


# ── messages_to_events ──────────────────────────────────────────────────


def test_messages_to_events_basic():
    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    events = messages_to_events(msgs)
    assert len(events) == 2
    assert events[0] == {"content": {"role": "user", "parts": [{"text": "Hello"}]}}
    assert events[1] == {"content": {"role": "model", "parts": [{"text": "Hi there"}]}}


def test_messages_to_events_skips_system():
    msgs = [
        SystemMessage(content="You are helpful"),
        HumanMessage(content="Hi"),
        AIMessage(content="Hello"),
    ]
    events = messages_to_events(msgs)
    assert len(events) == 2
    assert events[0]["content"]["role"] == "user"


def test_messages_to_events_max_turns():
    msgs = [HumanMessage(content=f"msg-{i}") for i in range(20)]
    events = messages_to_events(msgs, max_turns=3)
    assert len(events) == 3
    assert events[0]["content"]["parts"][0]["text"] == "msg-17"


def test_messages_to_events_empty():
    assert messages_to_events([]) == []


def test_messages_to_events_only_system():
    msgs = [SystemMessage(content="system only")]
    assert messages_to_events(msgs) == []


# ── create_recall_node (system mode) ────────────────────────────────────


def test_recall_node_system_mode():
    store = _make_store()
    # Mock search to return results
    mock_item = MagicMock()
    mock_item.value = {"fact": "User likes Python"}
    store._client.agent_engines.memories.retrieve.return_value = []

    with patch.object(store, "search", return_value=[mock_item]):
        recall = create_recall_node(store, recall_mode="system", top_k=3)
        result = recall(
            {"messages": [HumanMessage(content="What do I like?")]},
            _config(),
        )

    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, SystemMessage)
    assert "User likes Python" in msg.content


def test_recall_node_state_mode():
    store = _make_store()
    mock_item = MagicMock()
    mock_item.value = {"fact": "Prefers dark mode"}

    with patch.object(store, "search", return_value=[mock_item]):
        recall = create_recall_node(store, recall_mode="state", state_key="mem")
        result = recall(
            {"messages": [HumanMessage(content="What theme?")]},
            _config(),
        )

    assert "mem" in result
    assert "Prefers dark mode" in result["mem"]


def test_recall_node_no_messages():
    store = _make_store()
    recall = create_recall_node(store)
    result = recall({"messages": []}, _config())
    assert result == {}


def test_recall_node_no_user_id():
    store = _make_store()
    recall = create_recall_node(store)
    result = recall(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {}},
    )
    assert result == {}


def test_recall_node_no_human_message():
    store = _make_store()
    recall = create_recall_node(store)
    result = recall(
        {"messages": [AIMessage(content="Just AI talking")]},
        _config(),
    )
    assert result == {}


def test_recall_node_no_results():
    store = _make_store()
    with patch.object(store, "search", return_value=[]):
        recall = create_recall_node(store)
        result = recall(
            {"messages": [HumanMessage(content="hi")]},
            _config(),
        )
    assert result == {}


def test_recall_node_search_fails():
    store = _make_store()
    with patch.object(store, "search", side_effect=Exception("API down")):
        recall = create_recall_node(store)
        result = recall(
            {"messages": [HumanMessage(content="hi")]},
            _config(),
        )
    assert result == {}


def test_recall_node_custom_prefix():
    store = _make_store()
    mock_item = MagicMock()
    mock_item.value = {"fact": "Loves hiking"}

    with patch.object(store, "search", return_value=[mock_item]):
        recall = create_recall_node(store, system_prefix="Memory: ")
        result = recall(
            {"messages": [HumanMessage(content="hobbies?")]},
            _config(),
        )

    assert result["messages"][0].content.startswith("Memory: ")


# ── create_capture_node ─────────────────────────────────────────────────


def test_capture_node_blocking():
    store = _make_store()
    store.generate_memories = MagicMock()

    capture = create_capture_node(store, fire_and_forget=False)
    result = capture(
        {"messages": [HumanMessage(content="I love Python"), AIMessage(content="Great!")]},
        _config(),
    )

    assert result == {}
    store.generate_memories.assert_called_once()
    call_kwargs = store.generate_memories.call_args
    assert call_kwargs.kwargs["scope"] == {"user_id": "alice"}
    events = call_kwargs.kwargs["events"]
    assert len(events) == 2
    assert events[0]["content"]["role"] == "user"
    assert events[1]["content"]["role"] == "model"


def test_capture_node_fire_and_forget():
    store = _make_store()
    called = threading.Event()

    def mock_generate(**kwargs: Any) -> None:
        called.set()

    store.generate_memories = mock_generate  # type: ignore[assignment]

    capture = create_capture_node(store, fire_and_forget=True)
    result = capture(
        {"messages": [HumanMessage(content="I'm vegan"), AIMessage(content="Noted!")]},
        _config(),
    )

    assert result == {}
    # Wait for background thread
    assert called.wait(timeout=2.0), "Background thread did not run"


def test_capture_node_too_few_messages():
    store = _make_store()
    store.generate_memories = MagicMock()

    capture = create_capture_node(store, fire_and_forget=False)
    result = capture(
        {"messages": [HumanMessage(content="hi")]},
        _config(),
    )

    assert result == {}
    store.generate_memories.assert_not_called()


def test_capture_node_no_user_id():
    store = _make_store()
    store.generate_memories = MagicMock()

    capture = create_capture_node(store, fire_and_forget=False)
    result = capture(
        {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]},
        {"configurable": {}},
    )

    assert result == {}
    store.generate_memories.assert_not_called()


def test_capture_node_skips_system_messages():
    store = _make_store()
    store.generate_memories = MagicMock()

    capture = create_capture_node(store, fire_and_forget=False)
    capture(
        {
            "messages": [
                SystemMessage(content="You are helpful"),
                HumanMessage(content="I like tea"),
                AIMessage(content="Nice!"),
            ]
        },
        _config(),
    )

    events = store.generate_memories.call_args.kwargs["events"]
    assert len(events) == 2
    roles = [e["content"]["role"] for e in events]
    assert "system" not in roles


def test_capture_node_max_turns():
    store = _make_store()
    store.generate_memories = MagicMock()

    msgs: list[Any] = []
    for i in range(20):
        msgs.append(HumanMessage(content=f"user-{i}"))
        msgs.append(AIMessage(content=f"ai-{i}"))

    capture = create_capture_node(store, fire_and_forget=False, max_turns=4)
    capture({"messages": msgs}, _config())

    events = store.generate_memories.call_args.kwargs["events"]
    assert len(events) == 4


def test_capture_node_handles_generate_error():
    """generate_memories failure should not raise — just log a warning."""
    store = _make_store()
    store.generate_memories = MagicMock(side_effect=Exception("API error"))

    capture = create_capture_node(store, fire_and_forget=False)
    # Should not raise
    result = capture(
        {"messages": [HumanMessage(content="test"), AIMessage(content="reply")]},
        _config(),
    )
    assert result == {}


def test_capture_node_only_system_messages():
    """If all messages are system messages, events is empty → skip."""
    store = _make_store()
    store.generate_memories = MagicMock()

    capture = create_capture_node(store, fire_and_forget=False)
    result = capture(
        {
            "messages": [
                SystemMessage(content="sys1"),
                SystemMessage(content="sys2"),
                SystemMessage(content="sys3"),
            ]
        },
        _config(),
    )

    assert result == {}
    store.generate_memories.assert_not_called()


# ── generate_memories on store ──────────────────────────────────────────


def test_generate_memories_calls_sdk():
    store = _make_store()
    mock_generate = store._client.agent_engines.memories.generate
    mock_generate.return_value = MagicMock()

    events = [
        {"content": {"role": "user", "parts": [{"text": "I live in Portland"}]}},
        {"content": {"role": "model", "parts": [{"text": "Nice city!"}]}},
    ]
    store.generate_memories(scope={"user_id": "alice"}, events=events)

    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["name"] == store._engine_name
    assert call_kwargs["scope"] == {"user_id": "alice"}
    assert call_kwargs["direct_contents_source"] == {"events": events}
    assert call_kwargs["config"] == {"wait_for_completion": True}


def test_generate_memories_no_wait():
    store = _make_store()
    mock_generate = store._client.agent_engines.memories.generate

    store.generate_memories(
        scope={"user_id": "bob"},
        events=[{"content": {"role": "user", "parts": [{"text": "hi"}]}}],
        wait_for_completion=False,
    )

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["config"] == {"wait_for_completion": False}


def test_generate_memories_tracks_scope():
    store = _make_store()
    store._client.agent_engines.memories.generate.return_value = MagicMock()

    store.generate_memories(
        scope={"user_id": "carol"},
        events=[{"content": {"role": "user", "parts": [{"text": "test"}]}}],
    )

    assert (("user_id", "carol"),) in store._known_scopes


@pytest.mark.asyncio
async def test_agenerate_memories_calls_sdk():
    store = _make_store()
    mock_agenerate = AsyncMock()
    store._client.aio.agent_engines.memories.generate = mock_agenerate

    events = [{"content": {"role": "user", "parts": [{"text": "async test"}]}}]
    await store.agenerate_memories(scope={"user_id": "dave"}, events=events)

    mock_agenerate.assert_called_once()
    call_kwargs = mock_agenerate.call_args.kwargs
    assert call_kwargs["scope"] == {"user_id": "dave"}

"""Unit tests for create_recall_node and create_capture_node."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_vertex_memorybank.nodes import (
    _get_user_id,
    _last_human_text,
    create_capture_node,
    create_recall_node,
)
from langgraph_vertex_memorybank.store import VertexMemoryBankStore


def _make_mock_store() -> tuple[MagicMock, VertexMemoryBankStore]:
    """Create a VertexMemoryBankStore with a mocked SDK client."""
    mock_memories = MagicMock()
    with patch("langgraph_vertex_memorybank.store.vertexai.Client"):
        store = VertexMemoryBankStore(
            project_id="test-project",
            location="us-central1",
            reasoning_engine_id="123456",
        )
    store._client = MagicMock()
    store._client.agent_engines.memories = mock_memories
    return mock_memories, store


def _config(user_id: str = "alice") -> dict[str, Any]:
    """Build a LangGraph config dict."""
    return {"configurable": {"user_id": user_id}}


# ── Helper tests ────────────────────────────────────────────────────────


class TestGetUserId:
    def test_extracts_user_id(self) -> None:
        assert _get_user_id({"configurable": {"user_id": "alice"}}) == "alice"

    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="user_id"):
            _get_user_id({"configurable": {}})

    def test_empty_config_raises(self) -> None:
        with pytest.raises(ValueError, match="user_id"):
            _get_user_id({})


class TestLastHumanText:
    def test_finds_last(self) -> None:
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="reply"),
            HumanMessage(content="second"),
        ]
        assert _last_human_text(messages) == "second"

    def test_returns_none_if_no_human(self) -> None:
        messages = [AIMessage(content="only ai")]
        assert _last_human_text(messages) is None

    def test_returns_none_for_empty(self) -> None:
        assert _last_human_text([]) is None


# ── Recall node tests ──────────────────────────────────────────────────


class TestRecallNode:
    def test_recall_system_mode(self) -> None:
        mock_client, store = _make_mock_store()

        # Mock search to return results
        search_item = MagicMock()
        search_item.value = {"fact": "Likes Python"}

        with patch.object(store, "search", return_value=[search_item]):
            recall = create_recall_node(store, recall_mode="system", top_k=3)
            result = recall(
                {"messages": [HumanMessage(content="What do I like?")]},
                _config("alice"),
            )

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, SystemMessage)
        assert "Likes Python" in msg.content

    def test_recall_state_mode(self) -> None:
        mock_client, store = _make_mock_store()
        search_item = MagicMock()
        search_item.value = {"fact": "Lives in Portland"}

        with patch.object(store, "search", return_value=[search_item]):
            recall = create_recall_node(store, recall_mode="state", state_key="mem_ctx")
            result = recall(
                {"messages": [HumanMessage(content="Where do I live?")]},
                _config("bob"),
            )

        assert "mem_ctx" in result
        assert "Lives in Portland" in result["mem_ctx"]

    def test_recall_no_messages_returns_empty(self) -> None:
        _, store = _make_mock_store()
        recall = create_recall_node(store)
        result = recall({"messages": []}, _config())
        assert result == {}

    def test_recall_no_user_id_returns_empty(self) -> None:
        _, store = _make_mock_store()
        recall = create_recall_node(store)
        result = recall(
            {"messages": [HumanMessage(content="hi")]},
            {"configurable": {}},
        )
        assert result == {}

    def test_recall_no_results_returns_empty(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "search", return_value=[]):
            recall = create_recall_node(store)
            result = recall(
                {"messages": [HumanMessage(content="hi")]},
                _config(),
            )
        assert result == {}

    def test_recall_search_error_returns_empty(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "search", side_effect=Exception("boom")):
            recall = create_recall_node(store)
            result = recall(
                {"messages": [HumanMessage(content="hi")]},
                _config(),
            )
        assert result == {}

    def test_recall_custom_prefix(self) -> None:
        _, store = _make_mock_store()
        search_item = MagicMock()
        search_item.value = {"fact": "Test fact"}

        with patch.object(store, "search", return_value=[search_item]):
            recall = create_recall_node(
                store,
                system_prefix="User context:\n",
            )
            result = recall(
                {"messages": [HumanMessage(content="hi")]},
                _config(),
            )

        assert "User context:" in result["messages"][0].content

    def test_recall_filters_empty_facts(self) -> None:
        _, store = _make_mock_store()
        items = [
            MagicMock(value={"fact": "Real fact"}),
            MagicMock(value={"fact": ""}),
            MagicMock(value={}),
        ]
        with patch.object(store, "search", return_value=items):
            recall = create_recall_node(store)
            result = recall(
                {"messages": [HumanMessage(content="hi")]},
                _config(),
            )
        # Only the real fact should appear
        assert "Real fact" in result["messages"][0].content
        assert result["messages"][0].content.count("- ") == 1


# ── Capture node tests ─────────────────────────────────────────────────


class TestCaptureNode:
    def test_capture_calls_generate(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "generate_memories") as mock_gen:
            capture = create_capture_node(store, fire_and_forget=False)
            result = capture(
                {
                    "messages": [
                        HumanMessage(content="I live in Portland"),
                        AIMessage(content="Nice city!"),
                    ]
                },
                _config("alice"),
            )

        assert result == {}
        mock_gen.assert_called_once()
        gen_call = mock_gen.call_args
        # Scope should be alice
        scope = gen_call.kwargs.get("scope")
        assert scope == {"user_id": "alice"}
        # Events should have user + model messages
        events = gen_call.kwargs.get("events")
        assert len(events) == 2
        assert events[0]["content"]["role"] == "user"
        assert events[1]["content"]["role"] == "model"

    def test_capture_with_topics(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "generate_memories") as mock_gen:
            capture = create_capture_node(
                store, topics=["prefs", "facts"], fire_and_forget=False
            )
            capture(
                {
                    "messages": [
                        HumanMessage(content="hello"),
                        AIMessage(content="hi"),
                    ]
                },
                _config(),
            )

        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs.get("topics") == ["prefs", "facts"]

    def test_capture_skips_system_messages(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "generate_memories") as mock_gen:
            capture = create_capture_node(store, fire_and_forget=False)
            capture(
                {
                    "messages": [
                        SystemMessage(content="You are helpful"),
                        HumanMessage(content="hello"),
                        AIMessage(content="hi"),
                    ]
                },
                _config(),
            )

        events = mock_gen.call_args.args[1] if len(mock_gen.call_args.args) > 1 else mock_gen.call_args.kwargs.get("events", [])
        roles = [e["content"]["role"] for e in events]
        assert "system" not in roles
        assert roles == ["user", "model"]

    def test_capture_too_few_messages_returns_empty(self) -> None:
        _, store = _make_mock_store()
        capture = create_capture_node(store, fire_and_forget=False)
        result = capture(
            {"messages": [HumanMessage(content="hi")]},
            _config(),
        )
        assert result == {}

    def test_capture_no_user_id_returns_empty(self) -> None:
        _, store = _make_mock_store()
        capture = create_capture_node(store, fire_and_forget=False)
        result = capture(
            {
                "messages": [
                    HumanMessage(content="hi"),
                    AIMessage(content="hello"),
                ]
            },
            {"configurable": {}},
        )
        assert result == {}

    def test_capture_fire_and_forget(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "generate_memories") as mock_gen:
            capture = create_capture_node(store, fire_and_forget=True)
            result = capture(
                {
                    "messages": [
                        HumanMessage(content="hi"),
                        AIMessage(content="hello"),
                    ]
                },
                _config(),
            )

        assert result == {}
        # Give the background thread time to run
        time.sleep(0.2)
        mock_gen.assert_called_once()

    def test_capture_max_turns(self) -> None:
        _, store = _make_mock_store()
        messages = [
            HumanMessage(content=f"msg {i}") if i % 2 == 0 else AIMessage(content=f"reply {i}")
            for i in range(20)
        ]
        with patch.object(store, "generate_memories") as mock_gen:
            capture = create_capture_node(store, max_turns=4, fire_and_forget=False)
            capture({"messages": messages}, _config())

        events = mock_gen.call_args.args[1] if len(mock_gen.call_args.args) > 1 else mock_gen.call_args.kwargs.get("events", [])
        assert len(events) == 4  # only last 4 messages

    def test_capture_error_does_not_raise(self) -> None:
        _, store = _make_mock_store()
        with patch.object(store, "generate_memories", side_effect=Exception("boom")):
            capture = create_capture_node(store, fire_and_forget=False)
            # Should not raise
            result = capture(
                {
                    "messages": [
                        HumanMessage(content="hi"),
                        AIMessage(content="hello"),
                    ]
                },
                _config(),
            )
        assert result == {}

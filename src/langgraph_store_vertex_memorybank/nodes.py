"""Pre-built LangGraph nodes for automatic memory recall and capture.

Factory functions that create graph nodes wiring Memory Bank into a
LangGraph agent's conversation flow:

- :func:`create_recall_node` — searches memories and injects them as a
  ``SystemMessage`` or state field before the agent responds.
- :func:`create_capture_node` — extracts memories from the conversation
  via Memory Bank's LLM-powered ``generate_memories()``.

Example wiring::

    from langgraph.graph import StateGraph, START, END
    from langgraph_store_vertex_memorybank import (
        VertexMemoryBankStore,
        create_recall_node,
        create_capture_node,
    )

    store = VertexMemoryBankStore(...)
    recall = create_recall_node(store, recall_mode="system", top_k=5)
    capture = create_capture_node(store, fire_and_forget=True)

    builder = StateGraph(MessagesState)
    builder.add_node("recall", recall)
    builder.add_node("agent", agent_fn)
    builder.add_node("capture", capture)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "agent")
    builder.add_edge("agent", "capture")
    builder.add_edge("capture", END)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langgraph_store_vertex_memorybank.store import VertexMemoryBankStore

logger = logging.getLogger(__name__)


def _get_user_id(config: dict[str, Any]) -> str:
    """Extract ``user_id`` from LangGraph config.

    Args:
        config: LangGraph invocation config.

    Returns:
        The user_id string.

    Raises:
        ValueError: If user_id is not in ``config["configurable"]``.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if not user_id:
        raise ValueError(
            "user_id not found in config['configurable']. "
            "Pass it when invoking the graph: "
            'graph.invoke(inputs, {"configurable": {"user_id": "alice"}})'
        )
    return str(user_id)


def _last_human_text(messages: list[BaseMessage]) -> str | None:
    """Get the text content of the last ``HumanMessage``."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return None


def messages_to_events(
    messages: list[BaseMessage],
    *,
    max_turns: int = 10,
) -> list[dict[str, Any]]:
    """Convert LangChain messages to Vertex AI Memory Bank event format.

    Skips ``SystemMessage`` (injected context, not real conversation).
    Maps ``HumanMessage`` → ``"user"``, ``AIMessage`` → ``"model"``.

    Args:
        messages: LangChain message list.
        max_turns: Max recent messages to include.

    Returns:
        List of event dicts in Memory Bank format::

            [{"content": {"role": "user", "parts": [{"text": "..."}]}}]
    """
    events: list[dict[str, Any]] = []
    for msg in messages[-max_turns:]:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "model"
        else:
            continue
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        if text:
            events.append({"content": {"role": role, "parts": [{"text": text}]}})
    return events


def create_recall_node(
    store: VertexMemoryBankStore,
    *,
    recall_mode: Literal["system", "state"] = "system",
    top_k: int = 5,
    system_prefix: str = "You know the following facts about this user:\n",
    state_key: str = "memory_context",
) -> Callable[..., dict[str, Any]]:
    """Create a LangGraph node that recalls memories before the agent responds.

    Searches Memory Bank using the last user message as the query, then
    injects relevant memories into the conversation.

    Wire before your agent: ``START → recall → agent → capture → END``

    Args:
        store: :class:`VertexMemoryBankStore` instance.
        recall_mode: How to inject memories:

            - ``"system"`` (default): Prepend a ``SystemMessage`` with facts.
            - ``"state"``: Put facts into ``state[state_key]``.
        top_k: Number of memories to retrieve.
        system_prefix: Text prepended to the memory list in system mode.
        state_key: State field name when using ``recall_mode="state"``.

    Returns:
        A callable suitable for ``builder.add_node("recall", recall)``.

    Example::

        recall = create_recall_node(store, recall_mode="system", top_k=5)
        builder.add_node("recall", recall)
    """

    def recall_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {}

        try:
            user_id = _get_user_id(config)
        except ValueError as e:
            logger.warning("Recall skipped: %s", e)
            return {}

        query = _last_human_text(messages)
        if not query:
            return {}

        namespace = store.namespace_for_scope({"user_id": user_id})

        try:
            results = store.search(namespace, query=query, limit=top_k)
        except Exception as e:
            logger.warning("Memory recall failed: %s", e)
            return {}

        if not results:
            return {}

        facts = [r.value.get("fact", "") for r in results if r.value.get("fact")]
        if not facts:
            return {}

        memory_text = "\n".join(f"- {fact}" for fact in facts)

        if recall_mode == "system":
            system_msg = SystemMessage(content=f"{system_prefix}{memory_text}")
            return {"messages": [system_msg]}
        else:
            return {state_key: memory_text}

    return recall_node


def create_capture_node(
    store: VertexMemoryBankStore,
    *,
    fire_and_forget: bool = True,
    max_turns: int = 10,
) -> Callable[..., dict[str, Any]]:
    """Create a LangGraph node that captures memories from conversation.

    Converts recent messages to Memory Bank event format and calls
    :meth:`~VertexMemoryBankStore.generate_memories` to extract facts.
    By default runs in a background thread so it doesn't block the response.

    Wire after your agent: ``START → recall → agent → capture → END``

    Args:
        store: :class:`VertexMemoryBankStore` instance.
        fire_and_forget: If True (default), run generation in a background
            thread so it doesn't block the response.
        max_turns: Max recent messages to send for extraction.

    Returns:
        A callable suitable for ``builder.add_node("capture", capture)``.

    Example::

        capture = create_capture_node(store, fire_and_forget=True)
        builder.add_node("capture", capture)
    """

    def capture_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        messages: list[BaseMessage] = state.get("messages", [])
        if len(messages) < 2:
            return {}

        try:
            user_id = _get_user_id(config)
        except ValueError as e:
            logger.warning("Capture skipped: %s", e)
            return {}

        scope = {"user_id": user_id}
        events = messages_to_events(messages, max_turns=max_turns)
        if not events:
            return {}

        def _generate() -> None:
            try:
                store.generate_memories(scope=scope, events=events)
            except Exception as e:
                logger.warning("Memory capture failed: %s", e)

        if fire_and_forget:
            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()
            logger.debug("Memory capture started in background thread")
        else:
            _generate()

        return {}

    return capture_node

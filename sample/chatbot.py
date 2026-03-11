"""Sample chatbot using LangGraph + Vertex AI Memory Bank + Gemini 3.1 Flash.

Demonstrates:
    - Multi-turn conversation with long-term memory
    - Cross-session memory recall
    - LLM-powered memory extraction from conversation turns
    - Per-user scoped memory isolation

Usage:
    # Set your GCP project
    export GCP_PROJECT_ID=your-project
    export REASONING_ENGINE_ID=your-engine-id

    # Run
    python sample/chatbot.py

    # Or with custom user
    python sample/chatbot.py --user alice
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

# Add parent to path for local dev
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langgraph_store_vertex_memorybank import VertexMemoryBankStore


# ── State ───────────────────────────────────────────────────────────────


class ChatState(TypedDict):
    """Graph state for the chatbot."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    memories: str  # Retrieved memories as context


# ── Configuration ───────────────────────────────────────────────────────

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "zaf-sandbox")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
ENGINE_ID = os.environ.get("REASONING_ENGINE_ID", "8505178738172887040")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-preview")


def create_chatbot(user_id: str = "demo-user"):
    """Create a LangGraph chatbot with Memory Bank integration.

    Args:
        user_id: User identifier for memory scoping.

    Returns:
        Compiled graph and store instance.
    """
    # Initialize the Memory Bank store
    store = VertexMemoryBankStore(
        project_id=PROJECT_ID,
        location=LOCATION,
        reasoning_engine_id=ENGINE_ID,
    )

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0.7,
    )

    # Memory namespace for this user
    user_ns = ("memories", "user_id", user_id)
    user_scope = {"user_id": user_id}

    # ── Graph nodes ─────────────────────────────────────────────────

    def recall_memories(state: ChatState) -> dict[str, Any]:
        """Retrieve relevant memories before generating a response."""
        messages = state["messages"]
        if not messages:
            return {"memories": ""}

        # Use the last user message as the search query
        last_msg = messages[-1]
        query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

        try:
            results = store.search(user_ns, query=query, limit=5)
            if results:
                facts = [r.value.get("fact", "") for r in results if r.value.get("fact")]
                memory_context = "\n".join(f"- {fact}" for fact in facts)
                return {"memories": memory_context}
        except Exception as e:
            print(f"  [Memory recall failed: {e}]")

        return {"memories": ""}

    def generate_response(state: ChatState) -> dict[str, Any]:
        """Generate a response using the LLM with memory context."""
        messages = list(state["messages"])
        memories = state.get("memories", "")

        # Build system message with memory context
        system_parts = ["You are a helpful, friendly assistant."]
        if memories:
            system_parts.append(
                f"\nYou remember the following about this user:\n{memories}\n"
                "Use these memories naturally in conversation when relevant. "
                "Don't list them explicitly unless asked."
            )

        system_msg = SystemMessage(content="\n".join(system_parts))

        # Prepend system message
        full_messages = [system_msg] + messages

        response = llm.invoke(full_messages)
        return {"messages": [response]}

    def save_memories(state: ChatState) -> dict[str, Any]:
        """Extract and save memories from the conversation turn."""
        messages = state["messages"]
        if len(messages) < 2:
            return {}

        # Get the last user-assistant exchange
        recent_events = []
        for msg in messages[-2:]:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            recent_events.append({
                "content": {"role": role, "parts": [{"text": text}]}
            })

        try:
            store.generate_memories(
                scope=user_scope,
                events=recent_events,
                wait_for_completion=False,  # Don't block on memory generation
            )
        except Exception as e:
            print(f"  [Memory save failed: {e}]")

        return {}

    # ── Build graph ─────────────────────────────────────────────────

    graph = StateGraph(ChatState)
    graph.add_node("recall", recall_memories)
    graph.add_node("respond", generate_response)
    graph.add_node("save", save_memories)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "respond")
    graph.add_edge("respond", "save")
    graph.add_edge("save", END)

    # Use MemorySaver for within-session conversation state
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled, store


def run_interactive(user_id: str = "demo-user") -> None:
    """Run an interactive chat session."""
    print(f"🧠 Memory Bank Chatbot (user: {user_id})")
    print(f"   Project: {PROJECT_ID} | Engine: {ENGINE_ID}")
    print(f"   Model: {MODEL}")
    print("   Type 'quit' to exit, 'memories' to list memories, 'new' for new session\n")

    graph, store = create_chatbot(user_id)
    session_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": f"session-{session_id}"}}
    user_ns = ("memories", "user_id", user_id)

    print(f"   Session: {session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            session_id = uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": f"session-{session_id}"}}
            print(f"\n--- New session: {session_id} ---\n")
            continue
        if user_input.lower() == "memories":
            try:
                results = store.search(user_ns, limit=20)
                if results:
                    print("\n📝 Stored memories:")
                    for r in results:
                        fact = r.value.get("fact", "?")
                        print(f"   • {fact}")
                    print()
                else:
                    print("\n   No memories stored yet.\n")
            except Exception as e:
                print(f"\n   Error listing memories: {e}\n")
            continue

        # Run the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "user_id": user_id, "memories": ""},
            config=config,
        )

        # Print the assistant response
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print(f"\nBot: {ai_messages[-1].content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory Bank Chatbot")
    parser.add_argument("--user", default="demo-user", help="User ID for memory scoping")
    args = parser.parse_args()
    run_interactive(args.user)

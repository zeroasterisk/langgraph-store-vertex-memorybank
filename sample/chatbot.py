"""Sample chatbot using LangGraph + Vertex AI Memory Bank + Gemini 3.1 Flash.

Demonstrates:
    - Multi-turn conversation with long-term memory
    - Cross-session memory recall (start new session, memories persist)
    - LLM-powered memory extraction from conversation turns
    - Per-user scoped memory isolation
    - Pre-built recall and capture nodes

Usage:
    # Set your GCP project (or use defaults)
    export GCP_PROJECT_ID=your-project
    export REASONING_ENGINE_ID=your-engine-id

    # Install deps
    pip install "langgraph-vertex-memorybank[sample]"

    # Run
    python sample/chatbot.py

    # With custom user
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

from langgraph_vertex_memorybank import (
    VertexMemoryBankStore,
    create_capture_node,
    create_recall_node,
)


# ── State ───────────────────────────────────────────────────────────────


class ChatState(TypedDict):
    """Graph state for the chatbot."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str


# ── Configuration ───────────────────────────────────────────────────────

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "zaf-sandbox")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
ENGINE_ID = os.environ.get("REASONING_ENGINE_ID", "8505178738172887040")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-preview")


def create_chatbot(user_id: str = "demo-user"):
    """Create a LangGraph chatbot with Memory Bank integration.

    Uses pre-built recall and capture nodes for clean graph wiring.

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
        default_top_k=5,
        min_similarity_score=0.3,
        topics=["preferences", "facts", "instructions"],
    )

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.7)

    # Pre-built nodes
    recall = create_recall_node(store, recall_mode="system", top_k=5)
    capture = create_capture_node(
        store,
        topics=["preferences", "facts", "instructions"],
        fire_and_forget=True,  # Don't block the response
    )

    def respond(state: ChatState) -> dict[str, Any]:
        """Generate a response using the LLM."""
        messages = list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Build graph: recall → respond → capture
    graph = StateGraph(ChatState)
    graph.add_node("recall", recall)
    graph.add_node("respond", respond)
    graph.add_node("capture", capture)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "respond")
    graph.add_edge("respond", "capture")
    graph.add_edge("capture", END)

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
    config = {"configurable": {"thread_id": f"session-{session_id}", "user_id": user_id}}
    user_ns = store.namespace_for_scope({"user_id": user_id})

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
            config = {"configurable": {"thread_id": f"session-{session_id}", "user_id": user_id}}
            print(f"\n--- New session: {session_id} ---\n")
            continue
        if user_input.lower() == "memories":
            try:
                results = store.search(user_ns, limit=20)
                if results:
                    print("\n📝 Stored memories:")
                    for r in results:
                        fact = r.value.get("fact", "?")
                        score = f" ({r.score:.2f})" if r.score else ""
                        print(f"   • {fact}{score}")
                    print()
                else:
                    print("\n   No memories stored yet.\n")
            except Exception as e:
                print(f"\n   Error listing memories: {e}\n")
            continue

        # Run the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
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

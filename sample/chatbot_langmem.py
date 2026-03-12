"""Chatbot with langmem tools + Vertex AI Memory Bank.

Shows the recommended pattern: langmem tools handle memory management,
VertexMemoryBankStore provides the persistent backend.

Usage:
    pip install langgraph-store-vertex-memorybank[sample]
    python chatbot_langmem.py
"""

from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool

from langgraph_store_vertex_memorybank import VertexMemoryBankStore

# 1. Create the store (swap this one line for managed memory)
store = VertexMemoryBankStore(
    project_id="my-project",
    location="us-central1",
    reasoning_engine_id="123456",
)

# 2. Create an agent with memory tools
agent = create_react_agent(
    "google_genai:gemini-2.5-flash",
    tools=[
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)

# 3. Chat — the agent manages memory automatically
if __name__ == "__main__":
    config = {"configurable": {"user_id": "demo-user"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that I prefer dark mode and live in Portland."}]},
        config=config,
    )
    print(response["messages"][-1].content)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What do you know about me?"}]},
        config=config,
    )
    print(response["messages"][-1].content)

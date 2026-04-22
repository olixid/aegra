"""Echo graph — deterministic, no LLM, fast.

Useful for testing webhook delivery without consuming API tokens.

Behaviour:
- Appends an ``assistant`` message that echoes the last user message.
- If the last user message content is exactly ``"__error__"`` (case-insensitive),
  the graph raises ``RuntimeError`` — triggering a ``status=error`` webhook.
- Runs complete instantly: no network calls, no randomness.

Input schema::

    {"messages": [{"role": "user", "content": "<text>"}]}

Output schema::

    {"messages": [..., {"role": "assistant", "content": "Echo: <text>"}]}
"""

from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import StateGraph, add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """Minimal state: just a list of messages."""

    messages: Annotated[list[AnyMessage], add_messages]


async def echo(state: State) -> dict[str, list[AIMessage]]:
    """Return the last user message prefixed with 'Echo:'."""
    messages = state.get("messages", [])
    last_content = messages[-1].content if messages else ""

    if str(last_content).strip().lower() == "__error__":
        raise RuntimeError("Deliberate error triggered by __error__ sentinel")

    return {
        "messages": [
            AIMessage(content=f"Echo: {last_content}"),
        ]
    }


builder = StateGraph(State)
builder.add_node("echo", echo)
builder.add_edge("__start__", "echo")
builder.add_edge("echo", "__end__")

graph = builder.compile(name="Webhook Echo")

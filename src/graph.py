from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage

from state import SherpaState
from tools import build_tools
from nodes import build_answer_node, redact_node, escalation_node


def escalation_condition(state):
    """After the tools node: did the retriever find nothing? -> escalate, else keep answering."""
    last = state["messages"][-1]  # the ToolMessage the retriever just produced
    if isinstance(last, ToolMessage) and "NO_RELEVANT_INFO_FOUND" in last.content:
        return "escalation_node"
    return "answer_node"


def build_graph(vector_store):
    """Wire the agent: redact -> answer <-> tools loop, with escalation when retrieval is empty."""
    tools = build_tools(vector_store)
    answer_node = build_answer_node(tools)

    graph = StateGraph(SherpaState)
    graph.add_node("redact_node", redact_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("escalation_node", escalation_node)

    graph.add_edge(START, "redact_node")
    graph.add_edge("redact_node", "answer_node")
    graph.add_conditional_edges("answer_node", tools_condition)     # answer_node -> tools OR END
    graph.add_conditional_edges(
        "tools",
        escalation_condition,
        {"escalation_node": "escalation_node", "answer_node": "answer_node"},
    )
    graph.add_edge("escalation_node", END)

    return graph.compile(checkpointer=InMemorySaver())

import os
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from state import SherpaState

DOC_GAPS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "doc_gaps.md")


def redact(text: str) -> str:
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)


def save_doc_gap(question: str) -> None:
    with open(DOC_GAPS_FILE, "a") as f:
        f.write(f"- {question}\n")


def build_answer_node(tools):
    """Agent node: an LLM that can choose to call the tools."""
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    def answer_node(state: SherpaState):
        system = SystemMessage(
            content="You are Sherpa. Use the retriever tool to answer questions about the document. "
            "If the retriever returns nothing relevant, say you don't know."
        )
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    return answer_node


def redact_node(state: SherpaState):
    last = state["messages"][-1]
    cleaned = redact(last.content)
    if cleaned != last.content:
        return {"messages": [HumanMessage(content=cleaned, id=last.id)]}  # same id = overwrite
    return {}  # nothing to redact


def escalation_node(state: SherpaState):
    """Runs when the retriever found nothing. Ask the human to approve filing a doc gap."""
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "your question",
    )
    decision = interrupt({
        "action": "email_doc_gap",
        "question": question,
        "prompt": f"I couldn't find anything about \"{question}\" in the docs. "
                  "Shall I file a doc-gap report so the team adds it?",
    })  # PAUSE for approval

    if decision == "approve":
        save_doc_gap(question)  # side effect AFTER interrupt
        text = "I couldn't find this in the docs, so I've filed a doc-gap report for the team. ✅"
    else:
        text = "I couldn't find this in the docs, and I won't escalate it."
    return {"messages": [AIMessage(content=text)]}

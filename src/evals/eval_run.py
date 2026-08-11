"""Run Sherpa against the eval dataset and score answers with an LLM-as-judge."""
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langsmith import Client

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path
from rag import load_pdf, build_vector_store
from graph import build_graph

FILE_PATH = "/Users/priyanshu.singh/Documents/Contentstack/Documents/PDFs/Agents-Engine Architecture.pdf"
DATASET_NAME = "sherpa-eval"

# index the PDF ONCE, then reuse for every example (slow + costly per-question otherwise)
vector_store = build_vector_store(load_pdf(FILE_PATH))
workflow = build_graph(vector_store)


def sherpa(inputs: dict) -> dict:
    """The target: run Sherpa on one question. Fresh thread_id so examples don't share memory."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 10}
    result = workflow.invoke({"messages": [HumanMessage(content=inputs["question"])]}, config)
    return {"answer": result["messages"][-1].content}


class Grade(BaseModel):
    """The judge's verdict — structured so we can aggregate the score."""
    score: int = Field(description="1 if the answer is correct, else 0", ge=0, le=1)
    feedback: str = Field(description="Short reason for the score")


judge = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Grade)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: is Sherpa's answer consistent with the expected answer?"""
    prompt = (
        "You are grading an answer against the expected answer.\n"
        f"Question: {inputs['question']}\n"
        f"Expected: {reference_outputs['answer']}\n"
        f"Sherpa answered: {outputs['answer']}\n\n"
        "Give score 1 if Sherpa's answer is factually consistent with the expected answer "
        "(exact wording doesn't matter), else 0."
    )
    grade = judge.invoke(prompt)
    return {"key": "correctness", "score": grade.score, "comment": grade.feedback}


if __name__ == "__main__":
    client = Client()
    client.evaluate(
        sherpa,
        data=DATASET_NAME,
        evaluators=[correctness],
        experiment_prefix="sherpa",
        max_concurrency=2,
    )
    print("Done — open LangSmith to see the scores.")

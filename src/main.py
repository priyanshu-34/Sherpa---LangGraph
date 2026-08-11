from dotenv import load_dotenv
load_dotenv()  # must run before importing graph (it builds the LLM at import time)

from langchain_core.messages import HumanMessage
from rag import load_pdf, build_vector_store
from graph import build_graph
from langgraph.types import Command

FILE_PATH = "/Users/priyanshu.singh/Documents/Contentstack/Documents/PDFs/Agents-Engine Architecture.pdf"

def streaming(stream):
    for message_chunk, metadata in stream:
        if message_chunk.content and metadata["langgraph_node"] == "answer_node":
            print(message_chunk.content, end="", flush=True)

    print()




def main():
    text = load_pdf(FILE_PATH)
    vector_store = build_vector_store(text)
    print(f"Indexing done. Ask Sherpa anything (type 'exit' to quit).")

    workflow = build_graph(vector_store)
    config = {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 10}

    while True:
        user_input = input("Ask LLM: ")
        if user_input == "exit":
            print("------DONE-----")
            break
        stream = workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="messages",  
        )
        streaming(stream)

        snap = workflow.get_state(config)
        while snap.interrupts:
            payload = snap.interrupts[0].value
            print(f"\n{payload['prompt']}")
            choice = input("Approve? (y/n): ")
            decision = "approve" if choice.lower() == "y" else "reject"

            resume_stream = workflow.stream(Command(resume=decision), config, stream_mode="messages")
            streaming(resume_stream)
            # escalation node's message is static (not streamed) — print it explicitly
            print(workflow.get_state(config).values["messages"][-1].content)
            snap = workflow.get_state(config)



if __name__ == "__main__":
    main()

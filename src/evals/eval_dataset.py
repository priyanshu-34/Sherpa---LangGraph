"""Run ONCE to create the LangSmith eval dataset (the answer key)."""
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

DATASET_NAME = "sherpa-eval"

# ~10 question -> expected-answer pairs about the Agents-Engine PDF.
# NOTE: verify/expand these against your actual PDF — I only saw the table of contents.
examples = [
    {"inputs": {"question": "What is agents-engine?"},
     "outputs": {"answer": "Contentstack's runtime for AI agents."}},
    {"inputs": {"question": "How many families of agents are there?"},
     "outputs": {"answer": "Five families of agents."}},
    {"inputs": {"question": "What object represents an agent?"},
     "outputs": {"answer": "The Mastra Agent object."}},
    {"inputs": {"question": "What gives an agent extra abilities?"},
     "outputs": {"answer": "Tools."}},
    # a "should refuse" case — not in the doc (tests Guardrail 1)
    {"inputs": {"question": "What is the capital of France?"},
     "outputs": {"answer": "I don't know / not covered in the document."}},
]


def main():
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists — skipping create.")
        return
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created '{DATASET_NAME}' with {len(examples)} examples.")


if __name__ == "__main__":
    main()

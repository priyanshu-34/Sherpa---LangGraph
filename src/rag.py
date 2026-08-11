from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def load_pdf(file_path: str) -> str:
    """Open the PDF and return all its text as one string."""
    reader = PdfReader(file_path)
    return "".join(page.extract_text() for page in reader.pages)


def build_vector_store(text: str) -> InMemoryVectorStore:
    """Split text, embed the chunks, store them, return the vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_texts(chunks)
    print(f"Indexed {len(chunks)} chunks.")
    return vector_store


def retrieve(vector_store: InMemoryVectorStore, query: str, k: int = 4) -> list[str]:
    """Return the k most relevant chunk texts for the query."""
    results = vector_store.similarity_search_with_score(query, k=k)

    return [{"page_content": doc.page_content, "score": score} for doc, score in results]

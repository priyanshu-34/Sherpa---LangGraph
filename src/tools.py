from langchain_core.tools import tool
from rag import retrieve


def build_tools(vector_store):
    """Build the retriever tool bound to this vector store."""

    @tool
    def retriever(query: str) -> str:
        """Search the document and return chunks relevant to the query."""
        results = retrieve(vector_store, query)

        chunks = [r["page_content"] for r in results if r["score"] > 0.4]

        if len(chunks)== 0 :
            prompt = "NO_RELEVANT_INFO_FOUND"
            chunks = [prompt]

        return "\n\n".join(chunks)

    return [retriever]

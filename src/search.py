import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.embedding import EmbeddingManager
from src.vector_store import VectorStore


DEFAULT_LLM_MODEL = "openai/gpt-oss-120b:free"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"


class RAGRetriever:
    """Query the vector store and return retrieved chunk records."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        query_embedding = self.embedding_manager.generate_query_embedding(query)
        results = self.vector_store.query(query_embedding=query_embedding.tolist(), top_k=top_k)

        retrieved_docs: list[dict[str, Any]] = []
        if not results.get("documents") or not results["documents"][0]:
            print("No documents found")
            return retrieved_docs

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
            similarity_score = 1 - distance
            if similarity_score >= score_threshold:
                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "rank": i + 1,
                    }
                )
        print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
        return retrieved_docs


def build_llm(
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    api_key_env: str = "OPENROUTER_API_KEY_RAG",
    base_url: str = DEFAULT_LLM_BASE_URL,
) -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {api_key_env}")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_rag_messages(query: str, context: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(
            content="Answer the question concisely using only the retrieved context. "
            "If the context is insufficient, say so explicitly."
        ),
        HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Answer:"
            )
        ),
    ]


def _build_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:50] + "...",
        }
        for doc in results
    ]


def answer_with_sources(
    query: str,
    retriever: RAGRetriever,
    llm: ChatOpenAI,
    top_k: int = 3,
    score_threshold: float = 0.0,
    return_context: bool = False,
) -> dict[str, Any]:
    results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0, "context": ""}

    context = "\n\n".join(doc["content"] for doc in results)
    response = llm.invoke(_build_rag_messages(query, context))
    output = {
        "answer": response.content,
        "sources": _build_sources(results),
        "confidence": max(doc["similarity_score"] for doc in results),
    }
    if return_context:
        output["context"] = context
    return output


def rag_simple(query: str, retriever: RAGRetriever, llm: ChatOpenAI, top_k: int = 3) -> str:
    return answer_with_sources(query=query, retriever=retriever, llm=llm, top_k=top_k)["answer"]


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm: ChatOpenAI,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> dict[str, Any]:
    return answer_with_sources(
        query=query,
        retriever=retriever,
        llm=llm,
        top_k=top_k,
        score_threshold=min_score,
        return_context=return_context,
    )

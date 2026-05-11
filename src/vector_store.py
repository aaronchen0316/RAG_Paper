import hashlib
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
from langchain_core.documents import Document


DEFAULT_COLLECTION_NAME = "pdf_documents"


class VectorStore:
    """Persist chunk text, metadata, and embeddings in ChromaDB."""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME, persist_directory: Path | str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.client: chromadb.PersistentClient | None = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PDF document embeddings for RAG",
                "hnsw:space": "cosine",
            },
        )
        print(f"Vector store initialized. Collection: {self.collection_name}")
        print(f"Existing documents in collection: {self.collection.count()}")

    @staticmethod
    def make_chunk_id(doc: Document, fallback_index: int = 0) -> str:
        metadata = dict(doc.metadata)
        source_file = Path(str(metadata.get("source_file", "unknown"))).stem
        page = metadata.get("page", 0)
        chunk_index = metadata.get("chunk_index", fallback_index)
        text_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:10]
        return f"{source_file}_p{page}_c{chunk_index}_{text_hash}"

    def upsert_documents(self, documents: Sequence[Document], embeddings: np.ndarray) -> int:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if self.collection is None:
            raise RuntimeError("Vector store collection is not initialized")

        print(f"Upserting {len(documents)} documents into vector store...")
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents_text: list[str] = []
        embeddings_list: list[list[float]] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            ids.append(self.make_chunk_id(doc, fallback_index=i))
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )
        print(f"Successfully upserted {len(documents)} documents into vector store")
        print(f"Total documents in collection: {self.collection.count()}")
        return len(ids)

    def query(self, query_embedding: Sequence[float], top_k: int = 5) -> dict[str, Any]:
        if self.collection is None:
            raise RuntimeError("Vector store collection is not initialized")
        return self.collection.query(query_embeddings=[list(query_embedding)], n_results=top_k)

    def count(self) -> int:
        if self.collection is None:
            raise RuntimeError("Vector store collection is not initialized")
        return int(self.collection.count())

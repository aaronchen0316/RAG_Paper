import argparse
from pathlib import Path
import sys

from src.data_loader import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    load_and_split_pdfs,
)
from src.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingManager
from src.search import answer_with_sources, build_llm, rag_advanced, RAGRetriever
from src.vector_store import DEFAULT_COLLECTION_NAME, VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdf"
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
DEFAULT_TOP_K = 3
DEFAULT_SCORE_THRESHOLD = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Paper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build or update the Chroma index from PDFs.")
    ingest_parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)

    ask_parser = subparsers.add_parser("ask", help="Ask one question against the existing vector store.")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ask_parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    ask_parser.add_argument("--advanced", action="store_true")
    return parser


def run_ingest(args: argparse.Namespace) -> int:
    chunks = load_and_split_pdfs(
        pdf_directory=args.pdf_dir,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    embedding_manager = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
    embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])
    vector_store = VectorStore(collection_name=DEFAULT_COLLECTION_NAME, persist_directory=DEFAULT_VECTOR_STORE_DIR)
    vector_store.upsert_documents(documents=chunks, embeddings=embeddings)
    print(f"Vector store now contains {vector_store.count()} chunks")
    return 0


def run_ask(args: argparse.Namespace) -> int:
    llm = build_llm()
    embedding_manager = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
    vector_store = VectorStore(collection_name=DEFAULT_COLLECTION_NAME, persist_directory=DEFAULT_VECTOR_STORE_DIR)
    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)

    if args.advanced:
        result = rag_advanced(
            query=args.query,
            retriever=retriever,
            llm=llm,
            top_k=args.top_k,
            min_score=args.score_threshold,
            return_context=False,
        )
        print("\nAnswer:\n")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(
                f"- {source['source']} (page {source['page']}, score={source['score']:.3f}) "
                f"{source['preview']}"
            )
        print(f"\nConfidence: {result['confidence']:.3f}")
        return 0

    result = answer_with_sources(
        query=args.query,
        retriever=retriever,
        llm=llm,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )
    print("\nAnswer:\n")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- {source['source']} (page {source['page']}, score={source['score']:.3f}) "
            f"{source['preview']}"
        )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            return run_ingest(args)
        if args.command == "ask":
            return run_ask(args)
        parser.error(f"Unsupported command: {args.command}")
        return 2
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

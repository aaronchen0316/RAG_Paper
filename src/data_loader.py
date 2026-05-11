from collections import defaultdict
from pathlib import Path
from typing import Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def discover_pdf_files(pdf_directory: Path | str) -> list[Path]:
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("**/*.pdf")) + sorted(pdf_dir.glob("**/*.PDF"))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for pdf_file in pdf_files:
        resolved = pdf_file.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(pdf_file)
    return deduped


def process_all_pdfs(pdf_directory: Path | str) -> list[Document]:
    documents: list[Document] = []
    pdf_files = discover_pdf_files(pdf_directory)
    print(f"Found {len(pdf_files)} PDF files to ingest")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"
        documents.extend(loaded)
        print(f" Loaded {len(loaded)} pages\n")

    print(f"Total pages loaded {len(documents)}")
    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(list(documents))
    chunk_counts: dict[tuple[str, str], int] = defaultdict(int)
    for doc in split_docs:
        source = str(doc.metadata.get("source_file", doc.metadata.get("source", "unknown")))
        page = str(doc.metadata.get("page", "unknown"))
        key = (source, page)
        doc.metadata["chunk_index"] = chunk_counts[key]
        chunk_counts[key] += 1

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    if split_docs:
        print("Example chunk:")
        print(f"Context: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    return split_docs


def load_and_split_pdfs(
    pdf_directory: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    documents = process_all_pdfs(pdf_directory)
    if not documents:
        raise ValueError(f"No PDF documents found in {Path(pdf_directory)}")
    return split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

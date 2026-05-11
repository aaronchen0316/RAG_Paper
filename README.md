# RAG_Paper

Simple RAG prototype for asking questions over a local collection of research papers.

The current script workflow does two things:
- `ingest`: load PDFs, split them into chunks, create embeddings, and store them in Chroma
- `ask`: retrieve relevant chunks and generate an answer with an OpenRouter-backed chat model

## Prerequisites

- Python environment with the packages in [requirements.txt]
- A working virtualenv for this repo
- `OPENROUTER_API_KEY_RAG` available in your shell environment or `.env`
- Network access the first time the embedding model is downloaded, unless it is already cached locally

The default embedding model is `all-MiniLM-L6-v2`.

## Data Layout

- PDFs are read from `data/pdf`
- Chroma data is stored in `data/vector_store`

You can override the PDF directory at ingest time with `--pdf-dir`.

## Commands

From the repo root:

Show help:

```bash
python main.py --help
```

Build or update the vector store from the default PDF directory:

```bash
python main.py ingest
```

Build or update the vector store from a custom PDF directory:

```bash
python main.py ingest --pdf-dir /path/to/pdfs
```

Ask a question:

```bash
python main.py ask "What is FLARE force field?"
```

Ask with custom retrieval settings:

```bash
python main.py ask \
  "Why do we use machine learning force field for Gallium Nitride crystal growth study?" \
  --top-k 5 \
  --score-threshold 0.1 \
  --advanced
```

## Notes

- `ingest` uses deterministic chunk IDs plus Chroma `upsert`, so reruns should update existing chunks instead of blindly duplicating them.
- `ask` requires a populated vector store, so run `ingest` first.
- The notebook at [pdf_loader.ipynb] is for initial development only

# TODO

## RAG notebook follow-up

1. Standardize the LLM interface.
Use one `ChatOpenAI` message-based pattern across both `rag_simple()` and `rag_advanced()` instead of mixing message calls and plain prompt-list calls.

2. Make vector-store ingestion idempotent.
Replace random chunk IDs with deterministic IDs derived from stable source metadata, then use `upsert` or an explicit rebuild flow.

3. Define retrieval semantics explicitly.
Choose and document the Chroma distance metric and any embedding normalization instead of relying on implicit defaults.

4. Strengthen chunk provenance.
Persist stable source metadata for each chunk, including source file, page, chunk index, and optionally a content hash.

5. Extract notebook logic into `src/` modules.
Split the pipeline into ingestion, chunking, embeddings, vector store, retrieval, and RAG generation modules so the notebook becomes a thin orchestration layer.

6. Add a small retrieval evaluation set.
Create a handful of representative questions and verify that the top retrieved chunks come from the expected PDFs and pages.

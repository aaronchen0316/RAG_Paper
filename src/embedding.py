from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """Generate document and query embeddings using SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        print(f"Loading embedding model: {self.model_name}")
        try:
            model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load embedding model. Ensure the model is cached locally "
                "or that network access is available for the first download."
            ) from exc
        print(f"Model loaded. Embedding dimension: {model.get_embedding_dimension()}")
        return model

    def generate_embeddings(self, texts: Sequence[str], show_progress_bar: bool = True) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(list(texts), show_progress_bar=show_progress_bar)
        print(f"Generated embedding shape: {embeddings.shape}")
        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        return self.generate_embeddings([query], show_progress_bar=False)[0]

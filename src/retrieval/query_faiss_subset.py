"""
This file loads the saved FAISS subset index and metadata, embeds a claim,
and retrieves the top-k most similar Wikipedia sentences.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_metadata(file_path: str) -> List[Dict[str, Any]]:
    """
    Load metadata records saved alongside the FAISS index.

    Args:
        file_path: Path to metadata JSON file.

    Returns:
        A list of metadata records.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a single embedding vector using NumPy.

    Args:
        vector: A 2D numpy array of shape (1, dim).

    Returns:
        Normalized vector.
    """
    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vector / norms


def search_claim(
    claim: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for the most relevant sentences to a claim.

    Args:
        claim: Input natural language claim.
        model: Sentence embedding model.
        index: Loaded FAISS index.
        metadata: Metadata corresponding to index rows.
        top_k: Number of results to return.

    Returns:
        A list of retrieved results with score and sentence metadata.
    """
    query_embedding = model.encode([claim], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")
    query_embedding = normalize_vector(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        record = metadata[idx]
        results.append({
            "score": float(score),
            "page": record["page"],
            "sentence_id": record["sentence_id"],
            "text": record["text"]
        })

    return results


if __name__ == "__main__":
    index_path = "data/index/wiki_subset.index"
    metadata_path = "data/index/wiki_subset_metadata.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading model...")
    model = SentenceTransformer(model_name)

    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)

    claim = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."

    print("\nClaim:", claim)
    print("\nTop results:")
    results = search_claim(claim, model, index, metadata, top_k=5)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Page: {result['page']}")
        print(f"Sentence ID: {result['sentence_id']}")
        print(f"Text: {result['text']}")

    # Temporary workaround for local macOS/FAISS shutdown segfault.
    import os
    os._exit(0)
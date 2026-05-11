"""
query_faiss_targeted_subset.py

Runs dense retrieval over the targeted Wikipedia subset using FAISS.

Pipeline:
    Claim
    → embed query
    → FAISS similarity search
    → optional reranking
    → return ranked evidence

Supports multiple retrievers:
    - MiniLM
    - BGE
    - E5

E5 requires special query formatting:
    "query: ..."
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# =========================================================
# METADATA LOADING
# =========================================================

def load_metadata(
    metadata_path: str
) -> List[Dict[str, Any]]:
    """
    Load metadata mapping FAISS ids back to sentences.
    """

    path = Path(metadata_path)

    with path.open("r", encoding="utf-8") as file:

        metadata = json.load(file)

    return metadata


# =========================================================
# QUERY PREPARATION
# =========================================================

def prepare_query(
    claim: str,
    retriever_id: str,
) -> str:
    """
    Format query for retriever-specific requirements.

    E5 models require:
        query: ...

    Other retrievers use raw text.
    """

    if retriever_id == "e5":

        return "query: " + claim

    return claim


# =========================================================
# QUERY EMBEDDING
# =========================================================

def embed_query(
    claim: str,
    model: SentenceTransformer,
    retriever_id: str,
) -> np.ndarray:
    """
    Generate normalized query embedding.
    """

    formatted_claim = prepare_query(
        claim,
        retriever_id
    )

    embedding = model.encode(
        formatted_claim,
        convert_to_numpy=True,
    )

    embedding = embedding.astype("float32")

    norm = np.linalg.norm(embedding)

    norm = max(norm, 1e-12)

    embedding = embedding / norm

    return embedding.reshape(1, -1)


# =========================================================
# DENSE RETRIEVAL
# =========================================================

def retrieve_candidates(
    claim: str,
    model: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    retriever_id: str,
    top_k: int = 50,
):
    """
    Retrieve top-k dense candidates from FAISS.
    """

    query_embedding = embed_query(
        claim,
        model,
        retriever_id,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        record = metadata[idx]

        results.append({
            "page": record["page"],
            "sentence_id": record["sentence_id"],
            "text": record["text"],
            "score": float(score),
        })

    return results


# =========================================================
# OPTIONAL RERANKING
# =========================================================

def rerank_results(
    claim: str,
    candidates: List[Dict[str, Any]],
    reranker,
):
    """
    Rerank dense retrieval candidates.
    """

    if reranker is None:

        for candidate in candidates:

            candidate["rerank_score"] = None

        return candidates

    reranked = reranker.rerank(
        claim,
        candidates,
    )

    return reranked


# =========================================================
# MAIN SEARCH PIPELINE
# =========================================================

def search_claim(
    claim: str,
    model: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    retriever_id: str,
    reranker=None,
    top_k: int = 5,
    candidate_k: int = 50,
):
    """
    Full retrieval pipeline:
        dense retrieval
        → optional reranking
        → top-k evidence
    """

    # -----------------------------------------------------
    # DENSE RETRIEVAL
    # -----------------------------------------------------

    candidates = retrieve_candidates(
        claim=claim,
        model=model,
        index=index,
        metadata=metadata,
        retriever_id=retriever_id,
        top_k=candidate_k,
    )

    # -----------------------------------------------------
    # RERANKING
    # -----------------------------------------------------

    reranked = rerank_results(
        claim,
        candidates,
        reranker,
    )

    # -----------------------------------------------------
    # FINAL TOP-K
    # -----------------------------------------------------

    return reranked[:top_k]


# =========================================================
# LOCAL TESTING
# =========================================================

if __name__ == "__main__":

    from src.reranking.reranker import Reranker

    RETRIEVER_ID = "minilm"

    INDEX_PATH = (
        f"data/index/wiki_targeted_subset_{RETRIEVER_ID}.index"
    )

    METADATA_PATH = (
        f"data/index/wiki_targeted_subset_{RETRIEVER_ID}_metadata.json"
    )

    MODEL_NAMES = {

        "minilm": "sentence-transformers/all-MiniLM-L6-v2",

        "bge": "BAAI/bge-small-en-v1.5",

        "e5": "intfloat/e5-small-v2",
    }

    print("Loading model...")

    model = SentenceTransformer(
        MODEL_NAMES[RETRIEVER_ID]
    )

    print("Loading FAISS index...")

    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")

    metadata = load_metadata(METADATA_PATH)

    print("Loading reranker...")

    reranker = Reranker()

    claim = "Roman Atwood is a content creator."

    print("\nSearching...\n")

    results = search_claim(
        claim=claim,
        model=model,
        index=index,
        metadata=metadata,
        retriever_id=RETRIEVER_ID,
        reranker=reranker,
        top_k=5,
        candidate_k=50,
    )

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 60)

        print("Page:", result["page"])

        print("Sentence ID:", result["sentence_id"])

        print("Score:", result["score"])

        print(
            "Rerank Score:",
            result.get("rerank_score")
        )

        print("Text:", result["text"])
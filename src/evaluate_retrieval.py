"""
evaluate_retrieval.py

Evaluate retrieval quality using Recall@K.

This version is fully aligned with the project pipeline:
- Retrieval returns (page, sentence_id)
- FEVER gold evidence uses (page, sentence_id)
- Matching is done ONLY on these keys (no text matching)

Metric:
    Recall@K = % of claims where at least ONE gold evidence sentence
               appears in top-K retrieved results
"""

import json
from typing import List, Dict, Any, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from src.data_loader import load_jsonl
from src.preprocess import normalize_example
from src.query_faiss_targeted_subset import search_claim, load_metadata

# -----------------------------
# Helper: extract gold evidence keys
# -----------------------------

def get_gold_keys(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Extract all (page, sentence_id) pairs from FEVER evidence.

    Handles multiple evidence sets.
    """
    keys = []

    for evidence_set in example["evidence_sets"]:
        for item in evidence_set:
            keys.append((item["page"], item["sentence_id"]))

    return keys


# -----------------------------
# Core evaluation function
# -----------------------------

def compute_recall_at_k(
    data: List[Dict[str, Any]],
    model: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    k: int = 5
) -> float:
    """
    Compute Recall@K.

    Args:
        data: normalized FEVER examples
        model: embedding model
        index: FAISS index
        metadata: corpus metadata
        k: top-K retrieval

    Returns:
        recall@k
    """
    hits = 0
    total = 0

    for example in data:
        # Skip examples with no evidence (NOT ENOUGH INFO cases)
        if not example["evidence_sets"]:
            continue

        claim = example["claim"]
        gold_keys = set(get_gold_keys(example))

        results = search_claim(
            claim,
            model=model,
            index=index,
            metadata=metadata,
            top_k=k
        )

        # Extract retrieved keys
        retrieved_keys = set(
            (r["page"], r["sentence_id"]) for r in results
        )

        # Check if any gold evidence is retrieved
        if any(key in retrieved_keys for key in gold_keys):
            hits += 1

        total += 1

    return hits / total if total > 0 else 0.0


# -----------------------------
# Multi-K evaluation
# -----------------------------

def evaluate_all_k(data, model, index, metadata):
    ks = [1, 5, 10]
    results = {}

    for k in ks:
        print(f"\nEvaluating Recall@{k}...")
        recall = compute_recall_at_k(data, model, index, metadata, k=k)
        results[k] = recall
        print(f"Recall@{k}: {recall:.4f}")

    return results


# -----------------------------
# Main script
# -----------------------------

if __name__ == "__main__":

    # Paths (aligned with your subset pipeline)
    train_path = "data/raw/train.jsonl"
    index_path = "data/index/wiki_targeted_subset.index"
    metadata_path = "data/index/wiki_targeted_subset_metadata.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading FEVER data...")
    raw_data = load_jsonl(train_path)

    print("Normalizing FEVER examples...")
    data = [normalize_example(ex) for ex in raw_data[:200]]  # limit for speed

    print("Loading model...")
    model = SentenceTransformer(model_name)

    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)

    print("\nRunning evaluation...")
    results = evaluate_all_k(data, model, index, metadata)

    print("\nFinal Results:", results)
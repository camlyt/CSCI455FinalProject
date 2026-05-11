"""
evaluate_retrieval_with_reranker.py

Evaluates retrieval performance using dense retrieval followed by a CrossEncoder
reranker.

This script is separate from the stable dense-only retrieval pipeline so we can
compare whether reranking improves Recall@K or makes performance worse.

Pipeline:
    1. Load FEVER examples
    2. Load targeted FAISS index and metadata
    3. Retrieve top candidate_k dense candidates
    4. Rerank candidates using CrossEncoder
    5. Compute Recall@1, Recall@5, and Recall@10
"""

from typing import Dict, Any, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from src.data_loader import load_jsonl
from src.preprocess import normalize_example
from src.query_faiss_targeted_subset import search_claim, load_metadata
from src.reranker import Reranker


def get_gold_keys(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Extract gold evidence keys from a normalized FEVER example.

    Args:
        example: Normalized FEVER example.

    Returns:
        A list of (page, sentence_id) tuples.
    """
    keys = []

    for evidence_set in example["evidence_sets"]:
        for item in evidence_set:
            keys.append((item["page"], item["sentence_id"]))

    return keys


def evaluate_recall_with_reranker(
    data,
    model,
    index,
    metadata,
    reranker,
    candidate_k: int = 50
):
    """
    Evaluate Recall@1, Recall@5, and Recall@10 using dense retrieval + reranking.

    Args:
        data: Normalized FEVER examples.
        model: SentenceTransformer model for dense retrieval.
        index: FAISS index.
        metadata: Metadata records for FAISS rows.
        reranker: CrossEncoder reranker.
        candidate_k: Number of dense candidates retrieved before reranking.

    Returns:
        Dictionary containing Recall@1, Recall@5, and Recall@10.
    """
    ks = [1, 5, 10]
    hits = {k: 0 for k in ks}
    total = 0

    for i, example in enumerate(data, start=1):
        if not example["evidence_sets"]:
            continue

        print(f"\nEvaluating example {i}/{len(data)}")

        claim = example["claim"]
        gold_keys = set(get_gold_keys(example))

        # Retrieve candidate_k dense candidates, then rerank and keep top 10.
        # We retrieve top 10 final results so Recall@1, @5, and @10 can all be measured.
        results = search_claim(
            claim=claim,
            model=model,
            index=index,
            metadata=metadata,
            reranker=reranker,
            top_k=10,
            candidate_k=candidate_k
        )

        retrieved_keys = [
            (result["page"], result["sentence_id"])
            for result in results
        ]

        for k in ks:
            top_k_keys = set(retrieved_keys[:k])

            if any(gold_key in top_k_keys for gold_key in gold_keys):
                hits[k] += 1

        total += 1

        print("Claim:", claim)
        print("Gold keys:", gold_keys)
        print("Top retrieved keys:", retrieved_keys[:5])

    recall_results = {}

    print("\nRetrieval Results with Reranker")
    for k in ks:
        recall = hits[k] / total if total > 0 else 0.0
        recall_results[k] = recall
        print(f"Recall@{k}: {recall:.4f}")

    print(f"\nEvaluated examples with usable evidence: {total}")

    return recall_results


if __name__ == "__main__":
    NUM_EXAMPLES = 100
    CANDIDATE_K = 50

    train_path = "data/raw/train.jsonl"
    index_path = "data/index/wiki_targeted_subset.index"
    metadata_path = "data/index/wiki_targeted_subset_metadata.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading FEVER data...")
    raw_data = load_jsonl(train_path)

    print("Normalizing FEVER examples...")
    data = [normalize_example(example) for example in raw_data[:NUM_EXAMPLES]]

    print("Loading dense retrieval model...")
    model = SentenceTransformer(model_name)

    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)

    print("Loading reranker...")
    reranker = Reranker()

    print("\nRunning retrieval evaluation with reranker...")
    results = evaluate_recall_with_reranker(
        data=data,
        model=model,
        index=index,
        metadata=metadata,
        reranker=reranker,
        candidate_k=CANDIDATE_K
    )

    print("\nFinal Reranker Retrieval Results:", results)

    # Temporary workaround for local macOS/FAISS shutdown segfault.
    import os
    os._exit(0)
"""
evaluate_reranked_retrieval_save_outputs.py

Runs dense retrieval + reranking on FEVER examples and saves the reranked
top-k evidence outputs to disk.

This lets us test whether reranked evidence improves final verifier accuracy.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from src.data.data_loader import load_jsonl
from src.data.preprocess import normalize_example
from src.retrieval.query_faiss_targeted_subset import search_claim, load_metadata
from src.reranking.reranker import Reranker


def get_gold_keys(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    keys = []
    for evidence_set in example["evidence_sets"]:
        for item in evidence_set:
            keys.append((item["page"], item["sentence_id"]))
    return keys


def save_jsonl(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    NUM_EXAMPLES = 100
    CANDIDATE_K = 50

    train_path = "data/raw/train.jsonl"
    index_path = "data/index/wiki_targeted_subset.index"
    metadata_path = "data/index/wiki_targeted_subset_metadata.json"
    output_path = "data/processed/reranked_retrieval_outputs.jsonl"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading FEVER data...")
    raw_data = load_jsonl(train_path)

    print("Normalizing FEVER examples...")
    data = [normalize_example(ex) for ex in raw_data[:NUM_EXAMPLES]]

    print("Loading dense retrieval model...")
    model = SentenceTransformer(model_name)

    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)

    print("Loading reranker...")
    reranker = Reranker()

    output_records = []

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0

    for i, example in enumerate(data, start=1):
        if not example["evidence_sets"]:
            continue

        print(f"\nRetrieving + reranking example {i}/{len(data)}")

        claim = example["claim"]
        gold_label = example["label"]
        gold_keys = set(get_gold_keys(example))

        results = search_claim(
            claim=claim,
            model=model,
            index=index,
            metadata=metadata,
            reranker=reranker,
            top_k=10,
            candidate_k=CANDIDATE_K
        )

        retrieved_keys = [
            (r["page"], r["sentence_id"]) for r in results
        ]

        if any(key in set(retrieved_keys[:1]) for key in gold_keys):
            hits_at_1 += 1
        if any(key in set(retrieved_keys[:5]) for key in gold_keys):
            hits_at_5 += 1
        if any(key in set(retrieved_keys[:10]) for key in gold_keys):
            hits_at_10 += 1

        total += 1

        output_records.append({
            "claim": claim,
            "gold_label": gold_label,
            "gold_keys": list(gold_keys),
            "retrieved_evidence": results
        })

    print("\nReranked Retrieval Results")
    print(f"Recall@1: {hits_at_1 / total if total else 0:.4f}")
    print(f"Recall@5: {hits_at_5 / total if total else 0:.4f}")
    print(f"Recall@10: {hits_at_10 / total if total else 0:.4f}")

    save_jsonl(output_records, output_path)
    print(f"\nSaved reranked retrieval outputs to {output_path}")

    # Temporary workaround for local macOS/FAISS shutdown segfault.
    import os
    os._exit(0)
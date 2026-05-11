"""
rerank_saved_candidates.py

Loads saved dense retrieval candidates and reranks them with a CrossEncoder.
This avoids loading FAISS and the reranker in the same Python process.
"""

import json
from pathlib import Path

from src.reranking.reranker import Reranker


def load_jsonl(file_path):
    records = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = "data/processed/dense_candidate_outputs.jsonl"
    output_path = "data/processed/reranked_retrieval_outputs.jsonl"
    TOP_K = 10

    print("Loading dense candidates...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} examples")

    print("Loading reranker...")
    reranker = Reranker()

    output_records = []

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0

    for i, record in enumerate(records, start=1):
        print(f"\nReranking example {i}/{len(records)}")

        claim = record["claim"]
        gold_keys = set(tuple(key) for key in record["gold_keys"])
        candidates = record["dense_candidates"]

        reranked = reranker.rerank(
            claim=claim,
            candidates=candidates,
            top_k=TOP_K
        )

        retrieved_keys = [(r["page"], r["sentence_id"]) for r in reranked]

        if any(key in set(retrieved_keys[:1]) for key in gold_keys):
            hits_at_1 += 1
        if any(key in set(retrieved_keys[:5]) for key in gold_keys):
            hits_at_5 += 1
        if any(key in set(retrieved_keys[:10]) for key in gold_keys):
            hits_at_10 += 1

        total += 1

        output_records.append({
            "claim": record["claim"],
            "gold_label": record["gold_label"],
            "gold_keys": record["gold_keys"],
            "retrieved_evidence": reranked
        })

    print("\nReranked Retrieval Results")
    print(f"Recall@1: {hits_at_1 / total if total else 0:.4f}")
    print(f"Recall@5: {hits_at_5 / total if total else 0:.4f}")
    print(f"Recall@10: {hits_at_10 / total if total else 0:.4f}")

    save_jsonl(output_records, output_path)
    print(f"\nSaved reranked retrieval outputs to {output_path}")
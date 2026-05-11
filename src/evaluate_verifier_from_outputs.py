"""
evaluate_verifier_from_outputs.py

Loads saved retrieval outputs and evaluates the verifier separately.

This avoids loading FAISS, SentenceTransformer, reranker, and verifier all in
one process.
"""

import json
from pathlib import Path

from src.verifier import Verifier


def load_jsonl(file_path):
    records = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


if __name__ == "__main__":
    input_path = "data/processed/reranked_retrieval_outputs.jsonl"

    print("Loading retrieval outputs...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} examples")

    print("Loading verifier...")
    verifier = Verifier()

    correct = 0
    total = 0

    for i, record in enumerate(records, start=1):
        claim = record["claim"]
        gold_label = record["gold_label"]
        evidence = record["retrieved_evidence"][:5]

        print(f"\nVerifier example {i}/{len(records)}")
        print("Claim:", claim)
        print("Gold label:", gold_label)

        pred_label = verifier.predict(claim, evidence)

        print("Predicted label:", pred_label)

        if pred_label == gold_label:
            correct += 1

        total += 1
        print(f"Running accuracy: {correct}/{total} = {correct / total:.4f}")

    accuracy = correct / total if total > 0 else 0.0

    print("\nFinal Pipeline Accuracy:", accuracy)
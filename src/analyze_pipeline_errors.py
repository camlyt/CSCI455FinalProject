"""
analyze_pipeline_errors.py

Runs error analysis on saved retrieval outputs.

This script loads reranked retrieval outputs, runs the verifier, and saves
examples where the predicted label does not match the FEVER gold label.
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


def save_jsonl(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def contains_gold_evidence(record, top_k=5):
    gold_keys = set(tuple(key) for key in record["gold_keys"])

    retrieved_keys = set(
        (ev["page"], ev["sentence_id"])
        for ev in record["retrieved_evidence"][:top_k]
    )

    return any(key in retrieved_keys for key in gold_keys)


def categorize_error(record, pred_label, top_k=5):
    """
    Assign a rough error category based on retrieval and prediction behavior.
    This is not perfect, but it gives useful buckets for analysis.
    """
    gold_label = record["gold_label"]
    has_gold = contains_gold_evidence(record, top_k=top_k)

    if not has_gold:
        return "retrieval_miss"

    if pred_label == "NOT ENOUGH INFO" and gold_label in {"SUPPORTS", "REFUTES"}:
        return "verifier_too_conservative"

    if pred_label in {"SUPPORTS", "REFUTES"} and gold_label == "NOT ENOUGH INFO":
        return "verifier_overconfident"

    if pred_label != gold_label and has_gold:
        return "verifier_wrong_despite_gold_evidence"

    return "other"


if __name__ == "__main__":
    input_path = "data/processed/reranked_retrieval_outputs.jsonl"
    output_path = "data/processed/pipeline_errors.jsonl"

    print("Loading retrieval outputs...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} examples")

    print("Loading verifier...")
    verifier = Verifier()

    errors = []
    correct = 0
    total = 0

    for i, record in enumerate(records, start=1):
        claim = record["claim"]
        gold_label = record["gold_label"]
        evidence = record["retrieved_evidence"][:5]

        result = verifier.predict_with_scores(claim, evidence)
        pred_label = result["prediction"]

        if pred_label == gold_label:
            correct += 1
        else:
            error_category = categorize_error(record, pred_label, top_k=5)

            errors.append({
                "example_number": i,
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "error_category": error_category,
                "scores": {
                    "entailment": result["entailment"],
                    "contradiction": result["contradiction"],
                    "neutral": result["neutral"]
                },
                "gold_keys": record["gold_keys"],
                "retrieved_top_5": evidence,
                "gold_found_in_top_5": contains_gold_evidence(record, top_k=5)
            })

        total += 1

    accuracy = correct / total if total else 0.0

    print("\nAccuracy:", accuracy)
    print("Total errors:", len(errors))

    category_counts = {}
    for error in errors:
        category = error["error_category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\nError category counts:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    save_jsonl(errors, output_path)
    print(f"\nSaved errors to {output_path}")
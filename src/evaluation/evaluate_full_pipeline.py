from typing import List, Dict, Any, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from src.data.data_loader import load_jsonl
from src.data.preprocess import normalize_example
from src.retrieval.query_faiss_targeted_subset import search_claim, load_metadata
from src.reranking.reranker import Reranker
from src.verification.verifier import Verifier


# -----------------------------
# Helper: extract gold evidence keys
# -----------------------------

def get_gold_keys(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    keys = []
    for evidence_set in example["evidence_sets"]:
        for item in evidence_set:
            keys.append((item["page"], item["sentence_id"]))
    return keys


# -----------------------------
# Retrieval Evaluation (same as before)
# -----------------------------

def compute_recall_at_k(
    data,
    model,
    index,
    metadata,
    reranker,
    k=5
):
    hits = 0
    total = 0

    for example in data:
        if not example["evidence_sets"]:
            continue

        claim = example["claim"]
        gold_keys = set(get_gold_keys(example))

        results = search_claim(
            claim,
            model=model,
            index=index,
            metadata=metadata,
            retriever_id="minilm",
            reranker=reranker,
            top_k=k
        )

        retrieved_keys = set(
            (r["page"], r["sentence_id"]) for r in results
        )

        if any(key in retrieved_keys for key in gold_keys):
            hits += 1

        total += 1

    return hits / total if total > 0 else 0.0


def evaluate_all_k(data, model, index, metadata, reranker):
    ks = [1, 5, 10]
    results = {}

    for k in ks:
        print(f"\nEvaluating Recall@{k}...")
        recall = compute_recall_at_k(
            data,
            model,
            index,
            metadata,
            reranker,
            k=k
        )
        results[k] = recall
        print(f"Recall@{k}: {recall:.4f}")

    return results


# -----------------------------
# NEW: Full pipeline accuracy
# -----------------------------

def evaluate_accuracy(data, model, index, metadata, verifier):
    correct = 0
    total = 0

    for i, example in enumerate(data, start=1):
        claim = example["claim"]
        gold_label = example["label"]

        print(f"\nAccuracy example {i}/{len(data)}")
        print("Claim:", claim)
        print("Gold label:", gold_label)

        print("Retrieving evidence without reranker...")
        results = search_claim(
            claim,
            model=model,
            index=index,
            metadata=metadata,
            reranker=None,
            top_k=5,
            candidate_k=10
        )

        print("Retrieved evidence count:", len(results))

        print("Running verifier...")
        pred_label = verifier.predict(claim, results)

        print("Predicted label:", pred_label)

        if pred_label == gold_label:
            correct += 1

        total += 1
        print(f"Running accuracy: {correct}/{total} = {correct / total:.4f}")

    return correct / total if total > 0 else 0.0
# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    reranker = None

    train_path = "data/raw/train.jsonl"
    index_path = "data/index/wiki_targeted_subset.index"
    metadata_path = "data/index/wiki_targeted_subset_metadata.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading FEVER data...")
    raw_data = load_jsonl(train_path)

    print("Normalizing FEVER examples...")
    data = [normalize_example(ex) for ex in raw_data[:10]]

    print("Loading model...")
    model = SentenceTransformer(model_name)

    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)

    # -----------------------------
    # Retrieval metrics
    # -----------------------------
    print("\nRunning retrieval evaluation...")
    retrieval_results = evaluate_all_k(data, model, index, metadata, reranker)

    print("\nFinal Retrieval Results:", retrieval_results)

    # -----------------------------
    # Full pipeline accuracy
    # -----------------------------
    print("\nLoading verifier only for full pipeline evaluation...")
    verifier = Verifier()

    print("\nRunning full pipeline evaluation...")
    accuracy = evaluate_accuracy(
        data,
        model,
        index,
        metadata,
        verifier
    )

    print("\nFinal Pipeline Accuracy:", accuracy)

    # Temporary workaround for local macOS/FAISS shutdown segfault.
    import os

    os._exit(0)
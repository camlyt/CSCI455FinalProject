from typing import List, Dict, Any, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from src.data_loader import load_jsonl
from src.preprocess import normalize_example
from src.query_faiss_targeted_subset import search_claim, load_metadata
from src.reranker import Reranker
from src.verifier import Verifier


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

def evaluate_accuracy(data, model, index, metadata, reranker, verifier):
    correct = 0
    total = 0

    for example in data:   # ← example is defined HERE

        claim = example["claim"]
        gold_label = example["label"]

        # 👇 anything using example MUST be inside this loop
        gold_keys = set(get_gold_keys(example))

        results = search_claim(
            claim,
            model=model,
            index=index,
            metadata=metadata,
            reranker=reranker,
            top_k=5
        )

        pred_label = verifier.predict(claim, results)

        if pred_label == gold_label:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0
# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    reranker = Reranker()
    verifier = Verifier()

    train_path = "data/raw/train.jsonl"
    index_path = "data/index/wiki_targeted_subset.index"
    metadata_path = "data/index/wiki_targeted_subset_metadata.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading FEVER data...")
    raw_data = load_jsonl(train_path)

    print("Normalizing FEVER examples...")
    data = [normalize_example(ex) for ex in raw_data[:200]]

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
    print("\nRunning full pipeline evaluation...")
    accuracy = evaluate_accuracy(
        data,
        model,
        index,
        metadata,
        reranker,
        verifier
    )

    print("\nFinal Pipeline Accuracy:", accuracy)
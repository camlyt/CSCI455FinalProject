from typing import Any, Dict, List
from src.data_loader import load_jsonl

"""
This module handles preprocessing and normalization of the FEVER dataset.

The raw FEVER evidence field has a nested structure:
    - multiple possible evidence sets per claim
    - each evidence set contains one or more sentence references

This file simplifies that structure by:
    - extracting only relevant fields (page title and sentence index)
    - organizing evidence into clean, consistent formats

Main responsibilities:
    - Normalize raw FEVER examples
    - Convert nested evidence into structured dictionaries

This prepares the data for later steps such as:
    - retrieval evaluation
    - matching against Wikipedia sentence corpus
"""

def normalize_evidence(evidence: List[List[List[Any]]]) -> List[List[Dict[str, Any]]]:
    """
    Convert FEVER's nested evidence format into a cleaner structure.

    Raw structure:
        [[[annotation_id, evidence_id, page, sentence_id], ...], ...]

    Cleaned structure:
        [
            [
                {"page": str, "sentence_id": int},
                ...
            ],
            ...
        ]

    Args:
        evidence: Raw FEVER evidence field.

    Returns:
        A list of evidence sets. Each evidence set is a list of dictionaries
        containing page title and sentence index.
    """
    normalized_sets = []

    for evidence_set in evidence:
        cleaned_set = []

        for item in evidence_set:
            if len(item) >= 4:
                cleaned_set.append({
                    "page": item[2],
                    "sentence_id": item[3]
                })

        if cleaned_set:
            normalized_sets.append(cleaned_set)

    return normalized_sets


def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single FEVER example into a cleaner format.
    Keeps only the important fields:
        - claim
        - label
        - cleaned evidence sets
    Args:
        example: One raw FEVER example.

    Returns:
        A simplified dictionary with claim, label, and normalized evidence sets.
    """
    return {
        "id": example.get("id"),
        "claim": example.get("claim"),
        "label": example.get("label"),
        "evidence_sets": normalize_evidence(example.get("evidence", []))
    }


if __name__ == "__main__":
    train_data = load_jsonl("data/raw/train.jsonl")

    for i, example in enumerate(train_data[:3], start=1):
        normalized = normalize_example(example)
        print(f"\nNormalized Example {i}")
        print("-" * 50)
        print("Claim:", normalized["claim"])
        print("Label:", normalized["label"])
        print("Evidence Sets:", normalized["evidence_sets"])
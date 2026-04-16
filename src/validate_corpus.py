"""
This file checks whether FEVER gold evidence references can be found in the
processed Wikipedia sentence corpus.

The goal is to confirm that normalized FEVER evidence entries match the
(page, sentence_id) pairs extracted from the wiki dump.
"""

import json
from pathlib import Path
from typing import Dict, Any, Set, Tuple

from src.data_loader import load_jsonl
from src.preprocess import normalize_example


def load_corpus_keys(corpus_path: str, limit: int | None = None) -> Set[Tuple[str, int]]:
    """
    Load (page, sentence_id) pairs from the processed wiki sentence corpus.

    Args:
        corpus_path: Path to the processed wiki sentence JSONL file.
        limit: Optional line limit for quick testing.

    Returns:
        A set of (page, sentence_id) tuples.
    """
    keys = set()
    path = Path(corpus_path)

    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if limit is not None and i >= limit:
                break

            record = json.loads(line)
            keys.add((record["page"], record["sentence_id"]))

    return keys


def validate_examples(train_path: str, corpus_keys: Set[Tuple[str, int]], num_examples: int = 100) -> None:
    """
    Check whether FEVER gold evidence references exist in the corpus.

    Args:
        train_path: Path to FEVER training JSONL.
        corpus_keys: Set of valid (page, sentence_id) keys from the corpus.
        num_examples: Number of FEVER examples to validate.
    """
    train_data = load_jsonl(train_path)

    total_refs = 0
    matched_refs = 0

    for raw_example in train_data[:num_examples]:
        example = normalize_example(raw_example)

        for evidence_set in example["evidence_sets"]:
            for item in evidence_set:
                total_refs += 1
                key = (item["page"], item["sentence_id"])

                if key in corpus_keys:
                    matched_refs += 1
                else:
                    print("Missing evidence:", key)
                    print("Claim:", example["claim"])
                    print()

    print(f"Checked {num_examples} examples")
    print(f"Matched evidence refs: {matched_refs}/{total_refs}")


if __name__ == "__main__":
    corpus_keys = load_corpus_keys("data/processed/wiki_sentences.jsonl")
    validate_examples(
        train_path="data/raw/train.jsonl",
        corpus_keys=corpus_keys,
        num_examples=100
    )
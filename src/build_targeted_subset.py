"""
This file builds a targeted Wikipedia sentence subset using page names
found in FEVER gold evidence.

Instead of taking an arbitrary slice of the Wikipedia corpus, this script
collects evidence page titles from a small number of FEVER examples and
keeps only sentence entries from those pages.

This gives us a more meaningful retrieval test corpus because the correct
evidence pages are guaranteed to be included.
"""

import json
from pathlib import Path
from typing import Set

from src.data_loader import load_jsonl
from src.preprocess import normalize_example


def collect_target_pages(train_path: str, num_examples: int = 100) -> Set[str]:
    """
    Collect unique Wikipedia page names from FEVER gold evidence.

    Args:
        train_path: Path to FEVER training JSONL file.
        num_examples: Number of FEVER examples to inspect.

    Returns:
        A set of page titles referenced by FEVER gold evidence.
    """
    train_data = load_jsonl(train_path)
    pages = set()

    for raw_example in train_data[:num_examples]:
        example = normalize_example(raw_example)

        for evidence_set in example["evidence_sets"]:
            for item in evidence_set:
                page = item["page"]
                if page:
                    pages.add(page)

    return pages


def build_targeted_subset(
    corpus_path: str,
    output_path: str,
    target_pages: Set[str]
) -> None:
    """
    Filter the full wiki sentence corpus down to rows whose page is in target_pages.

    Args:
        corpus_path: Path to the full wiki sentence corpus JSONL file.
        output_path: Path to save the filtered subset.
        target_pages: Set of page names to keep.
    """
    corpus_file = Path(corpus_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    kept_rows = 0

    with corpus_file.open("r", encoding="utf-8") as infile, \
         output_file.open("w", encoding="utf-8") as outfile:

        for line in infile:
            record = json.loads(line)
            if record["page"] in target_pages:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_rows += 1

    print(f"Target pages collected: {len(target_pages)}")
    print(f"Saved {kept_rows} sentence rows to {output_path}")


if __name__ == "__main__":
    train_path = "data/raw/train.jsonl"
    corpus_path = "data/processed/wiki_sentences.jsonl"
    output_path = "data/processed/wiki_targeted_subset.jsonl"

    target_pages = collect_target_pages(train_path, num_examples=100)

    print("Sample target pages:")
    for page in list(target_pages)[:10]:
        print("-", page)

    build_targeted_subset(
        corpus_path=corpus_path,
        output_path=output_path,
        target_pages=target_pages
    )
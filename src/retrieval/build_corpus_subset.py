"""
This file creates a smaller subset of the processed Wikipedia sentence corpus
for early embedding and retrieval experiments.

Using a subset first makes it easier to test the pipeline before scaling to
the full 25M-sentence corpus.
"""

import json
from pathlib import Path


def build_subset(input_path: str, output_path: str, max_rows: int = 50000) -> None:
    """
    Copy the first max_rows entries from the full corpus into a smaller file.

    Args:
        input_path: Path to the full wiki sentence corpus JSONL file.
        output_path: Path to save the smaller subset JSONL file.
        max_rows: Number of rows to keep.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_file.open("r", encoding="utf-8") as infile, \
         output_file.open("w", encoding="utf-8") as outfile:

        for line in infile:
            if count >= max_rows:
                break

            record = json.loads(line)
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} rows to {output_path}")


if __name__ == "__main__":
    build_subset(
        input_path="data/processed/wiki_sentences.jsonl",
        output_path="data/processed/wiki_sentences_subset.jsonl",
        max_rows=50000
    )
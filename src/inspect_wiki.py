import json
from pathlib import Path

"""
This file is used to inspect the structure of the FEVER pre-processed
Wikipedia page files before building the retrieval corpus.

From running this file, we are able to determine the following:
Each wiki record has:
- id → page title
- text → full page text
- lines → sentence-level content in FEVER format

"""


def inspect_wiki_file(file_path: str, num_records: int = 3) -> None:
    """
    Print a few records from one FEVER wiki JSONL file.

    Args:
        file_path: Path to a wiki JSONL file.
        num_records: Number of records to print.
    """
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= num_records:
                break

            record = json.loads(line.strip())

            print(f"\nRecord {i + 1}")
            print("-" * 60)
            print(record)
            print("\nKeys:", list(record.keys()))


if __name__ == "__main__":
    inspect_wiki_file("data/raw/wiki-pages/wiki-001.jsonl")
"""
This module handles loading and basic inspection of the FEVER dataset.

The FEVER dataset is stored in JSONL format (one JSON object per line).
Each object represents a single claim with:
    - claim text
    - label (SUPPORTED, REFUTED, NOT ENOUGH INFO)
    - annotated evidence (nested structure)

Main responsibilities:
    - Load JSONL files into Python objects
    - Provide simple preview utilities to inspect dataset structure

This file does NOT modify or clean the data. It is only responsible for
reading raw data from disk.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
        Load a JSONL (JSON Lines) file into memory.

        Each line in the file is parsed as a separate JSON object.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            A list of dictionaries, where each dictionary represents one example.
    """
    data = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def preview_examples(data: List[Dict[str, Any]], num_examples: int = 3) -> None:
    """
        Print a small number of examples for quick inspection.

        Useful for understanding the structure of claims, labels, and evidence.

        Args:
            data: List of dataset examples.
            num_examples: Number of examples to print.
    """
    for i, example in enumerate(data[:num_examples], 1):
        print(f"\nExample {i}")
        print("-" * 40)
        print("Claim:", example.get("claim"))
        print("Label:", example.get("label"))
        print("Evidence:", example.get("evidence"))


if __name__ == "__main__":
    train_path = "data/raw/train.jsonl"
    train_data = load_jsonl(train_path)

    print(f"Loaded {len(train_data)} examples")
    preview_examples(train_data)
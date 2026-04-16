"""
This module converts the FEVER pre-processed Wikipedia dump into a
sentence-level retrieval corpus.

Each Wikipedia page contains a `lines` field where sentences are stored
with their sentence numbers. This file extracts those sentences and saves
them in a clean format with:
    - page title
    - sentence id
    - sentence text

The output of this script will be used later for embedding generation
and FAISS-based retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def parse_wiki_lines(lines_text: str) -> List[Dict[str, Any]]:
    """
    Parse the FEVER wiki `lines` field into sentence-level entries.

    Args:
        lines_text: Raw `lines` string from one wiki page record.

    Returns:
        A list of dictionaries with sentence_id and text.
    """
    sentences = []

    if not lines_text.strip():
        return sentences

    for raw_line in lines_text.split("\n"):
        if not raw_line.strip():
            continue

        parts = raw_line.split("\t")

        if len(parts) < 2:
            continue

        sentence_id_str = parts[0].strip()
        sentence_text = parts[1].strip()

        if not sentence_id_str.isdigit():
            continue

        if not sentence_text:
            continue

        sentences.append({
            "sentence_id": int(sentence_id_str),
            "text": sentence_text
        })

    return sentences


def extract_sentences_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract sentence-level entries from one wiki page record.

    Args:
        record: One JSON record from the FEVER wiki dump.

    Returns:
        A list of dictionaries containing page, sentence_id, and text.
    """
    page = record.get("id", "").strip()
    lines_text = record.get("lines", "")

    if not page:
        return []

    parsed_sentences = parse_wiki_lines(lines_text)

    extracted = []
    for sentence in parsed_sentences:
        extracted.append({
            "page": page,
            "sentence_id": sentence["sentence_id"],
            "text": sentence["text"]
        })

    return extracted


def process_wiki_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process one FEVER wiki JSONL file into sentence-level entries.

    Args:
        file_path: Path to one wiki JSONL file.

    Returns:
        A list of extracted sentence dictionaries.
    """
    extracted = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            extracted.extend(extract_sentences_from_record(record))

    return extracted


def process_wiki_folder(folder_path: str, output_path: str) -> None:
    """
    Process all FEVER wiki JSONL files in a folder and save one combined JSONL file.

    Args:
        folder_path: Path to the wiki-pages folder.
        output_path: Path to the combined output JSONL file.
    """
    folder = Path(folder_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    wiki_files = sorted(folder.glob("wiki-*.jsonl"))
    total_sentences = 0

    with output_file.open("w", encoding="utf-8") as out_file:
        for wiki_file in wiki_files:
            print(f"Processing {wiki_file.name}...")

            sentences = process_wiki_file(str(wiki_file))
            total_sentences += len(sentences)

            for sentence in sentences:
                out_file.write(json.dumps(sentence, ensure_ascii=False) + "\n")

            print(f"  Extracted {len(sentences)} sentence entries")

    print(f"\nFinished processing {len(wiki_files)} files")
    print(f"Total extracted sentence entries: {total_sentences}")
    print(f"Saved combined corpus to {output_path}")


if __name__ == "__main__":
    process_wiki_folder(
        folder_path="data/raw/wiki-pages",
        output_path="data/processed/wiki_sentences.jsonl"
    )
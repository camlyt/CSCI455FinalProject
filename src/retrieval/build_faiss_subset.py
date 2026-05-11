"""
This file builds a FAISS index for a smaller subset of the Wikipedia sentence
corpus. It is meant for early testing before scaling to the full dataset.

Steps:
    1. Load the subset corpus
    2. Extract sentence texts
    3. Generate embeddings
    4. Build a FAISS index
    5. Save the index and metadata for retrieval
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


def load_subset_corpus(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL subset corpus into memory.

    Args:
        file_path: Path to the subset JSONL file.

    Returns:
        A list of sentence records.
    """
    records = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def extract_texts(records: List[Dict[str, Any]]) -> List[str]:
    """
    Extract sentence text from each corpus record.

    Args:
        records: List of corpus records.

    Returns:
        A list of sentence strings.
    """
    return [record["text"] for record in records]


def save_metadata(records: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save corpus metadata to a JSON file so retrieved FAISS indices can be mapped
    back to page titles, sentence ids, and text.

    Args:
        records: List of corpus records.
        output_path: Path to save the metadata file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index using inner product similarity.

    Embeddings should be normalized before indexing so inner product behaves
    like cosine similarity.

    Args:
        embeddings: 2D numpy array of normalized embeddings.

    Returns:
        A FAISS index containing the embeddings.
    """
    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


if __name__ == "__main__":
    input_file = "data/processed/wiki_sentences_subset.jsonl"
    index_file = "data/index/wiki_subset.index"
    metadata_file = "data/index/wiki_subset_metadata.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading subset corpus...")
    records = load_subset_corpus(input_file)
    print(f"Loaded {len(records)} records")

    print("Extracting texts...")
    texts = extract_texts(records)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Embeddings generated")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)

    import faiss
    print("faiss imported")

    print("Converting embeddings to float32...")
    embeddings = embeddings.astype("float32")
    print("Converted to float32")

    print("Normalizing embeddings with NumPy...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    embeddings = embeddings / norms
    embeddings = embeddings.astype("float32")
    print("Embeddings normalized")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print("FAISS index built")

    Path(index_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, index_file)
    print("FAISS index saved")

    print(f"Saving metadata to {metadata_file}")
    save_metadata(records, metadata_file)
    print("Metadata saved")

    print("Done.")
    print(f"Index size: {index.ntotal}")
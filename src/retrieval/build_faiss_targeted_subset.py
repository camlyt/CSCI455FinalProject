"""
build_faiss_targeted_subset.py

Builds FAISS indexes for multiple retrieval models using the targeted
Wikipedia subset corpus.

For each retriever:
    1. Load the targeted subset corpus
    2. Generate embeddings
    3. Normalize embeddings
    4. Build a FAISS index
    5. Save index + metadata

This allows the frontend/backend to dynamically switch retrievers
while using matching embedding spaces + FAISS indexes.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# =========================================================
# RETRIEVER MODELS
# =========================================================

RETRIEVER_MODELS = {

    "minilm": "sentence-transformers/all-MiniLM-L6-v2",

    "bge": "BAAI/bge-small-en-v1.5",

    "e5": "intfloat/e5-small-v2",
}


# =========================================================
# DATA LOADING
# =========================================================

def load_subset_corpus(
    file_path: str
) -> List[Dict[str, Any]]:
    """
    Load targeted subset JSONL corpus.
    """

    records = []

    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line:

                records.append(json.loads(line))

    return records


# =========================================================
# TEXT PREPARATION
# =========================================================

def extract_texts(
    records: List[Dict[str, Any]],
    retriever_id: str,
) -> List[str]:
    """
    Prepare text formatting for different retrievers.

    E5 requires special prefixes:
        passage: ...
        query: ...

    MiniLM/BGE work fine with raw text.
    """

    if retriever_id == "e5":

        return [
            "passage: " + record["text"]
            for record in records
        ]

    return [
        record["text"]
        for record in records
    ]


# =========================================================
# METADATA SAVING
# =========================================================

def save_metadata(
    records: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save metadata mapping FAISS ids back to sentences.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w", encoding="utf-8") as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )


# =========================================================
# FAISS INDEX BUILDING
# =========================================================

def build_faiss_index(
    embeddings: np.ndarray
):
    """
    Build cosine-similarity FAISS index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# =========================================================
# SINGLE RETRIEVER PIPELINE
# =========================================================

def build_index(
    retriever_id: str,
    model_name: str,
):
    """
    Build one retriever-specific FAISS index.
    """

    print("\n" + "=" * 70)
    print(f"BUILDING INDEX FOR: {retriever_id}")
    print("=" * 70)

    # -----------------------------------------------------
    # PATHS
    # -----------------------------------------------------

    input_file = (
        "data/processed/wiki_targeted_subset.jsonl"
    )

    index_file = (
        f"data/index/wiki_targeted_subset_{retriever_id}.index"
    )

    metadata_file = (
        f"data/index/wiki_targeted_subset_{retriever_id}_metadata.json"
    )

    # -----------------------------------------------------
    # LOAD CORPUS
    # -----------------------------------------------------

    print("Loading subset corpus...")

    records = load_subset_corpus(input_file)

    print(f"Loaded {len(records)} records")

    # -----------------------------------------------------
    # PREPARE TEXTS
    # -----------------------------------------------------

    print("Preparing texts...")

    texts = extract_texts(
        records,
        retriever_id
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print(f"Loading model: {model_name}")

    model = SentenceTransformer(model_name)

    # -----------------------------------------------------
    # GENERATE EMBEDDINGS
    # -----------------------------------------------------

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Embeddings generated")

    print("Embedding shape:", embeddings.shape)

    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    print("Converting embeddings to float32...")

    embeddings = embeddings.astype("float32")

    print("Normalizing embeddings...")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.clip(
        norms,
        a_min=1e-12,
        a_max=None
    )

    embeddings = embeddings / norms

    embeddings = embeddings.astype("float32")

    print("Embeddings normalized")

    # -----------------------------------------------------
    # BUILD FAISS
    # -----------------------------------------------------

    print("Building FAISS index...")

    index = build_faiss_index(embeddings)

    print("FAISS index built")

    # -----------------------------------------------------
    # SAVE INDEX
    # -----------------------------------------------------

    Path(index_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Saving index to: {index_file}")

    faiss.write_index(index, index_file)

    print("Index saved")

    # -----------------------------------------------------
    # SAVE METADATA
    # -----------------------------------------------------

    print(f"Saving metadata to: {metadata_file}")

    save_metadata(
        records,
        metadata_file
    )

    print("Metadata saved")

    # -----------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------

    print(f"{retriever_id} complete ✅")

    print(f"Index size: {index.ntotal}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\nBuilding retriever indexes...\n")

    for retriever_id, model_name in RETRIEVER_MODELS.items():

        build_index(
            retriever_id=retriever_id,
            model_name=model_name,
        )

    print("\nAll indexes built successfully 🎉")
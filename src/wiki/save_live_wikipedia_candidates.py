"""
save_live_wikipedia_candidates.py

Runs LIVE Wikipedia retrieval and saves retrieved evidence
to disk for later evaluation.

Features:
    - persistent caching via saved JSONL
    - resumable processing
    - rate-limit protection
    - incremental saving

Outputs:
    data/processed/live_wikipedia_candidates.jsonl
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.data.data_loader import load_jsonl
from src.data.preprocess import normalize_example

from app.backend.wikipedia_retrieval import search_wikipedia

# =========================================================
# CONFIG
# =========================================================

NUM_EXAMPLES = 300

TOP_K = 5

OUTPUT_PATH = (
    "data/processed/live_wikipedia_candidates.jsonl"
)

REQUEST_DELAY = 20.0

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading retrieval model...")

model = SentenceTransformer(MODEL_NAME)

# =========================================================
# SAVE HELPERS
# =========================================================

def append_jsonl(record, output_path):

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            ) + "\n"
        )

# =========================================================
# LOAD COMPLETED CLAIMS
# =========================================================

def load_completed_claims(output_path):

    completed = set()

    path = Path(output_path)

    if not path.exists():

        return completed

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            try:

                record = json.loads(line)

                completed.add(
                    record["claim"]
                )

            except Exception:

                continue

    return completed

# =========================================================
# LOAD DATA
# =========================================================

print("Loading FEVER data...")

raw_data = load_jsonl(
    "data/raw/train.jsonl"
)

print("Normalizing examples...")

data = [
    normalize_example(ex)
    for ex in raw_data[:NUM_EXAMPLES]
]

# =========================================================
# LOAD CACHE
# =========================================================

completed_claims = load_completed_claims(
    OUTPUT_PATH
)

print(
    f"Loaded {len(completed_claims)} cached claims"
)

# =========================================================
# RUN LIVE RETRIEVAL
# =========================================================

for i, example in enumerate(data, start=1):

    claim = example["claim"]

    gold_label = example["label"]

    print("\n" + "=" * 70)

    print(f"Example {i}/{len(data)}")

    print("Claim:", claim)

    # -----------------------------------------------------
    # SKIP COMPLETED CLAIMS
    # -----------------------------------------------------

    if claim in completed_claims:

        print("Skipping cached claim")

        continue

    # -----------------------------------------------------
    # RUN RETRIEVAL
    # -----------------------------------------------------

    try:

        evidence = search_wikipedia(
            claim=claim,
            model=model,
            top_k=TOP_K,
        )

        record = {

            "claim": claim,

            "gold_label": gold_label,

            "evidence": evidence,
        }

        append_jsonl(
            record,
            OUTPUT_PATH,
        )

        print(
            f"Saved {len(evidence)} evidence items"
        )

    except Exception as e:

        print(f"FAILED: {e}")

    # -----------------------------------------------------
    # RATE LIMIT PROTECTION
    # -----------------------------------------------------

    print(
        f"Sleeping {REQUEST_DELAY} seconds..."
    )

    time.sleep(REQUEST_DELAY)

# =========================================================
# DONE
# =========================================================

print("\nDone.")
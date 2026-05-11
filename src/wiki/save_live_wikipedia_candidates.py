"""
save_live_wikipedia_candidates.py

Runs LIVE Wikipedia retrieval and saves retrieved evidence
to disk for later evaluation.

This prevents:
    - repeated API requests
    - rate limiting
    - expensive reruns

Outputs:
    data/processed/live_wikipedia_candidates.jsonl
"""

import json
import time
from pathlib import Path

from src.data.data_loader import load_jsonl
from src.data.preprocess import normalize_example

from app.backend.wikipedia_retrieval import search_wikipedia

# =========================================================
# CONFIG
# =========================================================

NUM_EXAMPLES = 75

TOP_K = 5

OUTPUT_PATH = (
    "data/processed/live_wikipedia_candidates.jsonl"
)

SAVE_EVERY = 1

REQUEST_DELAY = 20.0

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
# RUN LIVE RETRIEVAL
# =========================================================

for i, example in enumerate(data, start=1):

    claim = example["claim"]

    gold_label = example["label"]

    print("\n" + "=" * 70)

    print(f"Example {i}/{len(data)}")

    print("Claim:", claim)

    try:

        evidence = search_wikipedia(
            claim=claim,
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

    print(
        f"Sleeping {REQUEST_DELAY} seconds..."
    )

    time.sleep(REQUEST_DELAY)

print("\nDone.")
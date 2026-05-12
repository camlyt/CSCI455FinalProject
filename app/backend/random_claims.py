import random

from src.data.data_loader import load_jsonl
from src.data.preprocess import normalize_example

# =========================================================
# LOAD FEVER DATA ON STARTUP
# =========================================================

print("Loading FEVER claims for random generator...")

raw_data = load_jsonl(
    "data/raw/train.jsonl"
)

FEVER_DATA = [
    normalize_example(ex)
    for ex in raw_data
]

print(
    f"Loaded {len(FEVER_DATA)} FEVER claims"
)

# =========================================================
# RANDOM CLAIM
# =========================================================


def get_random_claim():

    example = random.choice(FEVER_DATA)

    return {
        "claim": example["claim"],
        "label": example["label"],
    }
"""
evaluate_live_wikipedia_outputs.py

Evaluates verifier performance using
PRE-SAVED live Wikipedia retrieval outputs.

No internet requests are made here.
"""

import json
from collections import Counter

from app.backend.pipeline_service import verifier

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = (
    "data/processed/live_wikipedia_candidates.jsonl"
)

# =========================================================
# LOAD SAVED OUTPUTS
# =========================================================

records = []

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
) as file:

    for line in file:

        records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# =========================================================
# EVALUATION
# =========================================================

correct = 0
total = 0

prediction_counts = Counter()

for i, record in enumerate(records, start=1):

    claim = record["claim"]

    gold_label = record["gold_label"]

    evidence = record["evidence"]

    print("\n" + "=" * 70)

    print(f"Example {i}/{len(records)}")

    print("Claim:", claim)

    print("Gold:", gold_label)

    # -----------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------

    verifier_result = verifier.predict_with_scores(
        claim,
        evidence,
    )

    prediction = verifier_result["prediction"]

    prediction_counts[prediction] += 1

    print("Prediction:", prediction)

    if prediction == gold_label:

        correct += 1

        print("✅ Correct")

    else:

        print("❌ Incorrect")

    total += 1

    # -----------------------------------------------------
    # TOP EVIDENCE
    # -----------------------------------------------------

    if evidence:

        top_ev = evidence[0]

        print("\nTop Evidence:")

        print(
            f"[{top_ev['display_page']}]"
        )

        print(top_ev["text"])

# =========================================================
# FINAL RESULTS
# =========================================================

accuracy = (
    correct / total
    if total
    else 0
)

print("\n" + "=" * 70)
print("LIVE WIKIPEDIA RESULTS")
print("=" * 70)

print(f"Examples Evaluated: {total}")

print(f"Accuracy: {accuracy:.4f}")

print("\nPrediction Distribution:")

for label, count in prediction_counts.items():

    print(f"{label}: {count}")

print("\nDone.")
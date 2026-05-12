"""
evaluate_live_wikipedia_outputs.py

Evaluates verifier performance using
PRE-SAVED live Wikipedia retrieval outputs.

No internet requests are made here.
"""

import json
from collections import Counter

import numpy as np

from app.backend.pipeline_service import verifier

# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = (
    "data/processed/live_wikipedia_candidates.jsonl"
)

MAX_EXAMPLES = 100

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

# Limit evaluation size
records = records[:MAX_EXAMPLES]

print(f"Evaluating first {len(records)} records")

# =========================================================
# EVALUATION
# =========================================================

correct = 0
total = 0

prediction_counts = Counter()

gold_counts = Counter()

per_label_correct = Counter()

confusion = Counter()

entailment_scores = []
neutral_scores = []
contradiction_scores = []

high_confidence_wrong = []

no_evidence_count = 0

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

    gold_counts[gold_label] += 1

    confusion[(gold_label, prediction)] += 1

    if verifier_result["entailment"] is not None:

        entailment_scores.append(
            verifier_result["entailment"]
        )

        neutral_scores.append(
            verifier_result["neutral"]
        )

        contradiction_scores.append(
            verifier_result["contradiction"]
        )

    if not evidence:

        no_evidence_count += 1

    print("Prediction:", prediction)

    print(
        "Scores:",
        {
            "entailment":
                verifier_result["entailment"],
            "neutral":
                verifier_result["neutral"],
            "contradiction":
                verifier_result["contradiction"],
        }
    )

    if prediction == gold_label:

        correct += 1

        per_label_correct[gold_label] += 1

        print("✅ Correct")

    else:

        valid_scores = [
            s for s in [
                verifier_result["entailment"],
                verifier_result["neutral"],
                verifier_result["contradiction"],
            ]
            if s is not None
        ]

        best_score = (
            max(valid_scores)
            if valid_scores
            else 0.0
        )

        high_confidence_wrong.append(
            {
                "claim": claim,
                "gold": gold_label,
                "pred": prediction,
                "confidence": best_score,
            }
        )

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

# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

print("\nPrediction Distribution:")

for label, count in prediction_counts.items():

    print(f"{label}: {count}")

# =========================================================
# PER-LABEL ACCURACY
# =========================================================

print("\n" + "=" * 70)
print("PER-LABEL ACCURACY")
print("=" * 70)

for label in gold_counts:

    label_total = gold_counts[label]

    label_correct = per_label_correct[label]

    label_acc = (
        label_correct / label_total
        if label_total
        else 0
    )

    print(
        f"{label}: "
        f"{label_correct}/{label_total} "
        f"= {label_acc:.4f}"
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = [
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO",
]

for gold in labels:

    print(f"\nGold: {gold}")

    for pred in labels:

        count = confusion[(gold, pred)]

        print(
            f"  Pred {pred}: {count}"
        )

# =========================================================
# AVERAGE VERIFIER SCORES
# =========================================================

print("\n" + "=" * 70)
print("AVERAGE VERIFIER SCORES")
print("=" * 70)

print(
    f"Avg Entailment: "
    f"{np.mean(entailment_scores):.4f}"
)

print(
    f"Avg Neutral: "
    f"{np.mean(neutral_scores):.4f}"
)

print(
    f"Avg Contradiction: "
    f"{np.mean(contradiction_scores):.4f}"
)

# =========================================================
# RETRIEVAL COVERAGE
# =========================================================

print("\n" + "=" * 70)
print("RETRIEVAL COVERAGE")
print("=" * 70)

print(
    f"No Evidence Retrieved: "
    f"{no_evidence_count}"
)

coverage_rate = (
    1 - (no_evidence_count / total)
    if total
    else 0
)

print(
    f"Coverage Rate: "
    f"{coverage_rate:.4f}"
)

# =========================================================
# HIGH-CONFIDENCE FAILURES
# =========================================================

print("\n" + "=" * 70)
print("TOP HIGH-CONFIDENCE FAILURES")
print("=" * 70)

high_confidence_wrong = sorted(
    high_confidence_wrong,
    key=lambda x: x["confidence"],
    reverse=True,
)

for failure in high_confidence_wrong[:5]:

    print("\nClaim:", failure["claim"])

    print(
        f"Gold: {failure['gold']}"
    )

    print(
        f"Predicted: {failure['pred']}"
    )

    print(
        f"Confidence: "
        f"{failure['confidence']:.4f}"
    )

print("\nDone.")
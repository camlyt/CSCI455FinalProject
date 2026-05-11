from sentence_transformers import CrossEncoder
import numpy as np


class Verifier:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

        id2label = self.model.model.config.id2label
        print("Verifier label mapping:", id2label)

        self.entailment_idx = None
        self.contradiction_idx = None
        self.neutral_idx = None

        for idx, label in id2label.items():
            label_lower = label.lower()

            if "entail" in label_lower:
                self.entailment_idx = int(idx)
            elif "contrad" in label_lower:
                self.contradiction_idx = int(idx)
            elif "neutral" in label_lower:
                self.neutral_idx = int(idx)

        if self.entailment_idx is None or self.contradiction_idx is None or self.neutral_idx is None:
            raise ValueError(f"Could not identify label indices from: {id2label}")

        print("Entailment index:", self.entailment_idx)
        print("Contradiction index:", self.contradiction_idx)
        print("Neutral index:", self.neutral_idx)

    def predict(self, claim, evidence_list):
        if not evidence_list:
            return "NOT ENOUGH INFO"

        # Combine retrieved evidence into one context block.
        combined_evidence = " ".join(ev["text"] for ev in evidence_list)

        scores = self.model.predict([(combined_evidence, claim)])
        scores = np.array(scores)

        # scores shape may be (3,) or (1, 3), depending on model behavior.
        if scores.ndim == 2:
            scores = scores[0]

        entailment_score = scores[self.entailment_idx]
        contradiction_score = scores[self.contradiction_idx]
        neutral_score = scores[self.neutral_idx]

        print("Entailment:", entailment_score)
        print("Contradiction:", contradiction_score)
        print("Neutral:", neutral_score)

        best_idx = int(np.argmax(scores))

        if best_idx == self.entailment_idx:
            return "SUPPORTS"
        elif best_idx == self.contradiction_idx:
            return "REFUTES"
        else:
            return "NOT ENOUGH INFO"

    def predict_with_scores(self, claim, evidence_list):
        if not evidence_list:
            return {
                "prediction": "NOT ENOUGH INFO",
                "entailment": None,
                "contradiction": None,
                "neutral": None
            }

        combined_evidence = " ".join(ev["text"] for ev in evidence_list)

        scores = self.model.predict([(combined_evidence, claim)])
        scores = np.array(scores)

        if scores.ndim == 2:
            scores = scores[0]

        entailment_score = float(scores[self.entailment_idx])
        contradiction_score = float(scores[self.contradiction_idx])
        neutral_score = float(scores[self.neutral_idx])

        best_idx = int(np.argmax(scores))

        if best_idx == self.entailment_idx:
            prediction = "SUPPORTS"
        elif best_idx == self.contradiction_idx:
            prediction = "REFUTES"
        else:
            prediction = "NOT ENOUGH INFO"

        return {
            "prediction": prediction,
            "entailment": entailment_score,
            "contradiction": contradiction_score,
            "neutral": neutral_score
        }
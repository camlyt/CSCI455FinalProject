from sentence_transformers import CrossEncoder
import numpy as np


class Verifier:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    def predict(self, claim, evidence_list):
        if not evidence_list:
            return "NOT ENOUGH INFO"

        pairs = [(claim, ev["text"]) for ev in evidence_list]
        scores = self.model.predict(pairs)

        scores = np.array(scores)

        # scores: [entailment, neutral, contradiction]

        entailment_scores = scores[:, 0]
        contradiction_scores = scores[:, 2]

        max_entailment = entailment_scores.max()
        max_contradiction = contradiction_scores.max()

        # threshold-based decision
        if max_entailment > 0.8:
            return "SUPPORTS"
        elif max_contradiction > 0.8:
            return "REFUTES"
        else:
            return "NOT ENOUGH INFO"
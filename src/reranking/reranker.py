from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, claim, candidates, top_k=5):
        """
        candidates: list of dicts with 'text'
        """

        pairs = [(claim, c["text"]) for c in candidates]

        scores = self.model.predict(pairs)

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]
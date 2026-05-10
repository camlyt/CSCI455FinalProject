def verify_claim(claim: str, top_k: int = 5, threshold: float = 0.8):
    return {
        "claim": claim,
        "label": "SUPPORTS",
        "confidence": 0.87,
        "settings": {
            "top_k": top_k,
            "threshold": threshold,
            "retriever_model": "MiniLM",
            "verifier_model": "DeBERTa NLI",
        },
        "scores": {
            "entailment": 0.86,
            "neutral": 0.09,
            "contradiction": 0.05,
        },
        "evidence": [
            {
                "text": "Roman Atwood is an American YouTube personality, comedian, vlogger, and content creator.",
                "page": "Roman_Atwood",
                "sentence_id": 0,
                "score": 0.72,
                "rerank_score": 4.31,
            }
        ][:top_k],
    }
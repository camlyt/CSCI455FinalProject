from sentence_transformers import SentenceTransformer
import faiss

from src.retrieval.query_faiss_targeted_subset import (search_claim,load_metadata,)
from src.reranking.reranker import Reranker
from src.verification.verifier import Verifier


model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(model_name)

print("Loading FAISS index...")
index = faiss.read_index("data/index/wiki_targeted_subset.index")

print("Loading metadata...")
metadata = load_metadata("data/index/wiki_targeted_subset_metadata.json")

print("Loading reranker...")
reranker = Reranker()

print("Loading verifier...")
verifier = Verifier()

print("Backend pipeline ready ✅")


def verify_claim(claim: str, top_k: int = 5, threshold: float = 0.8):
    print("\n--- VERIFY REQUEST ---")
    print("Claim:", claim)
    print("Top K:", top_k)
    print("Threshold:", threshold)

    evidence = search_claim(
        claim=claim,
        model=model,
        index=index,
        metadata=metadata,
        reranker=reranker,
        top_k=top_k,
        candidate_k=50,
    )

    verifier_result = verifier.predict_with_scores(claim, evidence)

    label = verifier_result["prediction"]

    entailment = verifier_result["entailment"]
    contradiction = verifier_result["contradiction"]
    neutral = verifier_result["neutral"]

    score_values = [
        s for s in [entailment, contradiction, neutral]
        if s is not None
    ]

    confidence = max(score_values) if score_values else 0.0

    print("Predicted label:", label)
    print("Scores:", verifier_result)
    print("Evidence count:", len(evidence))
    print("----------------------\n")

    return {
        "claim": claim,
        "label": label,
        "confidence": confidence,
        "settings": {
            "top_k": top_k,
            "threshold": threshold,
            "retriever_model": "MiniLM + FAISS",
            "verifier_model": "DeBERTa NLI",
        },
        "scores": {
            "entailment": entailment,
            "neutral": neutral,
            "contradiction": contradiction,
        },
        "evidence": evidence,
    }
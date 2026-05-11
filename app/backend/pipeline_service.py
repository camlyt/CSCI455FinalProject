from sentence_transformers import SentenceTransformer
import faiss

from src.query_faiss_targeted_subset import search_claim, load_metadata
from src.reranker import Reranker
from src.verifier import Verifier


# ---- Load once at startup (IMPORTANT) ----

model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(model_name)

print("Loading FAISS index...")
index = faiss.read_index("data/index/wiki_targeted_subset.index")

print("Loading metadata...")
metadata = load_metadata("data/index/wiki_targeted_subset_metadata.json")

print("Loading reranker + verifier...")
reranker = Reranker()
verifier = Verifier()

print("Backend pipeline ready ✅")


# ---- Main function used by FastAPI ----

def verify_claim(claim: str, top_k: int = 5, threshold: float = 0.8):
    
    print("\n--- VERIFY REQUEST ---")
    print("Claim:", claim)

    # 1. Retrieve + rerank
    results = search_claim(
        claim,
        model=model,
        index=index,
        metadata=metadata,
        reranker=reranker,
        top_k=top_k
    )

    print("\nTop Evidence:")
    for i, r in enumerate(results[:3]):
        print(f"{i+1}.", r["text"][:120])

    # 2. Predict label
    label = verifier.predict(claim, results)

    print("\nPredicted Label:", label)
    print("----------------------\n")

    confidence = 0.85

    return {
        "claim": claim,
        "label": label,
        "confidence": confidence,
        "settings": {
            "top_k": top_k,
            "threshold": threshold,
            "retriever_model": "MiniLM",
            "verifier_model": "DeBERTa NLI",
        },
        "evidence": results
    }
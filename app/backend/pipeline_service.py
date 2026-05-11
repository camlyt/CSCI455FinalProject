from sentence_transformers import SentenceTransformer
import faiss

from src.retrieval.query_faiss_targeted_subset import (
    search_claim,
    load_metadata,
)
from app.backend.wikipedia_retrieval import (
    search_wikipedia,
)
from src.reranking.reranker import Reranker
from src.verification.verifier import Verifier


# =========================================================
# RETRIEVER MODELS
# =========================================================

RETRIEVER_MODELS = {

    "minilm": "sentence-transformers/all-MiniLM-L6-v2",

    "bge": "BAAI/bge-small-en-v1.5",

    "e5": "intfloat/e5-small-v2",
}


# =========================================================
# INDEX PATHS
# =========================================================

INDEX_PATHS = {

    "minilm":
        "data/index/wiki_targeted_subset_minilm.index",

    "bge":
        "data/index/wiki_targeted_subset_bge.index",

    "e5":
        "data/index/wiki_targeted_subset_e5.index",
}


# =========================================================
# METADATA PATHS
# =========================================================

METADATA_PATHS = {

    "minilm":
        "data/index/wiki_targeted_subset_minilm_metadata.json",

    "bge":
        "data/index/wiki_targeted_subset_bge_metadata.json",

    "e5":
        "data/index/wiki_targeted_subset_e5_metadata.json",
}


# =========================================================
# LOAD RETRIEVERS
# =========================================================

print("Loading retrieval models...")

retrievers = {}

for key, model_name in RETRIEVER_MODELS.items():

    print(f"Loading retriever: {key}")

    retrievers[key] = SentenceTransformer(model_name)

print("Retrievers loaded ✅")


# =========================================================
# LOAD FAISS INDEXES
# =========================================================

print("Loading FAISS indexes...")

indexes = {}

for key, path in INDEX_PATHS.items():

    print(f"Loading index: {key}")

    indexes[key] = faiss.read_index(path)

print("Indexes loaded ✅")


# =========================================================
# LOAD METADATA
# =========================================================

print("Loading metadata...")

metadata_store = {}

for key, path in METADATA_PATHS.items():

    print(f"Loading metadata: {key}")

    metadata_store[key] = load_metadata(path)

print("Metadata loaded ✅")


# =========================================================
# LOAD RERANKER
# =========================================================

print("Loading reranker...")

reranker = Reranker()

print("Reranker loaded ✅")


# =========================================================
# LOAD VERIFIER
# =========================================================

print("Loading verifier...")

verifier = Verifier()

print("Verifier loaded ✅")


print("Backend pipeline ready ✅")


# =========================================================
# TEMPORARY INTERNET RETRIEVAL PLACEHOLDER
# =========================================================


# =========================================================
# MAIN VERIFICATION PIPELINE
# =========================================================

def verify_claim(
    claim: str,

    top_k: int = 5,

    retriever: str = "minilm",

    use_reranker: bool = True,

    retrieval_mode: str = "fever",
):

    print("\n--- VERIFY REQUEST ---")

    print("Claim:", claim)

    print("Top K:", top_k)

    print("Retriever:", retriever)

    print("Use Reranker:", use_reranker)

    print("Retrieval Mode:", retrieval_mode)

    # =====================================================
    # INTERNET RETRIEVAL
    # =====================================================

    if retrieval_mode == "internet":

        print("Using live Wikipedia retrieval...")

        evidence = search_wikipedia(
            claim=claim,
            top_k=top_k * 3,
        )

        # -------------------------------------------------
        # OPTIONAL RERANKING
        # -------------------------------------------------

        if use_reranker:

            evidence = reranker.rerank(
                claim,
                evidence,
            )

        evidence = evidence[:top_k]

    # =====================================================
    # FEVER RETRIEVAL
    # =====================================================

    else:

        print("Using FEVER retrieval pipeline...")

        active_reranker = (
            reranker
            if use_reranker
            else None
        )

        evidence = search_claim(
            claim=claim,

            model=retrievers[retriever],

            index=indexes[retriever],

            metadata=metadata_store[retriever],

            retriever_id=retriever,

            reranker=active_reranker,

            top_k=top_k,

            candidate_k=50,
        )

        print("\nTop evidence:\n")

        # logs for switching retrievers
        for i, ev in enumerate(evidence, start=1):

            print(f"{i}. {ev['page']}")
            print(f"Score: {ev['score']}")

            if ev.get("rerank_score") is not None:
                print(f"Rerank: {ev['rerank_score']}")

            print(ev["text"][:200])
            print("-" * 50)

        # -------------------------------------------------
        # FRONTEND METADATA
        # -------------------------------------------------

        for ev in evidence:

            ev["display_page"] = (
                ev["page"].replace("_", " ")
            )

            ev["url"] = (
                f"https://en.wikipedia.org/wiki/{ev['page']}"
            )

            if "rerank_score" not in ev:

                ev["rerank_score"] = None

    # =====================================================
    # VERIFICATION
    # =====================================================

    verifier_result = verifier.predict_with_scores(
        claim,
        evidence
    )

    label = verifier_result["prediction"]

    entailment = verifier_result["entailment"]

    contradiction = verifier_result["contradiction"]

    neutral = verifier_result["neutral"]

    # =====================================================
    # CONFIDENCE
    # =====================================================

    score_values = [
        s for s in [
            entailment,
            contradiction,
            neutral,
        ]
        if s is not None
    ]

    confidence = (
        max(score_values)
        if score_values
        else 0.0
    )

    print("Predicted label:", label)

    print("Scores:", verifier_result)

    print("Evidence count:", len(evidence))

    print("----------------------\n")

    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "claim": claim,

        "label": label,

        "confidence": confidence,

        "retrieval_mode": retrieval_mode,

        "settings": {

            "top_k": top_k,

            "retriever_model": retriever,

            "use_reranker": use_reranker,

            "verifier_model": "DeBERTa NLI",
        },

        "scores": {

            "entailment": entailment,

            "neutral": neutral,

            "contradiction": contradiction,
        },

        "evidence": evidence,
    }
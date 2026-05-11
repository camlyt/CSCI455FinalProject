import subprocess
import sys

"""
run_pipeline.py

Central pipeline runner for the FEVER claim verification project.

Project structure:
    src/data/           -> dataset loading + preprocessing
    src/retrieval/      -> FAISS indexing + dense retrieval
    src/reranking/      -> CrossEncoder reranking
    src/verification/   -> NLI verification
    src/evaluation/     -> metrics + error analysis

Run from project root using:

    python -m src.pipeline.run_pipeline
"""

PIPELINE = [

    # =========================================================
    # DATA PREPROCESSING
    # =========================================================

    ("src.data.data_loader", False),
    ("src.data.preprocess", False),
    ("src.data.inspect_wiki", False),
    ("src.data.wiki_preprocess", False),
    ("src.data.validate_corpus", False),

    # =========================================================
    # RETRIEVAL / INDEX BUILDING
    # =========================================================

    ("src.retrieval.build_corpus_subset", True),
    ("src.retrieval.build_targeted_subset", True),
    ("src.retrieval.build_faiss_subset", True),
    ("src.retrieval.build_faiss_targeted_subset", True),

    # =========================================================
    # OFFICIAL FINAL PIPELINE
    # =========================================================

    ("src.retrieval.save_dense_candidates", True),
    ("src.reranking.rerank_saved_candidates", True),
    ("src.verification.evaluate_verifier_from_outputs", True),
    ("src.evaluation.analyze_pipeline_errors", True),
]


def run_step(module_name: str) -> None:
    """
    Execute one pipeline step as a Python module.
    """

    print(f"\n{'=' * 70}")
    print(f"RUNNING: {module_name}")
    print(f"{'=' * 70}")

    try:
        subprocess.run(
            [sys.executable, "-m", module_name],
            check=True
        )

        print(f"\n✅ Completed: {module_name}")

    except subprocess.CalledProcessError:
        print(f"\n❌ Pipeline failed at: {module_name}")
        sys.exit(1)


def main() -> None:
    """
    Run all enabled pipeline steps in order.
    """

    print("\nStarting FEVER claim verification pipeline...\n")

    for module_name, should_run in PIPELINE:

        if not should_run:
            print(f"⏭️  Skipping: {module_name}")
            continue

        run_step(module_name)

    print(f"\n{'=' * 70}")
    print("🎉 Pipeline completed successfully!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
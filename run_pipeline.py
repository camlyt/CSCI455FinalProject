import subprocess
import sys

"""
run_pipeline.py

Pipeline runner for the current CSCI455 final project workflow.

Current official workflow:
1. Build a targeted Wikipedia subset from FEVER evidence pages.
2. Build a FAISS index over the targeted subset.
3. Run dense retrieval + reranking and save retrieved evidence.
4. Run verifier evaluation from saved retrieval outputs.
5. Run error analysis.

Older preprocessing/debug scripts are kept in the repository but skipped here
because their outputs have already been generated or they are only used for
inspection/debugging.
"""

PIPELINE = [
    # -----------------------------
    # Data + preprocessing/debug steps
    # Already completed or optional
    # -----------------------------
    ("data_loader", False),
    ("preprocess", False),
    ("inspect_wiki", False),
    ("wiki_preprocess", False),
    ("validate_corpus", False),

    # -----------------------------
    # Index-building steps
    # Run only if changing NUM_EXAMPLES or rebuilding data/index files
    # -----------------------------
    ("build_targeted_subset", False),
    ("build_faiss_targeted_subset", False),

    # -----------------------------
    # Official final evaluation workflow
    # -----------------------------
    ("save_dense_candidates", True),
    ("rerank_saved_candidates", True),
    ("evaluate_verifier_from_outputs", True),
    ("analyze_pipeline_errors", True),
]


def run_step(module_name):
    print(f"\n=== Running {module_name} ===")

    try:
        subprocess.run(
            [sys.executable, "-m", f"src.{module_name}"],
            check=True
        )
        print(f"✅ Completed {module_name}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed at {module_name}")
        sys.exit(1)


def main():
    for module_name, should_run in PIPELINE:
        if not should_run:
            print(f"⏭️ Skipping {module_name}")
            continue

        run_step(module_name)

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
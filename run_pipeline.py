import subprocess
import sys

"""
Pipeline for CSCI455 Final Project

This version reflects the CURRENT correct workflow:
- Old subset pipeline is removed
- Uses targeted subset pipeline for meaningful evaluation
- Allows skipping expensive steps after first run
"""

PIPELINE = [
    # -----------------------------
    # Data + preprocessing (run once)
    # -----------------------------
    ("data_loader", False),
    ("preprocess", False),
    ("inspect_wiki", False),      # optional debug
    ("wiki_preprocess", False),   # VERY expensive — run once
    ("validate_corpus", False),

    # -----------------------------
    # Targeted retrieval pipeline (core)
    # -----------------------------
    ("build_targeted_subset", True),        # run once unless dataset changes
    ("build_faiss_targeted_subset", True), # run once unless rebuilding index
    ("query_faiss_targeted_subset", True),  # quick sanity check
    ("evaluate_retrieval", True),           # YOUR MAIN STEP
]


def run_step(module_name):
    print(f"\n=== Running {module_name} ===")

    try:
        subprocess.run(
            ["python", "-m", f"src.{module_name}"],
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
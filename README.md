# FEVER Fact Verification System via RAG

This project implements a Retrieval-Augmented Generation (RAG) style fact-verification pipeline using the FEVER dataset. The system takes a natural language claim, retrieves relevant Wikipedia evidence, reranks retrieved evidence, and predicts whether the claim is:

- `SUPPORTS`
- `REFUTES`
- `NOT ENOUGH INFO`

The project is built around a modular pipeline so that retrieval, reranking, verification, and error analysis can be tested separately.

---

## Project Overview

The system has four main stages:

1. **Data Processing**
   - Load FEVER claims and evidence annotations.
   - Normalize FEVER evidence into clean `(page, sentence_id)` references.
   - Convert FEVER Wikipedia dump files into a sentence-level retrieval corpus.

2. **Retrieval**
   - Encode Wikipedia sentences using `sentence-transformers/all-MiniLM-L6-v2`.
   - Build a FAISS index over sentence embeddings.
   - Retrieve top candidate evidence sentences for each claim.

3. **Reranking**
   - Use a CrossEncoder reranker to reorder dense retrieval candidates.
   - Improve the ranking of gold evidence within the top-k results.

4. **Verification**
   - Use an NLI CrossEncoder verifier to classify claims based on retrieved evidence.
   - The verifier receives retrieved evidence as the premise and the claim as the hypothesis.

---

## Repository Structure

```
CSCI455FinalProject/
│
├── data/
│   ├── raw/              # Raw FEVER data and wiki dump files (not committed)
│   ├── processed/        # Processed JSONL outputs (not committed)
│   └── index/            # FAISS indexes and metadata (not committed)
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── inspect_wiki.py
│   ├── wiki_preprocess.py
│   ├── validate_corpus.py
│   │
│   ├── build_targeted_subset.py
│   ├── build_faiss_targeted_subset.py
│   ├── save_dense_candidates.py
│   ├── rerank_saved_candidates.py
│   ├── evaluate_verifier_from_outputs.py
│   ├── analyze_pipeline_errors.py
│   │
│   ├── query_faiss_targeted_subset.py
│   ├── reranker.py
│   ├── verifier.py
│   └── debug_verifier.py
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```
---
## Setup
1. Create a virtual environment

Python 3.12 is recommended.

```
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

2. Install PyTorch first

PyTorch should be installed separately before the rest of the requirements.
```
pip install torch torchvision torchaudio
```
3. Install project dependencies
```
pip install -r requirements.txt
```
The recommended requirements.txt is:
```
pandas
numpy<2
scikit-learn
faiss-cpu==1.8.0
sentence-transformers==2.7.0
transformers>=4.43.3
tqdm
fastapi
uvicorn
```

---
## Data Requirements
This project uses the FEVER dataset.
Required files:
1. FEVER training dataset
- Example: train.jsonl
2. FEVER pre-processed Wikipedia pages

- Example folder: wiki-pages/
- Contains files such as:
  - wiki-001.jsonl 
  - wiki-002.jsonl
---
### Expected local structure:
```
data/raw/train.jsonl
data/raw/wiki-pages/wiki-001.jsonl
data/raw/wiki-pages/wiki-002.jsonl
```
Raw data files are not committed to GitHub.

---
## Current Official Pipeline
The official pipeline is split into stages to avoid local FAISS / CrossEncoder stability issues on macOS.
The final workflow is:
```
save_dense_candidates.py    
↓
rerank_saved_candidates.py    
↓
evaluate_verifier_from_outputs.py    
↓
analyze_pipeline_errors.py
```
Run the current official pipeline with:
```
python run_pipeline.py
```
This assumes the targeted subset and FAISS index have already been built.


--- 
### Rebuilding the Targeted Subset and FAISS Index
Only run these steps if:
- NUM_EXAMPLES changes 
- the targeted subset is deleted 
- the FAISS index is deleted 
- the embedding model changes 
- the data changes

Run:
```
python -m src.build_targeted_subset
python -m src.build_faiss_targeted_subset
```
These generate:
```
data/processed/wiki_targeted_subset.jsonl
data/index/wiki_targeted_subset.index
data/index/wiki_targeted_subset_metadata.json
```
After that, run:
```
python run_pipeline.py
```
---
 Pipeline Scripts
---
`build_targeted_subset.py`

Builds a targeted Wikipedia sentence subset using FEVER gold evidence pages.
Input:
```
data/raw/train.jsonl
data/processed/wiki_sentences.jsonl
```
Output:
```
data/processed/wiki_targeted_subset.jsonl
```
---
`build_faiss_targeted_subset.py`

Builds a FAISS index over the targeted Wikipedia subset.

Input:
```
data/processed/wiki_targeted_subset.jsonl
```
Output:
```
data/index/wiki_targeted_subset.index
data/index/wiki_targeted_subset_metadata.json
```
---
`save_dense_candidates.py`

Runs dense retrieval using FAISS and saves top candidate evidence.
Input:
```
data/index/wiki_targeted_subset.index
data/index/wiki_targeted_subset_metadata.json
```
Output:
```
data/processed/dense_candidate_outputs.jsonl
```
---
`rerank_saved_candidates.py`

Loads dense retrieval candidates and reranks them using a CrossEncoder reranker.

Input:
```
data/processed/dense_candidate_outputs.jsonl
```
Output:
```
data/processed/reranked_retrieval_outputs.jsonl
```
---
`evaluate_verifier_from_outputs.py`

Loads reranked evidence and evaluates the verifier.

Input:
```
data/processed/reranked_retrieval_outputs.jsonl
```
Output:
Final pipeline accuracy

`analyze_pipeline_errors.py`

Runs error analysis on incorrect final predictions.

Input:
```
data/processed/reranked_retrieval_outputs.jsonl
```
Output:
```
data/processed/pipeline_errors.jsonl
```
---
## Current Results
Evaluation was performed on a targeted subset built from the first 100 FEVER examples. After removing examples with no usable gold evidence, 75 examples were evaluated.
### Retrieval Results
| Pipeline | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| Dense Retrieval | 0.6667 | 0.9333 | 0.9733 |
| Dense Retrieval + Reranker | 0.8133 | 0.9467 | 0.9733 |### Final Pipeline Result

### Final Pipeline Result

| Pipeline | Accuracy |
|---|---:|
| Dense Retrieval + Reranker + Verifier | 0.7733 |

--- 
## Error Analysis
The final reranked pipeline produced 17 errors out of 75 evaluated examples.

| Error Type | Count |
|---|---:|
| Retrieval miss | 1 |
| Verifier wrong despite gold evidence | 9 |
| Verifier too conservative | 7 |

This suggests that retrieval is generally effective, and most remaining errors come from the verifier stage rather than missing evidence.

---
## Important Implementation Notes
### FAISS normalization
The project uses NumPy normalization instead of `faiss.normalize_L2()` because `faiss.normalize_L2()` caused local native crashes on macOS.
Instead of:
```
faiss.normalize_L2(embeddings)
```
we use:
```
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.clip(norms, a_min=1e-12, a_max=None)
embeddings = embeddings / norms
embeddings = embeddings.astype("float32")
```

### Split evaluation design
The pipeline is intentionally split into multiple scripts because loading FAISS, SentenceTransformer, reranker CrossEncoder, and verifier CrossEncoder in the same Python process caused local instability.
The stable approach is:
```
FAISS + SentenceTransformer    
    → save candidates
CrossEncoder reranker    
    → save reranked evidence
CrossEncoder verifier    
    → evaluate final labels
```
This also makes the pipeline easier to debug and reproduce.

### Verifier input order
The verifier uses an NLI model, which expects:
premise, hypothesis
For this project:
```
premise = retrieved evidence
hypothesis = claim
```
So the correct input order is:
```
(combined_evidence, claim)
```
not:
```
(claim, combined_evidence)
```

### Running Individual Steps
Run dense candidate retrieval:
```
python -m src.save_dense_candidates
```
Run reranking:
```
python -m src.rerank_saved_candidates
```
Run verifier evaluation:
```
python -m src.evaluate_verifier_from_outputs
```
Run error analysis:
```
python -m src.analyze_pipeline_errors
```
Run all official evaluation steps:
```
python run_pipeline.py
```
--- 

## Experimental / Debug Scripts
The following files were used during development and are not part of the current official evaluation workflow:
```
build_corpus_subset.py
build_faiss_subset.py
query_faiss_subset.py
evaluate_full_pipeline.py
evaluate_retrieval_with_reranker.py
evaluate_retrieval_save_outputs.py
debug_verifier.py
```
They may still be useful for debugging, but the official pipeline is defined in run_pipeline.py.

--- 
## Launch the App

Start the backend server from the project root:
```
python -m uvicorn app.backend.main:app --reload
```

Frontend setup (React + Vite)
Oppen up a new terminal.
```
cd app/frontend
npm isntall
npm run dev
```
The app will be hosted at: http://localhost:3000

--- 
## Limitations
This project currently uses a targeted subset of the Wikipedia corpus rather than the full FEVER Wikipedia corpus for evaluation. This makes the experiment computationally manageable and useful for controlled testing, but it is not a full FEVER benchmark evaluation.
### Known limitations:
- The targeted subset assumes relevant evidence pages are already included.

- Some FEVER examples require multi-hop reasoning across multiple evidence sentences.

- The verifier can be too conservative and predict NOT ENOUGH INFO even when relevant evidence is retrieved.

- Some failures occur when gold evidence is retrieved but the verifier misinterprets the evidence.

- Full-corpus retrieval would require more compute and a more scalable indexing setup.


--- 
## Future Work
Potential improvements include:
- Scaling retrieval to the full FEVER Wikipedia corpus.

- Improving entity resolution and page retrieval.

- Fine-tuning the verifier on FEVER claim-evidence pairs.

- Adding a stronger multi-hop reasoning component.

- Improving confidence scoring for UI display.

- Integrating the final pipeline into a FastAPI backend and React frontend.

- Running larger-scale evaluation in a Linux or Colab environment.


--- 
## Final Summary

The current system successfully implements a modular RAG-style fact verification pipeline. On the targeted 100-example FEVER evaluation, the final dense retrieval + reranker + verifier pipeline achieved:

| Metric | Score |
|---|---:|
| Recall@1 | 0.8133 |
| Recall@5 | 0.9467 |
| Recall@10 | 0.9733 |
| Accuracy | 0.7733 |

The error analysis shows that retrieval is strong, and future improvements should focus primarily on the verifier and multi-hop reasoning.

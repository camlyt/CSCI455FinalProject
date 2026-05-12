import re
import time
import requests
import numpy as np

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# =========================================================
# SIMPLE IN-MEMORY PAGE CACHE
# =========================================================

PAGE_CACHE = {}

# =========================================================
# REQUEST HEADERS
# =========================================================

HEADERS = {
    "User-Agent": "CSCI455ClaimVerification/1.0"
}

# =========================================================
# REQUEST SETTINGS
# =========================================================

REQUEST_DELAY = 1.0

MAX_RETRIES = 3

RETRY_DELAY = 2.0

# =========================================================
# WIKIPEDIA SEARCH
# =========================================================

def wikipedia_search(claim: str, top_k: int = 3):

    time.sleep(REQUEST_DELAY)

    search_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": claim,
        "format": "json",
        "srlimit": top_k,
    }

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.get(
                search_url,
                params=params,
                headers=HEADERS,
                timeout=10,
            )

            print(
                "Wikipedia search status:",
                response.status_code
            )

            response.raise_for_status()

            data = response.json()

            return data.get(
                "query",
                {}
            ).get(
                "search",
                []
            )

        except Exception as e:

            print(
                f"Search retry {attempt + 1} failed: {e}"
            )

            time.sleep(RETRY_DELAY)

    print("Wikipedia search failed")

    return []

# =========================================================
# PAGE EXTRACT
# =========================================================

def fetch_page_extract(title: str):

    # Cache hit
    if title in PAGE_CACHE:

        print(f"Cache hit: {title}")

        return PAGE_CACHE[title]

    time.sleep(REQUEST_DELAY)

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=10,
            )

            print(
                f"Page fetch status ({title}):",
                response.status_code
            )

            response.raise_for_status()

            data = response.json()

            pages = data.get(
                "query",
                {}
            ).get(
                "pages",
                {}
            )

            page = next(iter(pages.values()))

            extract = page.get(
                "extract",
                ""
            )

            PAGE_CACHE[title] = extract

            return extract

        except Exception as e:

            print(
                f"Retry {attempt + 1} failed for {title}: {e}"
            )

            time.sleep(RETRY_DELAY)

    print(f"Failed to fetch page: {title}")

    return ""

# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_sentences(text: str):

    text = text.replace("\n", " ")

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 40:
            continue

        if sentence.startswith("="):
            continue

        cleaned.append(sentence)

    return cleaned

# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(a, b):

    return np.dot(a, b)

# =========================================================
# SEMANTIC RANKING
# =========================================================

def semantic_rank_sentences(
    claim: str,
    sentences: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 15,
):

    if not sentences:
        return []

    sentence_texts = [
        s["text"]
        for s in sentences
    ]

    # E5 formatting
    if "e5" in model.__class__.__name__.lower():

        query_text = "query: " + claim

        sentence_texts = [
            "passage: " + t
            for t in sentence_texts
        ]

    else:

        query_text = claim

    # -----------------------------------------------------
    # EMBED QUERY
    # -----------------------------------------------------

    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    query_embedding /= np.linalg.norm(
        query_embedding
    )

    # -----------------------------------------------------
    # EMBED SENTENCES
    # -----------------------------------------------------

    sentence_embeddings = model.encode(
        sentence_texts,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    sentence_embeddings = (
        sentence_embeddings.astype("float32")
    )

    sentence_embeddings /= np.linalg.norm(
        sentence_embeddings,
        axis=1,
        keepdims=True,
    )

    # -----------------------------------------------------
    # COMPUTE SCORES
    # -----------------------------------------------------

    scored_sentences = []

    for sentence, embedding in zip(
        sentences,
        sentence_embeddings,
    ):

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        sentence["score"] = float(similarity)

        scored_sentences.append(sentence)

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    scored_sentences.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return scored_sentences[:top_k]

# =========================================================
# MAIN RETRIEVAL PIPELINE
# =========================================================

def search_wikipedia(
    claim: str,
    model: SentenceTransformer,
    top_k: int = 5,
):

    # -----------------------------------------------------
    # SEARCH WIKIPEDIA
    # -----------------------------------------------------

    search_results = wikipedia_search(
        claim,
        top_k=3,
    )

    # -----------------------------------------------------
    # COLLECT ALL CANDIDATE SENTENCES
    # -----------------------------------------------------

    candidates = []

    evidence_id = 0

    for result in search_results:

        title = result["title"]

        url_title = title.replace(
            " ",
            "_"
        )

        page_url = (
            f"https://en.wikipedia.org/wiki/{url_title}"
        )

        print(f"Fetching page: {title}")

        try:

            extract = fetch_page_extract(title)

        except Exception as e:

            print(
                f"Failed to fetch {title}: {e}"
            )

            continue

        if not extract:
            continue

        sentences = split_sentences(extract)

        for sentence in sentences:

            candidates.append({

                "page": url_title,

                "display_page": title,

                "sentence_id": evidence_id,

                "text": sentence,

                "score": 0.0,

                "rerank_score": None,

                "url": page_url,
            })

            evidence_id += 1

    print(
        f"Collected {len(candidates)} candidate sentences"
    )

    # -----------------------------------------------------
    # SEMANTIC RANKING
    # -----------------------------------------------------

    ranked_sentences = semantic_rank_sentences(
        claim=claim,
        sentences=candidates,
        model=model,
        top_k=top_k * 5,
    )

    print(
        f"Selected {len(ranked_sentences)} semantic candidates"
    )

    return ranked_sentences
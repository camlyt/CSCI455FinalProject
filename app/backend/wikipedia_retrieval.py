import re
import time
import requests
from typing import List, Dict, Any

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
# REQUEST HELPERS
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

    # -----------------------------------------------------
    # CACHE HIT
    # -----------------------------------------------------

    if title in PAGE_CACHE:

        print(f"Cache hit: {title}")

        return PAGE_CACHE[title]

    # -----------------------------------------------------
    # RATE LIMIT PROTECTION
    # -----------------------------------------------------

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
# MAIN RETRIEVAL PIPELINE
# =========================================================

def search_wikipedia(claim: str, top_k: int = 5):

    search_results = wikipedia_search(
        claim,
        top_k=min(top_k, 3),
    )

    evidence = []

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

        # -------------------------------------------------
        # LIMIT SENTENCES PER PAGE
        # -------------------------------------------------

        for sentence in sentences[:3]:

            evidence.append({
                "page": url_title,
                "display_page": title,
                "sentence_id": evidence_id,
                "text": sentence,
                "score": 1.0,
                "rerank_score": None,
                "url": page_url,
            })

            evidence_id += 1

    print(
        f"Collected {len(evidence)} evidence sentences"
    )

    return evidence
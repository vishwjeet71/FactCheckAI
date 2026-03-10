"""
All the process related to data collection and filtering
will happen here.
"""

import trafilatura, json, tldextract
from concurrent.futures import ThreadPoolExecutor, as_completed
from similarity import ModelFunctions


# ── Standardised response builders ──────────────────────────────────────── #

def _ok(data: dict) -> str:
    """Successful result with matched articles."""
    return {"status": "success", "results": data}

def _no_match(fetched: int, scored: int, threshold: float) -> str:
    """Articles were fetched and scored, but none crossed the threshold."""
    return {
        "status": "no_match",
        "reason": (
            f"Fetched {fetched} article(s) and scored {scored}, "
            f"but none had a similarity score >= {threshold}. "
            "The available content may not be directly related to the input topic."
        )
    }

def _no_content(total_links: int, failed: int) -> str:
    """Every URL either failed to fetch or returned no extractable text."""
    return {
        "status": "error",
        "error": (
            f"Could not retrieve content from any of the {total_links} link(s). "
            f"{failed} link(s) failed during fetch/extraction. "
            "Possible causes: paywalls, network timeouts, bot-blocking, or invalid URLs."
        )
    }

def _bad_input(detail: str) -> str:
    """Caller passed invalid arguments."""
    return {"status": "error", "error": f"Invalid input — {detail}"}

def _internal_error(context: str, exc: Exception) -> str:
    """Unexpected exception in a named context."""
    return {
        "status": "error",
        "error": f"Unexpected failure in [{context}]: {type(exc).__name__}: {exc}"
    }


# Similarity threshold
SIMILARITY_THRESHOLD = 0.4


class DataCollector():

    def __init__(self, ModelFunctionsObj):
        self.object = ModelFunctionsObj

    # ------------------------------------------------------------------ #
    # Fetches a single URL — returns (link, text, error_reason)           #
    # error_reason is None on success, a short string on failure          #
    # ------------------------------------------------------------------ #
    def _fetch_one(self, link: str) -> tuple:
        try:
            html = trafilatura.fetch_url(link)
            if not html:
                return link, None, "no HTML returned (possible bot-block or empty page)"
            text = trafilatura.extract(html)
            if not text:
                return link, None, "HTML fetched but no text could be extracted"
            return link, text, None
        except Exception as e:
            return link, None, f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ #
    # Parallel fetch + batch similarity                                   #
    # ------------------------------------------------------------------ #
    def retriever(self, OriginalContent: str, links: list) -> str:

        # validate inputs 
        if not isinstance(OriginalContent, str) or not OriginalContent.strip():
            return _bad_input("OriginalContent must be a non-empty string.")
        if not isinstance(links, list) or not links:
            return _bad_input("links must be a non-empty list of URL strings.")

        try:
            # Step 1: Parallel fetch
            fetched = {}            # link -> raw text
            fetch_failures = []     # track failures for diagnostics

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(self._fetch_one, link): link for link in links}
                for future in as_completed(futures):
                    link, text, reason = future.result()
                    if text:
                        fetched[link] = text
                    else:
                        fetch_failures.append(f"{link} → {reason}")

            # Log which URLs failed
            if fetch_failures:
                print(f"[DataRetrieval] {len(fetch_failures)}/{len(links)} link(s) failed:")
                for f in fetch_failures:
                    print(f"  ✗ {f}")

            # Zero articles retrieved — no point going further
            if not fetched:
                return _no_content(len(links), len(fetch_failures))

            # ── Step 2: Extract titles ───────────────────────────────────
            valid_links  = []
            valid_titles = []
            valid_texts  = []

            for link, text in fetched.items():
                try:
                    title = text.strip().split(".")[0].lower()
                except (AttributeError, IndexError):
                    title = ""      # empty string still gets scored, just poorly

                valid_links.append(link)
                valid_titles.append(title)
                valid_texts.append(text)

            # ── Step 3: Single batch similarity pass ─────────────────────
            try:
                scores = self.object.BatchSimilarityScores(OriginalContent, valid_titles)
            except Exception as e:
                return _internal_error("BatchSimilarityScores", e)

            # ── Step 4: Filter by threshold ──────────────────────────────
            data = {}
            for link, text, score in zip(valid_links, valid_texts, scores):
                # print(f"[Score] {score:.4f}  {link}") # only for testing dev
                if score >= SIMILARITY_THRESHOLD:
                    try:
                        data[f"searchresult{len(data) + 1}"] = {
                            "organization": tldextract.extract(link).domain,
                            "score": score,
                            "article": text
                        }
                    except Exception as e:
                        print(f"[DataRetrieval] Could not save result for {link}: {e} — skipping.")
                        continue

            # ── Step 5: Return with clear status ─────────────────────────
            if not data:
                return _no_match(
                    fetched=len(fetched),
                    scored=len(valid_titles),
                    threshold=SIMILARITY_THRESHOLD
                )

            return _ok(data)

        except Exception as e:
            return _internal_error("retriever main block", e)

    # ------------------------------------------------------------------ #
    # top_results — handles both old bare-dict and new status-wrapped fmt #
    # ------------------------------------------------------------------ #
    def top_results(self, data, num_of_articals: int = 2):

        try:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    print(f"[top_results] Failed to parse JSON input: {e}")
                    return None

            # Unwrap new response format if present
            if isinstance(data, dict) and "results" in data:
                data = data["results"]

            if not isinstance(data, dict) or not data:
                print("[top_results] Invalid or empty data — nothing to sort.")
                return None

            sorted_items = sorted(
                data.items(),
                key=lambda item: item[1]["score"],
                reverse=True
            )

            num_of_articals = min(num_of_articals, len(sorted_items))
            top_n = sorted_items[:num_of_articals]

            result = {}
            for i, (_, value) in enumerate(top_n, start=1):
                result[f"searchresult{i}"] = value

            return result

        except Exception as e:
            print(f"[top_results] Unexpected error: {e}")
            return None


# ── Standalone helper: fetch and parse a single user-supplied article ─────── #

def get_user_article(user_link: str) -> dict:

    if not isinstance(user_link, str) or not user_link.strip():
        return {"status": "error", "error": "Invalid or empty URL provided."}

    try:
        try:
            html = trafilatura.fetch_url(user_link)
            article_text = trafilatura.extract(html) if html else None
        except Exception as e:
            msg = f"Network or extraction failure: {type(e).__name__}: {e}"
            print(f"[get_user_article] {msg}")
            return {"status": "error", "error": msg}

        if not article_text:
            return {
                "status": "error",
                "error": (
                    "Could not extract readable text from the provided URL. "
                    "The page may be paywalled, JavaScript-rendered, or block scrapers."
                )
            }

        try:
            title = article_text.strip().split(".")[0].lower()
        except (AttributeError, IndexError, TypeError, ValueError):
            title = None

        try:
            organization = tldextract.extract(user_link).domain
        except (AttributeError, TypeError, ValueError):
            organization = None

        return {
            "status": "success",
            "organization": organization,
            "title": title,
            "article": article_text
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected failure in get_user_article: {type(e).__name__}: {e}"
        }
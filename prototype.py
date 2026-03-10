import json
import requests
from querygeneration import KeywordExtractor, KeywordExtractionError
from conflictdetection import ConflictDetector
from dataretrieval import DataCollector, get_user_article
from apisetup import Apicaller
from similarity import ModelFunctions

_SUPPORTED_PROVIDERS  = ("superdev",)
_SUPPORTED_LLM_PROVIDERS = ("anthropic", "openai", "google", "groq")

_FACT_CHECK_SYSTEM_PROMPT = """
You are an AI assistant that explains fact-check results to normal users.
Your job is to convert structured analysis data into a simple explanation that anyone can understand.
Important rules:
1. NEVER mention internal system terms such as: pipeline, retrieval, similarity score, model, API, system status, JSON, or technical errors.
2. If the system could not find reliable information, explain it in simple language like: "We could not find reliable news reports about this claim."
3. Use very simple and clear English. Avoid technical or academic words.
4. Only use the information provided in the input data.
5. Do NOT invent facts or add outside knowledge.
6. Keep the explanation short and easy to read.
7. Maximum length: 100 words.

Output format:
Verdict: <True / False / Partially True / Inconclusive>

Explanation:
Explain the situation in plain language so an average person can understand what was found.

Evidence Summary:
* Briefly mention what different news sources reported.

Notes:
Mention uncertainty, conflicting reports, or lack of reliable information in simple words.
"""


class Prototype:
    def __init__(self, api_key: str, key_provider: str):
        self.key_provider = key_provider
        self.keyword_extractor = KeywordExtractor()
        self.api_caller = Apicaller(api_key)
        self.model_functions = ModelFunctions()
        self.data_collector = DataCollector(self.model_functions)
        self.conflict_detector = ConflictDetector(strictness=0.7)

    def extract_keywords(self, input_type: str, data: str) -> list[str]:
        try:
            return self.keyword_extractor.extract(data)
        except KeywordExtractionError as e:
            print(f"[Pipeline] Keyword extraction failed, using fallback: {e}")
            if input_type == "article_link":
                return [data.split(".")[0]]
            return [data]

    def collect_links(self, keywords: list[str]) -> list[str]:
        try:
            if self.key_provider == "superdev":
                return self.api_caller.superdev(keywords)
            return []
        except Exception as e:
            print(f"[Pipeline] Link collection failed: {e}")
            return []

    def retrieve_online_data(self, article: str, links: list[str]) -> dict:
        raw_data = self.data_collector.retriever(article, links)
        if not isinstance(raw_data, dict) or raw_data.get("status") != "success":
            return raw_data
        top_data = self.data_collector.top_results(raw_data)
        if top_data is None:
            return {
                "status": "error",
                "error": "top_results() returned no data after a successful fetch. Internal scoring failure."
            }
        return {"status": "success", "results": top_data}

    def detect_conflicts(self, original_data: dict | str, collected_data: dict) -> str:
        if isinstance(original_data, dict):
            original_article = original_data["article"]
            organization = original_data.get("organization", "unknown")
        else:
            original_article = original_data
            organization = "unknown"

        results = {}
        for result_name, result in collected_data.items():
            try:
                result["conflict"] = self.conflict_detector.report(
                    original_article, result["article"], organization,
                )
                results[result_name] = result
            except Exception as e:
                print(f"[Pipeline] Conflict detection skipped for '{result_name}': {e}")
        
        return json.dumps(results, indent=4)


    # structured summary for the AI bot

    def final_explanation(self, userTypedquery: str, userinputType: str, raw_aggregated: str, article_text: str = None) -> dict:
        """
        Converts raw pipeline output into a structured dict for the AI bot.

        Preserves every distinct status from dataretrieval.py:
          "success"              → pipeline ran fully, analysis list included
          "no_match"             → articles fetched but none passed similarity threshold
                                   (has "reason" key, not "error")
          "error"                → fetch/scraping/scoring failure
                                   (has "error" key)
          "INSUFFICIENT_CONTENT" → input too short to extract claims
          empty dict {}          → conflict loop ran but all articles threw individually

        The AI bot receives the exact reason string from each case — not a
        generic "error" label — so it can give the user a specific explanation.
        """
        input_key = "user_article" if userinputType == "article_link" else "user_query"

        # For article_link: use the full scraped text if fetch succeeded, otherwise
        # fall back to the URL (fetch failed — no text available).
        input_value = article_text if (userinputType == "article_link" and article_text) else userTypedquery
        final: dict = {
            input_key: input_value,
        }

        # -- Parse raw_aggregated ----------------------------------------
        if isinstance(raw_aggregated, dict):
            data = raw_aggregated

        elif isinstance(raw_aggregated, str):
            try:
                data = json.loads(raw_aggregated)
            except (json.JSONDecodeError, ValueError) as e:
                final["pipeline_status"] = "error"
                final["problem"] = {
                    "status": "error",
                    "error": f"Pipeline produced unparseable output: {e}",
                }
                return final
        else:
            final["pipeline_status"] = "error"
            final["problem"] = {
                "status": "error",
                "error": (
                    f"Unexpected type for raw_aggregated: {type(raw_aggregated).__name__}. "
                    "Expected a JSON string or dict."
                ),
            }
            return final

        if not isinstance(data, dict):
            final["pipeline_status"] = "error"
            final["problem"] = {
                "status": "error",
                "error": "Parsed pipeline output is not a dict — cannot interpret results.",
            }
            return final

        # -- Identify output shape ----------------------------------------
        # detect_conflicts() success → keys like "searchresult1", "searchresult2"
        # dataretrieval / pipeline error → top-level "status" key
        # Empty dict                     → conflict loop ran, all articles skipped
        has_search_results  = any(k.startswith("searchresult") for k in data)
        top_level_status    = data.get("status")  # None if not present

        if has_search_results and top_level_status is None:
            # ── FULL SUCCESS ─────────────────────────────────────────────
            final["pipeline_status"] = "success"
            final["retrieved_articles"] = len(data)
            final["analysis"] = list(data.values())

        elif top_level_status == "no_match":
            # ── NO_MATCH from _no_match() ────────────────────────────────
            # dataretrieval fetched articles and scored them, but none crossed
            # the similarity threshold. pipeline_status is "error" because the
            # pipeline could not produce an analysis — "reason" key inside
            # problem preserves the specific cause for the AI bot.
            final["pipeline_status"] = "error"
            final["problem"] = data   # preserves "status" and "reason" as-is

        elif top_level_status == "error":
            # ── HARD ERROR from _no_content / _bad_input / _internal_error ──
            # or from any pipeline stage (provider check, fetch fail, etc.)
            # Carries an "error" key with the specific reason string.
            final["pipeline_status"] = "error"
            final["problem"] = data   # preserves "status" and "error" as-is

        elif top_level_status == "INSUFFICIENT_CONTENT":
            # ── SHORT INPUT from conflictdetection.report() ──────────────
            # Input text could not be broken into claims of >= 6 words.
            # Distinct from an error — the pipeline ran fine, input was just too short.
            final["pipeline_status"] = "insufficient_input"
            final["problem"] = data

        elif not data:
            # ── EMPTY DICT ───────────────────────────────────────────────
            # detect_conflicts() ran but every article was individually skipped.
            final["pipeline_status"] = "error"
            final["problem"] = {
                "status": "error",
                "error": (
                    "Conflict detection completed but produced no results. "
                    "Every retrieved article was individually skipped — "
                    "possible causes: model errors, empty article text, or encoding failures."
                ),
            }

        else:
            # ── UNKNOWN SHAPE ────────────────────────────────────────────
            # Pipeline produced a dict we don't recognise. Pass it through
            # rather than silently discarding it — the AI can still read it.
            final["pipeline_status"] = "error"
            final["problem"] = {
                "status": "error",
                "error": (
                    f"Pipeline returned an unrecognised output shape "
                    f"(top-level keys: {list(data.keys())}). Cannot classify result."
                ),
            }
            
        return final

    
    # Send the structured summary to any supported LLM
    
    def get_response(
        self,
        final_aggregatedStructure: dict,
        user_any_llm_api_key: str,
        llm_provider: str = "anthropic",
    ) -> str:
        # -- Input validation --------------------------------------------
        if not isinstance(user_any_llm_api_key, str) or not user_any_llm_api_key.strip():
            return json.dumps({
                "status": "error",
                "error": "LLM API key is missing or empty.",
            }, indent=4)

        llm_provider = llm_provider.strip().lower()
        if llm_provider not in _SUPPORTED_LLM_PROVIDERS:
            return json.dumps({
                "status": "error",
                "error": (
                    f"Unsupported LLM provider: '{llm_provider}'. "
                    f"Supported providers are: {', '.join(_SUPPORTED_LLM_PROVIDERS)}."
                ),
            }, indent=4)

        user_message_text = (
            "Here is the fact-checking pipeline output. "
            "Generate the fact-checking report according to your instructions.\n\n"
            f"{json.dumps(final_aggregatedStructure, indent=2)}"
        )

        # -- Build provider-specific request -----------------------------
        try:
            if llm_provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key":         user_any_llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                }
                body = {
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system":     _FACT_CHECK_SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": user_message_text}],
                }

            elif llm_provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {user_any_llm_api_key}",
                    "content-type":  "application/json",
                }
                body = {
                    "model":    "gpt-4o",
                    "messages": [
                        {"role": "system",  "content": _FACT_CHECK_SYSTEM_PROMPT},
                        {"role": "user",    "content": user_message_text},
                    ],
                }

            elif llm_provider == "google":
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.0-flash-lite:generateContent?key={user_any_llm_api_key}"
                )
                headers = {"content-type": "application/json"}
                body = {
                    "systemInstruction": {
                        "parts": [{"text": _FACT_CHECK_SYSTEM_PROMPT}]
                    },
                    "contents": [
                        {"parts": [{"text": user_message_text}]}
                    ],
                }

            elif llm_provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {user_any_llm_api_key}",
                    "content-type":  "application/json",
                }
                body = {
                    "model":    "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": _FACT_CHECK_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message_text},
                    ],
                }

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to build request for provider '{llm_provider}': {e}",
            }, indent=4)

        # -- Make the HTTP call ------------------------------------------
        try:
            response = requests.post(url, headers=headers, json=body, timeout=30)

        except requests.exceptions.Timeout:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Request timed out after 30 seconds.",
            }, indent=4)

        except requests.exceptions.ConnectionError as e:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Could not connect. Check your network: {e}",
            }, indent=4)

        except requests.exceptions.RequestException as e:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Request failed: {type(e).__name__}: {e}",
            }, indent=4)

        # -- Handle non-200 HTTP responses --------------------------------
        if response.status_code == 401:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] API key is invalid or unauthorized (HTTP 401).",
            }, indent=4)

        if response.status_code == 429:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Rate limit exceeded (HTTP 429). Wait and retry.",
            }, indent=4)

        if response.status_code != 200:
            body_preview = response.text[:300] if response.text else "(empty body)"
            return json.dumps({
                "status": "error",
                "error": (
                    f"[{llm_provider}] HTTP {response.status_code}. "
                    f"Body preview: {body_preview}"
                ),
            }, indent=4)

        # -- Parse the response body -------------------------------------
        try:
            response_data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Returned a non-JSON body: {e}",
            }, indent=4)

        # -- Extract text using provider-specific response shape ---------
        try:
            if llm_provider == "anthropic":
                # {"content": [{"type": "text", "text": "..."}]}
                content_blocks = response_data.get("content", [])
                explanation = "\n".join(
                    block["text"]
                    for block in content_blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()

            elif llm_provider == "openai":
                # {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
                choices = response_data.get("choices", [])
                if not choices:
                    return json.dumps({
                        "status": "error",
                        "error": "[openai] Response contained no choices.",
                    }, indent=4)
                explanation = choices[0].get("message", {}).get("content", "").strip()

            elif llm_provider == "google":
                # {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
                candidates = response_data.get("candidates", [])
                if not candidates:
                    return json.dumps({
                        "status": "error",
                        "error": "[google] Response contained no candidates.",
                    }, indent=4)
                parts = candidates[0].get("content", {}).get("parts", [])
                explanation = "\n".join(
                    p["text"] for p in parts if isinstance(p, dict) and "text" in p
                ).strip()
    
            elif llm_provider == "groq":
                # {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
                choices = response_data.get("choices", [])
                if not choices:
                    return json.dumps({
                        "status": "error",
                        "error": "[groq] Response contained no choices.",
                    }, indent=4)
                explanation = choices[0].get("message", {}).get("content", "").strip()

        except (KeyError, IndexError, TypeError) as e:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Unexpected response structure: {e}",
            }, indent=4)

        if not explanation:
            return json.dumps({
                "status": "error",
                "error": f"[{llm_provider}] Response contained no text.",
            }, indent=4)

        return explanation

    
    
    # Full pipeline — single entry point
    
    def run(
        self,
        user_input: str,
        input_type: str,
        llm_api_key: str = None,
        llm_provider: str = "anthropic",
    ) -> str:
        raw_aggregated: str | None = None
        article_data      = None
        collected_data    = None
        text_for_keywords = None

        # -- Step 1: Validate key_provider --------------------------------
        if self.key_provider not in _SUPPORTED_PROVIDERS:
            raw_aggregated = json.dumps({
                "status": "error",
                "error": (
                    f"Unsupported key provider: '{self.key_provider}'. "
                    f"Supported providers are: {', '.join(_SUPPORTED_PROVIDERS)}."
                ),
            }, indent=4)

        # -- Step 2: Fetch article / validate query -----------------------
        if raw_aggregated is None:
            if input_type == "article_link":
                article_data = get_user_article(user_input)
                print(f"[Pipeline] Article fetch status: {article_data.get('status')}")
                if article_data.get("status") != "success":
                    raw_aggregated = json.dumps(article_data, indent=4)
                else:
                    text_for_keywords = article_data["article"]
            else:
                claims_preview = self.conflict_detector.split_into_claims(user_input)
                if not claims_preview:
                    raw_aggregated = json.dumps({
                        "status": "error",
                        "error": (
                            "Input text is too short for conflict detection. "
                            "Provide at least one complete sentence (6+ words) so the "
                            "system can extract claims to compare against retrieved articles."
                        ),
                    }, indent=4)
                else:
                    article_data      = user_input
                    text_for_keywords = user_input

        # -- Step 3: Extract keywords -------------------------------------
        if raw_aggregated is None:
            keywords = self.extract_keywords(input_type, text_for_keywords)
            print(f"[Pipeline] Keywords: {keywords}")

        # -- Step 4: Collect links ----------------------------------------
        if raw_aggregated is None:
            links = self.collect_links(keywords)
            print(f"[Pipeline] Collected {len(links)} links")
            if not links:
                raw_aggregated = json.dumps({
                    "status": "error",
                    "error": (
                        "No search result links were collected. "
                        "Possible causes: API key is invalid or rate-limited, "
                        "all extracted keywords were too short, or a network failure occurred."
                    ),
                }, indent=4)

        # -- Step 5: Retrieve and rank articles ---------------------------
        if raw_aggregated is None:
            retrieval_result = self.retrieve_online_data(text_for_keywords, links)
            print(f"[Pipeline] Retrieval status: {retrieval_result.get('status')}")
            if retrieval_result.get("status") != "success":
                raw_aggregated = json.dumps(retrieval_result, indent=4)
            else:
                collected_data = retrieval_result["results"]
                print(f"[Pipeline] Retrieved {len(collected_data)} results")

        # -- Step 6: Conflict detection -----------------------------------
        if raw_aggregated is None:
            raw_aggregated = self.detect_conflicts(article_data, collected_data)

        # -- Step 7: Build AI-readable summary ----------------------------

        fetched_article_text = (
            article_data.get("article")
            if isinstance(article_data, dict) and input_type == "article_link"
            else None
        )
        final_structure = self.final_explanation(user_input, input_type, raw_aggregated, fetched_article_text)

        # -- Step 8: Generate AI explanation (optional) -------------------
        if llm_api_key:
            return self.get_response(final_structure, llm_api_key, llm_provider)

        return json.dumps(final_structure, indent=4)

'''
Collecting all the links through api
'''
import requests, json
from langdetect import detect, LangDetectException
# super dev 

class Apicaller():
    def __init__(self, key):
        self.api = key
    
    def superdev(self, keywords: list[str]):

        if not isinstance(keywords, list):
            raise ValueError("Invalid data type")
        
        attempts = 0
        links = []

        for keyword in keywords:
            
            if len(keyword) < 3:
                continue

            url = "https://google.serper.dev/news"

            payload = {
            "q": keyword,
            "gl": "in",
            "hl": "en"
            }

            headers = {
            'X-API-KEY': self.api,
            'Content-Type': 'application/json'
            }

            # FIX 1: requests.request() itself can throw RequestException (timeout,
            # DNS failure, no network) before .status_code is ever accessible.
            # Catch it, log it, count it as a failed attempt, and move on.
            try:
                response = requests.request("POST", url, headers=headers, json=payload)
            except requests.exceptions.RequestException as e:
                print(f"[Apicaller] Network error for keyword '{keyword}': {e}")
                attempts += 1
                if attempts >= len(keywords) / 2:
                    print("Too many network failures, exiting...")
                    break
                continue

            if response.status_code != 200:
                # Wrong api key or something else
                print(f"Request failed with status code: {response.status_code}")
                attempts += 1

                if attempts >= len(keywords) / 2:
                    print("Wrrong api key or somthing else exiting...")
                    break

                continue

            # FIX 2: response_json["news"] raises KeyError when the API returns
            # a different top-level structure (e.g. an error payload with no "news" key).
            # Skip the keyword and continue rather than crashing the whole call.
            try:
                response_json = response.json()
                news_items = response_json["news"]
            except (KeyError, ValueError) as e:
                print(f"[Apicaller] Unexpected response structure for keyword '{keyword}': {e}")
                continue

            for result in news_items:

                # FIX 3: result["title"] and result["link"] raise KeyError when a
                # news entry is missing expected fields. Skip the malformed entry.
                try:
                    title = result["title"]
                    link  = result["link"]
                except KeyError as e:
                    print(f"[Apicaller] Missing field {e} in news entry — skipping entry.")
                    continue

                # FIX 4: langdetect.detect() raises LangDetectException on very short,
                # numeric, or character-ambiguous titles. Treat detection failure as
                # non-English so the link is simply not added (safe, conservative default).
                try:
                    if detect(title) == "en":
                        links.append(link)
                except LangDetectException as e:
                    print(f"[Apicaller] Language detection failed for title '{title[:40]}': {e} — skipping.")
                    continue
        
        return links
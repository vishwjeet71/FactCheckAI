from keybert import KeyBERT
from transformers import logging

logging.set_verbosity_error()


class KeywordExtractionError(Exception):
    """Raised when keyword extraction fails and no fallback is possible."""
    pass


class KeywordExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.kw_model = KeyBERT(model=model_name)
        except Exception as e:
            raise KeywordExtractionError(
                f"Failed to load KeyBERT model '{model_name}': {e}"
            ) from e

    def extract(
        self,
        text: str,
        num_keywords: int = 3,
        ngram_range: tuple = (1, 2),
    ) -> list[str]:
        """
        Extract keywords from text.
        Returns a list of keyword strings.
        Raises KeywordExtractionError if extraction fails completely.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                top_n=num_keywords,
            )

            # extract_keywords returns list of (keyword, score) tuples
            result = [kw[0] for kw in keywords if kw]

            if not result:
                raise KeywordExtractionError("Model returned no keywords for the given text.")

            return result

        except KeywordExtractionError:
            raise  # let it bubble up cleanly

        except Exception as e:
            raise KeywordExtractionError(f"Unexpected error during keyword extraction: {e}") from e

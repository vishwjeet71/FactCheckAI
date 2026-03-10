import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import logging

logging.set_verbosity_error()


class SimilarityModelError(Exception):
    """Raised when the similarity model fails to load or run."""
    pass


class ModelFunctions:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        try:
            self.sbert_model = SentenceTransformer(model_name)
        except Exception as e:
            raise SimilarityModelError(
                f"Failed to load SentenceTransformer model '{model_name}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Single pair comparison
    # ------------------------------------------------------------------
    def SimilarityScore(self, sent1: str, sent2: str) -> float:
        """
        Returns cosine similarity between two sentences (0.0 to 1.0).
        Returns 0.0 on failure instead of crashing — safe to use in loops.
        """
        if not isinstance(sent1, str) or not isinstance(sent2, str):
            print("[ModelFunctions] SimilarityScore: both inputs must be strings.")
            return 0.0

        if not sent1.strip() or not sent2.strip():
            print("[ModelFunctions] SimilarityScore: received empty string input.")
            return 0.0

        try:
            embeddings = self.sbert_model.encode(
                [sent1, sent2],
                convert_to_tensor=True,
                batch_size=32,
            )
            emb1, emb2 = embeddings
            score = F.cosine_similarity(emb1, emb2, dim=0)
            return round(score.item(), 4)

        except Exception as e:
            print(f"[ModelFunctions] SimilarityScore failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Batch comparison — one forward pass for all candidates
    # ------------------------------------------------------------------
    def BatchSimilarityScores(self, original: str, candidates: list[str]) -> list[float]:
        """
        Compares `original` against every string in `candidates` in a
        single encoding pass. Returns a list of scores in the same order.

        Why this is faster than calling SimilarityScore() N times:
        SentenceTransformer.encode() has per-call overhead (tokenization,
        GPU dispatch). One call with N+1 sentences amortises that cost once.
        """
        if not isinstance(original, str) or not original.strip():
            print("[ModelFunctions] BatchSimilarityScores: original must be a non-empty string.")
            return [0.0] * len(candidates)

        if not candidates:
            return []

        # Filter out any non-string or empty entries, track their positions
        valid_candidates = []
        index_map = []  # maps valid index → original index

        for i, candidate in enumerate(candidates):
            if isinstance(candidate, str) and candidate.strip():
                valid_candidates.append(candidate)
                index_map.append(i)
            else:
                print(f"[ModelFunctions] Skipping invalid candidate at index {i}.")

        if not valid_candidates:
            print("[ModelFunctions] No valid candidates to compare against.")
            return [0.0] * len(candidates)

        try:
            all_texts = [original] + valid_candidates

            embeddings = self.sbert_model.encode(
                all_texts,
                convert_to_tensor=True,
                batch_size=64,
                show_progress_bar=False,
            )

            orig_emb = embeddings[0]       # shape: (hidden,)
            cand_embs = embeddings[1:]     # shape: (N, hidden)

            scores = F.cosine_similarity(
                orig_emb.unsqueeze(0).expand_as(cand_embs),
                cand_embs,
                dim=1,
            )

            # Rebuild full result list with 0.0 for skipped entries
            result = [0.0] * len(candidates)
            for valid_idx, original_idx in enumerate(index_map):
                result[original_idx] = round(scores[valid_idx].item(), 4)

            return result

        except Exception as e:
            print(f"[ModelFunctions] BatchSimilarityScores failed: {e}")
            return [0.0] * len(candidates)

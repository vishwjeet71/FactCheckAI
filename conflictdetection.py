import re
import json
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from dataclasses import dataclass
from typing import List, Tuple

logging.set_verbosity_error()


# Spacy model — loaded once at module level with proper error handling

_SPACY_MODEL_NAME = "en_core_web_trf"

def _load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)

    except OSError:
        print(f"[ConflictDetector] spaCy model '{model_name}' not found. Downloading...")
        try:
            spacy.cli.download(model_name)
            print(f"[ConflictDetector] Model '{model_name}' downloaded successfully.")
            return spacy.load(model_name)
        except Exception as e:
            raise RuntimeError(f"[ConflictDetector] spaCy model download failed: {e}") from e

    except Exception as e:
        raise RuntimeError(f"[ConflictDetector] spaCy model loading failed: {e}") from e


try:
    _spacy_model = _load_spacy_model(_SPACY_MODEL_NAME)
except RuntimeError as e:
    print(f"[ConflictDetector] WARNING: spaCy model could not be loaded: {e}")
    print("[ConflictDetector] NER-based conflict classification will fall back to 'Factual Conflict'.")
    _spacy_model = None


# Data class

@dataclass
class Conflict:
    sentence_a: str
    sentence_b: str
    conflict_type: str
    severity: str
    confidence: float
    contradiction_score: float


# ConflictDetector

class ConflictDetector:
    def __init__(self, strictness: float = 0.7):
        if not (0.0 <= strictness <= 1.0):
            raise ValueError(f"strictness must be between 0.0 and 1.0, got {strictness}")

        self.strictness = strictness

        print("[ConflictDetector] Loading semantic similarity model...")
        try:
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(f"[ConflictDetector] Failed to load similarity model: {e}") from e

        print("[ConflictDetector] Loading NLI contradiction detection model...")
        try:
            _nli_model_name = "cross-encoder/nli-deberta-v3-base"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(_nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(_nli_model_name)
            self.nli_model.eval()
        except Exception as e:
            raise RuntimeError(f"[ConflictDetector] Failed to load NLI model: {e}") from e

        print("[ConflictDetector] Loading NER model...")
        self.nlp = _spacy_model

        self.ignore_patterns = [
            r"\b(published|updated|posted|written by|author|reporter|editor)\b",
            r"\b\d{1,2}:\d{2}\s?(am|pm|AM|PM)\b",
            r"\bfollow us\b|\bsubscribe\b|\bclick here\b",
            r"\bcopyright\b|\ball rights reserved\b",
        ]

        print("[ConflictDetector] All models loaded.\n")

    def split_into_claims(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 6:
                continue
            if any(re.search(p, sent, re.IGNORECASE) for p in self.ignore_patterns):
                continue
            claims.append(sent)

        return claims

    def find_similar_pairs(self, claims_a, claims_b):
        if not claims_a or not claims_b:
            return []

        similarity_threshold = 0.75 - (self.strictness * 0.25)

        try:
            embeddings_a = self.similarity_model.encode(claims_a, batch_size=24, convert_to_tensor=True)
            embeddings_b = self.similarity_model.encode(claims_b, batch_size=24, convert_to_tensor=True)
        except Exception as e:
            print(f"[ConflictDetector] Encoding failed during similarity search: {e}")
            return []

        cosine_scores = util.cos_sim(embeddings_a, embeddings_b)

        pairs = []
        for i in range(len(claims_a)):
            for j in range(len(claims_b)):
                score = cosine_scores[i][j].item()
                if score >= similarity_threshold:
                    pairs.append((claims_a[i], claims_b[j], score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def check_contradiction(self, sentence_a: str, sentence_b: str) -> float:
        try:
            inputs = self.nli_tokenizer(
                sentence_a, sentence_b,
                return_tensors="pt", truncation=True, max_length=512,
            )
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0][0].item()
        except Exception as e:
            print(f"[ConflictDetector] NLI check failed for pair: {e}")
            return 0.0

    def classify_conflict_type(self, sentence_a: str, sentence_b: str) -> str:
        try:
            doc_a = self.nlp(sentence_a)
            doc_b = self.nlp(sentence_b)
        except Exception as e:
            print(f"[ConflictDetector] NER classification failed: {e}")
            return "Factual Conflict"

        entities_a = {ent.label_: ent.text for ent in doc_a.ents}
        entities_b = {ent.label_: ent.text for ent in doc_b.ents}

        entity_type_map = {
            "PERSON":   "Name Mismatch",
            "ORG":      "Organization Mismatch",
            "GPE":      "Location Mismatch",
            "LOC":      "Location Mismatch",
            "DATE":     "Date Mismatch",
            "TIME":     "Time Mismatch",
            "CARDINAL": "Number Mismatch",
            "ORDINAL":  "Order/Rank Mismatch",
            "MONEY":    "Financial Mismatch",
            "PERCENT":  "Statistics Mismatch",
            "EVENT":    "Event Mismatch",
        }

        conflicts_found = []
        for entity_label, conflict_name in entity_type_map.items():
            val_a = entities_a.get(entity_label)
            val_b = entities_b.get(entity_label)
            if val_a and val_b and val_a.lower() != val_b.lower():
                conflicts_found.append(conflict_name)

        return " & ".join(set(conflicts_found)) if conflicts_found else "Factual Conflict"

    def get_severity(self, contradiction_score: float, conflict_type: str) -> str:
        high_priority_types = [
            "Date Mismatch", "Location Mismatch", "Number Mismatch",
            "Event Mismatch", "Factual Conflict",
        ]
        is_high_priority = any(t in conflict_type for t in high_priority_types)

        if contradiction_score >= 0.85:
            return "HIGH"
        elif contradiction_score >= 0.65:
            return "HIGH" if is_high_priority else "MEDIUM"
        else:
            return "MEDIUM" if is_high_priority else "LOW"

    def detect_conflicts(self, doc_a: str, doc_b: str) -> List[Conflict]:
        contradiction_threshold = 0.85 - (self.strictness * 0.35)
        print(f"[ConflictDetector] Strictness: {self.strictness} | Contradiction threshold: {contradiction_threshold:.2f}")

        claims_a = self.split_into_claims(doc_a)
        claims_b = self.split_into_claims(doc_b)
        print(f"[ConflictDetector] Doc A: {len(claims_a)} claims | Doc B: {len(claims_b)} claims")

        if not claims_a or not claims_b:
            print("[ConflictDetector] One or both documents produced no claims. Skipping.")
            return []

        similar_pairs = self.find_similar_pairs(claims_a, claims_b)
        print(f"[ConflictDetector] Similar pairs found: {len(similar_pairs)}")

        conflicts = []
        seen_pairs: set = set()

        for sent_a, sent_b, sim_score in similar_pairs:
            pair_key = (sent_a[:50], sent_b[:50])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            contradiction_score = self.check_contradiction(sent_a, sent_b)

            if contradiction_score >= contradiction_threshold:
                conflict_type = self.classify_conflict_type(sent_a, sent_b)
                severity = self.get_severity(contradiction_score, conflict_type)

                conflicts.append(Conflict(
                    sentence_a=sent_a,
                    sentence_b=sent_b,
                    conflict_type=conflict_type,
                    severity=severity,
                    confidence=round(sim_score, 3),
                    contradiction_score=round(contradiction_score, 3),
                ))

        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        conflicts.sort(key=lambda x: (severity_order[x.severity], -x.contradiction_score))

        return conflicts

    def report(self, doc_a: str, doc_b: str, external_source: str = "unknown") -> dict:
        """
        Runs conflict detection and returns a structured dict.
        Always returns a dict — .
        """
        # BUG FIX: Previously, when doc_a had no extractable claims (input too
        # short, or all sentences under 6 words), detect_conflicts() returned []
        # and report() returned {"status": "NO_CONFLICTS"}. That is a false result —
        # the pipeline had no basis to say "no conflicts"; it simply couldn't read
        # the input. The AI bot receiving NO_CONFLICTS would tell the user the
        # article is consistent, which is a wrong conclusion from an empty analysis.
        # Now we detect this before running the full pipeline and return a distinct
        # INSUFFICIENT_CONTENT status that accurately describes what happened.
        claims_a = self.split_into_claims(doc_a)
        if not claims_a:
            return {
                "status": "INSUFFICIENT_CONTENT",
                "error": (
                    "The input text could not be broken into verifiable claims. "
                    "It may be too short (under 6 words per sentence) or contain "
                    "only boilerplate/metadata. Provide a paragraph or more of "
                    "substantive text for meaningful conflict analysis."
                ),
                "total": 0,
                "conflicts": {},
            }

        try:
            conflicts = self.detect_conflicts(doc_a, doc_b)
        except Exception as e:
            print(f"[ConflictDetector] detect_conflicts raised unexpectedly: {e}")
            return {
                "status": "ERROR",
                "error": f"Detection pipeline failed: {type(e).__name__}: {e}",
                "total": 0,
                "conflicts": {},
            }

        if not conflicts:
            return {"status": "NO_CONFLICTS", "total": 0, "conflicts": {}}

        high   = [c for c in conflicts if c.severity == "HIGH"]
        medium = [c for c in conflicts if c.severity == "MEDIUM"]
        low    = [c for c in conflicts if c.severity == "LOW"]

        if len(high) >= 3:
            verdict = "BIG_MISMATCH"
        elif len(high) >= 1:
            verdict = "MISMATCH_DETECTED"
        elif len(medium) >= 2:
            verdict = "MINOR_MISMATCH"
        else:
            verdict = "MOSTLY_CONSISTENT"

        return {
            "status": verdict,
            "total": len(conflicts),
            "high": len(high),
            "medium": len(medium),
            "low": len(low),
            "conflicts": {
                f"conflict_{i}": {
                    "conflict_type":       conflict.conflict_type,
                    "severity":            conflict.severity.lower(),
                    "contradiction_score": conflict.contradiction_score,
                    "similarity_score":    conflict.confidence,
                    "user_claim":          conflict.sentence_a,
                    external_source:       conflict.sentence_b,
                }
                for i, conflict in enumerate(conflicts, 1)
            },
        }
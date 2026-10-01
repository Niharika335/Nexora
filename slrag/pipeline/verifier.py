"""Fail-closed verifier running 5 distinct checks (lexical, semantic, coreference, hallucinated ID, consistency)."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from slrag.config import VerifierConfig, DEFAULT_CONFIG
from slrag.contracts.events import (
    CheckResult,
    Claim,
    ClaimStatus,
    ClaimVerificationEvent,
)
from slrag.corpus.models import Chunk
from slrag.nlp.lemmas import content_lemmas, has_negation, numbers, split_sentences
from slrag.retrieval.dense import generate_bge_small_embedding
from slrag.telemetry.bus import TelemetryBus, GLOBAL_BUS

logger = logging.getLogger(__name__)


class FailClosedVerifier:
    """Fail-closed verification engine enforcing 5 rigorous grounding checks."""

    def __init__(self, config: VerifierConfig = DEFAULT_CONFIG.verifier):
        self.config = config

    def check_lexical(self, claim_text: str, cited_text: str) -> CheckResult:
        """1. Lexical Check: Checks token/keyword overlap against cited chunk."""
        stop_words = {
            "a", "an", "the", "in", "on", "of", "to", "for", "is", "are", "was", "were",
            "and", "or", "it", "this", "that", "with", "by", "at", "from", "as"
        }
        c_words = [w.lower() for w in re.findall(r"\b\w+\b", claim_text) if w.lower() not in stop_words]
        if not c_words:
            return CheckResult(name="lexical", passed=True, score=1.0, details="No lexical content words to check.")

        doc_words = set(w.lower() for w in re.findall(r"\b\w+\b", cited_text))
        matched = sum(1 for w in c_words if w in doc_words)
        overlap = matched / len(c_words)

        passed = overlap >= self.config.lexical
        return CheckResult(
            name="lexical",
            passed=passed,
            score=round(overlap, 3),
            details=f"Lexical overlap={overlap:.2f} (threshold={self.config.lexical})",
        )

    def check_semantic(self, claim_text: str, cited_text: str) -> CheckResult:
        """2. Semantic Check: cosine similarity between the claim and the cited text, taking the better of the
        whole cited text and its best-matching sentence (a short claim drawn from one sentence of a long
        chunk is otherwise diluted by the rest of the chunk)."""
        c_vec = generate_bge_small_embedding(claim_text)
        passages = [cited_text] + [s for s in split_sentences(cited_text) if s.strip() and s.strip() != cited_text.strip()]
        sim = max(float(np.dot(c_vec, generate_bge_small_embedding(p))) for p in passages)
        score = (sim + 1.0) / 2.0  # Normalize to [0, 1]

        passed = score >= self.config.semantic
        return CheckResult(
            name="semantic",
            passed=passed,
            score=round(score, 3),
            details=f"Semantic similarity={score:.2f} (threshold={self.config.semantic})",
        )

    def check_coreference(self, claim_text: str, cited_text: str) -> CheckResult:
        """3. Coreference Check: Verifies pronouns have grounded entity antecedents in context."""
        unresolved_pronouns = {"he", "she", "they", "it", "this", "these", "those"}
        words = [w.lower() for w in re.findall(r"\b\w+\b", claim_text)]

        if words and words[0] in unresolved_pronouns:
            # First word is a free pronoun without immediate subject in the claim
            # Check if cited text contains strong entity nouns
            entities = re.findall(r"\b[A-Z][a-zA-Z]+\b", cited_text)
            if not entities:
                return CheckResult(
                    name="coreference",
                    passed=False,
                    score=0.0,
                    details=f"Unresolved pronoun '{words[0]}' without entity antecedent in cited text.",
                )

        return CheckResult(
            name="coreference",
            passed=True,
            score=1.0,
            details="Coreference check passed: entities grounded.",
        )

    def check_hallucinated_id(self, cited_ids: List[str], allowed_ids: Set[str]) -> CheckResult:
        """4. Hallucinated ID Check: Verifies cited Doc IDs exist in the retrieved pool."""
        if not cited_ids:
            return CheckResult(
                name="hallucinated_id",
                passed=False,
                score=0.0,
                details="No citation provided; claims must cite valid Doc IDs.",
            )

        invalid_ids = [doc_id for doc_id in cited_ids if doc_id not in allowed_ids]
        if invalid_ids:
            return CheckResult(
                name="hallucinated_id",
                passed=False,
                score=0.0,
                details=f"Hallucinated Doc IDs detected: {invalid_ids}",
            )

        return CheckResult(
            name="hallucinated_id",
            passed=True,
            score=1.0,
            details="All cited Doc IDs belong to retrieved candidate set.",
        )

    def check_consistency(
        self,
        claim_text: str,
        previously_verified: List[Claim],
        cited_text: str,
    ) -> CheckResult:
        """5. Consistency Check: non-contradiction with the cited evidence and with the ledger.

        Fails if (a) the claim's negation polarity differs from the cited sentence it is closest
        to, (b) the claim states a number absent from the cited text, or (c) it restates a
        previously verified claim's proposition with flipped negation or different numbers.
        """
        def fail(details: str) -> CheckResult:
            return CheckResult(name="consistency", passed=False, score=0.0, details=details)

        claim_lemmas = content_lemmas(claim_text)

        # (a) Negation parity against the most lexically similar cited sentence
        sentences = split_sentences(cited_text)
        best_sentence = max(sentences, key=lambda s: len(claim_lemmas & content_lemmas(s)))
        if has_negation(claim_text) != has_negation(best_sentence):
            return fail(f"Negation polarity differs from cited sentence: '{best_sentence}'")

        # (b) Every number in the claim must appear in the cited text
        missing_numbers = numbers(claim_text) - numbers(cited_text)
        if missing_numbers:
            return fail(f"Numbers not present in cited text: {sorted(missing_numbers)}")

        # (c) Contradiction with previously verified claims about the same proposition
        claim_words = claim_lemmas - numbers(claim_text)
        for prev in previously_verified:
            prev_words = content_lemmas(prev.text) - numbers(prev.text)
            union = claim_words | prev_words
            if not union or len(claim_words & prev_words) / len(union) < 0.6:
                continue
            if has_negation(claim_text) != has_negation(prev.text):
                return fail(f"Claim negates previously verified claim '{prev.claim_id}': '{prev.text}'")
            claim_nums, prev_nums = numbers(claim_text), numbers(prev.text)
            if claim_nums and prev_nums and claim_nums != prev_nums:
                return fail(f"Claim changes the numbers of previously verified claim '{prev.claim_id}': '{prev.text}'")

        return CheckResult(
            name="consistency",
            passed=True,
            score=1.0,
            details="Claim is consistent with cited evidence and prior verified claims.",
        )

    def verify_claim(
        self,
        claim: Claim,
        retrieved_chunks_map: Dict[str, Chunk],
        allowed_doc_ids: Set[str],
        previously_verified: List[Claim],
    ) -> Tuple[bool, Dict[str, CheckResult], Optional[str]]:
        """Run all 5 checks. Fail-closed: returns False if ANY check fails."""
        checks: Dict[str, CheckResult] = {}

        # 1. Hallucinated ID check
        h_check = self.check_hallucinated_id(claim.doc_ids, allowed_doc_ids)
        checks["hallucinated_id"] = h_check
        if not h_check.passed:
            return False, checks, f"Failed hallucinated_id check: {h_check.details}"

        # Combine text of all cited chunks
        cited_chunks_text = " ".join(
            retrieved_chunks_map[doc_id].text
            for doc_id in claim.doc_ids
            if doc_id in retrieved_chunks_map
        )

        if not cited_chunks_text:
            return False, checks, "Cited document chunk text not found in retrieved pool."

        # 2. Lexical check
        l_check = self.check_lexical(claim.text, cited_chunks_text)
        checks["lexical"] = l_check
        if not l_check.passed:
            return False, checks, f"Failed lexical check: {l_check.details}"

        # 3. Semantic check
        s_check = self.check_semantic(claim.text, cited_chunks_text)
        checks["semantic"] = s_check
        if not s_check.passed:
            return False, checks, f"Failed semantic check: {s_check.details}"

        # 4. Coreference check
        c_check = self.check_coreference(claim.text, cited_chunks_text)
        checks["coreference"] = c_check
        if not c_check.passed:
            return False, checks, f"Failed coreference check: {c_check.details}"

        # 5. Consistency check
        con_check = self.check_consistency(claim.text, previously_verified, cited_chunks_text)
        checks["consistency"] = con_check
        if not con_check.passed:
            return False, checks, f"Failed consistency check: {con_check.details}"

        return True, checks, None

    async def verify_and_emit(
        self,
        claim: Claim,
        retrieved_chunks_map: Dict[str, Chunk],
        allowed_doc_ids: Set[str],
        previously_verified: List[Claim],
        turn_id: str,
        session_id: str = "default_session",
        seq: int = 0,
        bus: Optional[TelemetryBus] = None,
    ) -> Tuple[Claim, bool]:
        """Run verification and emit ClaimVerificationEvent to telemetry bus."""
        telemetry_bus = bus or GLOBAL_BUS
        passed, checks, rejection_reason = self.verify_claim(
            claim=claim,
            retrieved_chunks_map=retrieved_chunks_map,
            allowed_doc_ids=allowed_doc_ids,
            previously_verified=previously_verified,
        )

        status = ClaimStatus.VERIFIED if passed else ClaimStatus.REJECTED
        updated_claim = claim.model_copy(update={"verification_status": status, "turn_id": turn_id})

        event = ClaimVerificationEvent(
            session_id=session_id,
            turn_id=turn_id,
            seq=seq,
            claim_id=claim.claim_id,
            claim_text=claim.text,
            doc_ids=claim.doc_ids,
            checks=checks,
            passed=passed,
            rejection_reason=rejection_reason,
        )
        await telemetry_bus.emit(event)

        return updated_claim, passed

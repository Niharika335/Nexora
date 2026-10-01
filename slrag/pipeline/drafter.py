"""Claim drafter enforcing enum-constrained citation formats."""

import re
from typing import List, Set, Tuple
from slrag.contracts.events import Claim, ClaimStatus


class ClaimDrafter:
    """Drafts atomic claims and enforces enum constraints on cited document identifiers."""

    CITATION_REGEX = re.compile(r"\[([\w\-]+(?:§[\w\-]+)?)\]")

    def __init__(self, allowed_doc_ids: Set[str]):
        self.allowed_doc_ids = allowed_doc_ids

    def draft_from_raw_claims(self, raw_claims: List[Tuple[str, List[str]]], turn_id: str) -> List[Claim]:
        """Draft claims from raw (text, doc_ids) tuples with enum citation enforcement."""
        claims: List[Claim] = []
        for text, doc_ids in raw_claims:
            clean_text = self.CITATION_REGEX.sub("", text).strip()
            if not clean_text:
                continue

            # Filter or record cited doc IDs
            cited_ids = [doc_id.strip("[]") for doc_id in doc_ids]
            
            # Create pending claim
            claims.append(
                Claim(
                    text=clean_text,
                    doc_ids=cited_ids,
                    verification_status=ClaimStatus.PENDING,
                    turn_id=turn_id,
                )
            )
        return claims

    def draft_from_text(self, text: str, turn_id: str) -> List[Claim]:
        """Extract sentences from text, parse citations, and construct Claim objects."""
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        claims: List[Claim] = []

        for sent in sentences:
            citations = self.CITATION_REGEX.findall(sent)
            clean_sent = re.sub(r"\s+([.!?,;:])", r"\1", self.CITATION_REGEX.sub("", sent)).strip()
            if not clean_sent:
                continue

            claims.append(
                Claim(
                    text=clean_sent,
                    doc_ids=citations,
                    verification_status=ClaimStatus.PENDING,
                    turn_id=turn_id,
                )
            )

        return claims

    def validate_enum_constraints(self, claim: Claim) -> Tuple[bool, List[str]]:
        """Verify that every cited Doc ID belongs strictly to the allowed enum set."""
        if not claim.doc_ids:
            return False, ["Missing citation: claim must cite at least one valid Doc ID."]

        hallucinated = [doc_id for doc_id in claim.doc_ids if doc_id not in self.allowed_doc_ids]
        if hallucinated:
            return False, [f"Hallucinated Doc ID: '{doc_id}' is not in retrieved context." for doc_id in hallucinated]

        return True, []

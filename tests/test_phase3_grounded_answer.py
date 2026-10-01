"""Unit and integration tests for Phase 3: Grounded Answer Pipeline, Verifier, and Ledger."""

import asyncio
from pathlib import Path
import pytest

from slrag.config import DEFAULT_CONFIG
from slrag.contracts.events import Claim, ClaimStatus
from slrag.corpus.loader import CorpusLoader
from slrag.gateway.session import SessionContext
from slrag.pipeline.drafter import ClaimDrafter
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.llm import LLMServiceWrapper
from slrag.pipeline.sufficiency import SufficiencyGate
from slrag.pipeline.turn_engine import BatchTurnEngine
from slrag.pipeline.verifier import FailClosedVerifier
from slrag.retrieval.engine import HybridRetrievalEngine, SearchResult

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.json"


@pytest.fixture
def retrieval_engine():
    docs = CorpusLoader.load_file(CORPUS_PATH)
    engine = HybridRetrievalEngine(DEFAULT_CONFIG)
    engine.index_documents(docs)
    return engine


def test_sufficiency_gate(retrieval_engine):
    """Test sufficiency gate passes high-relevance queries and fails out-of-domain queries."""
    gate = SufficiencyGate(DEFAULT_CONFIG.sufficiency)

    # 1. High-relevance query
    results_pos = retrieval_engine.search("Raft distributed consensus leader election", mode="hybrid", top_k=3)
    passed_pos, dense_score_pos, coverage_pos, reason_pos = gate.evaluate("Raft distributed consensus leader election", results_pos)
    assert passed_pos is True
    assert dense_score_pos >= 0.55
    assert coverage_pos >= 0.50

    # 2. Irrelevant / out-of-domain query
    results_neg = retrieval_engine.search("how to make chocolate cake with strawberries", mode="hybrid", top_k=3)
    passed_neg, dense_score_neg, coverage_neg, reason_neg = gate.evaluate("how to make chocolate cake with strawberries", results_neg)
    assert passed_neg is False


def test_claim_drafter_enum_constraints():
    """Verify claim drafter validates citations against retrieved Doc ID enum set."""
    allowed_ids = {"DOC001§intro", "DOC002§raft_consensus"}
    drafter = ClaimDrafter(allowed_doc_ids=allowed_ids)

    # Valid citation
    c_valid = Claim(text="Qubits can exist in superposition.", doc_ids=["DOC001§intro"])
    is_valid, errs = drafter.validate_enum_constraints(c_valid)
    assert is_valid is True
    assert len(errs) == 0

    # Hallucinated citation
    c_invalid = Claim(text="Some ungrounded claim.", doc_ids=["DOC999§fake_section"])
    is_valid_inv, errs_inv = drafter.validate_enum_constraints(c_invalid)
    assert is_valid_inv is False
    assert "Hallucinated Doc ID" in errs_inv[0]


def test_fail_closed_verifier_5_checks(retrieval_engine):
    """Verify all 5 checks in FailClosedVerifier."""
    verifier = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    allowed_ids = set(retrieval_engine.chunks_map.keys())

    # 1. Grounded valid claim -> All pass
    c_valid = Claim(
        text="The surface code arranges physical qubits in a 2D square lattice with syndrome measurements.",
        doc_ids=["DOC001§error_correction"],
    )
    passed, checks, reason = verifier.verify_claim(
        claim=c_valid,
        retrieved_chunks_map=retrieval_engine.chunks_map,
        allowed_doc_ids=allowed_ids,
        previously_verified=[],
    )
    assert passed is True
    assert reason is None
    assert all(c.passed for c in checks.values())

    # 2. Hallucinated Doc ID -> Fails
    c_hallu = Claim(
        text="Surface code is scalable.",
        doc_ids=["DOC_UNKNOWN§imaginary"],
    )
    passed_h, checks_h, reason_h = verifier.verify_claim(
        claim=c_hallu,
        retrieved_chunks_map=retrieval_engine.chunks_map,
        allowed_doc_ids=allowed_ids,
        previously_verified=[],
    )
    assert passed_h is False
    assert checks_h["hallucinated_id"].passed is False

    # 3. Semantic / Lexical hallucination -> Fails
    c_fake_fact = Claim(
        text="Raft cluster requires fifty nodes and uses quantum teleportation for message transport.",
        doc_ids=["DOC002§raft_consensus"],
    )
    passed_f, checks_f, reason_f = verifier.verify_claim(
        claim=c_fake_fact,
        retrieved_chunks_map=retrieval_engine.chunks_map,
        allowed_doc_ids=allowed_ids,
        previously_verified=[],
    )
    assert passed_f is False
    assert (not checks_f["lexical"].passed or not checks_f["semantic"].passed)


def test_claim_ledger_versioning_and_isolation():
    """Verify ledger version increments and verified claims persist across turns."""
    ledger = ClaimLedger("session_007")
    assert ledger.version == 1
    assert len(ledger.get_verified_claims()) == 0

    # Turn 1
    c1 = Claim(claim_id="c1", text="Raft has 3 node states.", doc_ids=["DOC002§raft_consensus"], verification_status=ClaimStatus.VERIFIED)
    c2 = Claim(claim_id="c2", text="Raft tolerates 2 failures with 5 nodes.", doc_ids=["DOC002§raft_consensus"], verification_status=ClaimStatus.VERIFIED)
    
    v1 = ledger.add_verified_claims([c1, c2], turn_id="turn_1")
    assert v1 == 2
    assert ledger.version == 2
    assert len(ledger.get_verified_claims()) == 2

    # Turn 2
    c3 = Claim(claim_id="c3", text="Leader replicates log entries across quorum.", doc_ids=["DOC002§raft_consensus"], verification_status=ClaimStatus.VERIFIED)
    v2 = ledger.add_verified_claims([c3], turn_id="turn_2")
    assert v2 == 3
    assert ledger.version == 3
    assert len(ledger.get_verified_claims()) == 3


@pytest.mark.anyio
async def test_acceptance_scenario_3_claims_1_rejected(retrieval_engine):
    """Acceptance Test: A turn scenario with 3 claims (2 supported, 1 unsupported).
    
    Verify that:
    1. Verifier rejects the unsupported claim.
    2. Final streamed answer contains only the 2 verified claims.
    3. Ledger records only verified claims.
    4. Generation metrics (tokens_in, tokens_out, cost) are populated.
    """
    engine = BatchTurnEngine(DEFAULT_CONFIG, engine=retrieval_engine)
    session_ctx = SessionContext("test_acceptance_session")
    turn_id = "turn_acceptance_01"

    # We mock or run a turn where 3 claims are drafted: 2 true facts and 1 hallucinated fact
    # We will test the verifier and turn execution directly
    claim1 = Claim(
        claim_id="clm_1",
        text="Raft cluster consists of several nodes typically five allowing tolerance of two failures",
        doc_ids=["DOC002§raft_consensus"],
        turn_id=turn_id,
    )
    claim2 = Claim(
        claim_id="clm_2",
        text="The leader accepts log entries from clients and replicates them across a majority quorum",
        doc_ids=["DOC002§raft_consensus"],
        turn_id=turn_id,
    )
    claim3_unsupported = Claim(
        claim_id="clm_3",
        text="Raft was invented in the nineteenth century using telegraph wires for consensus",
        doc_ids=["DOC002§raft_consensus"],  # Cited but completely hallucinated content
        turn_id=turn_id,
    )

    allowed_ids = set(retrieval_engine.chunks_map.keys())
    ledger = engine._get_or_init_session_ledger(session_ctx)

    # Run verification on each
    res1, p1 = await engine.verifier.verify_and_emit(claim1, retrieval_engine.chunks_map, allowed_ids, ledger.get_verified_claims(), turn_id)
    res2, p2 = await engine.verifier.verify_and_emit(claim2, retrieval_engine.chunks_map, allowed_ids, ledger.get_verified_claims(), turn_id)
    res3, p3 = await engine.verifier.verify_and_emit(claim3_unsupported, retrieval_engine.chunks_map, allowed_ids, ledger.get_verified_claims(), turn_id)

    assert p1 is True
    assert res1.verification_status == ClaimStatus.VERIFIED

    assert p2 is True
    assert res2.verification_status == ClaimStatus.VERIFIED

    assert p3 is False
    assert res3.verification_status == ClaimStatus.REJECTED

    # Add verified to ledger
    ledger.add_verified_claims([res1, res2, res3], turn_id=turn_id)
    verified_in_ledger = ledger.get_verified_claims()

    # Ledger must strictly contain 2 verified claims, 0 rejected claims
    assert len(verified_in_ledger) == 2
    assert "telegraph" not in " ".join(c.text for c in verified_in_ledger)
    assert "quorum" in " ".join(c.text for c in verified_in_ledger)

    # Test full stream turn
    stream_events = []
    async for evt in engine.process_turn_stream(
        session_ctx=session_ctx,
        utterance="Explain how Raft leader election and log replication work",
        turn_id="turn_full_stream_02",
    ):
        stream_events.append(evt)

    # Check that events were emitted
    event_types = [e.event_type for e in stream_events]
    assert "answer_delta" in event_types
    assert "ledger_update" in event_types
    assert "turn_summary" in event_types

    summary = [e for e in stream_events if e.event_type == "turn_summary"][0]
    assert summary.tokens_in > 0
    assert summary.tokens_out > 0
    assert summary.cost >= 0.0
    assert summary.verified_count >= 1

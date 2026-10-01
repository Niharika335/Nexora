"""Phase 7 delta planner: enum-constrained output, validation and the delta_fallback paths."""

import asyncio
import json
import logging

import pytest

from slrag.llm.heuristic import heuristic_delta_llm
from slrag.llm.schemas import SchemaError, delta_planner_schema, validate
from slrag.plan.delta_planner import DeltaPlanner

SUBS = [{"id": "sub_1", "text": "What is the refund policy?"}, {"id": "sub_2", "text": "How are invoices sent?"}]
CLAIMS = [
    {"claim_id": "c1", "sub_intent_id": "sub_1", "first_20_words": "Refunds are issued within 14 days of purchase."},
    {"claim_id": "c2", "sub_intent_id": "sub_2", "first_20_words": "Invoices are emailed on the first of the month."},
]


def fixed(output):
    async def fn(prompt, schema, context):
        return output
    return fn


def plan(llm_fn, utterance="The purchase was international.", **kw):
    return asyncio.run(DeltaPlanner(llm_fn, **kw).plan(utterance, SUBS, CLAIMS))


def test_valid_modifies_plan():
    p = plan(fixed(json.dumps({"relation": "modifies", "affected_sub_intents": ["sub_1"],
                               "delta_queries": ["international refund policy"]})))
    assert (p.relation, p.affected_sub_intents, p.delta_queries, p.fallback) == ("modifies", ["sub_1"], ["international refund policy"], False)
    assert p.payload() == {"relation": "modifies", "affected_sub_intents": ["sub_1"],
                           "delta_queries": ["international refund policy"], "fallback": False, "fallback_reason": None}
    assert p.tokens_in > 0 and p.tokens_out > 0


def test_schema_enumerates_current_sub_intent_ids():
    schema = delta_planner_schema(["sub_1", "sub_2"])
    validate({"relation": "adds", "affected_sub_intents": [], "delta_queries": ["q"]}, schema)
    with pytest.raises(SchemaError):
        validate({"relation": "modifies", "affected_sub_intents": ["sub_9"], "delta_queries": ["q"]}, schema)
    with pytest.raises(SchemaError):
        validate({"relation": "replaces", "affected_sub_intents": [], "delta_queries": ["q"]}, schema)
    with pytest.raises(SchemaError):  # at most 2 delta queries of <= 160 chars
        validate({"relation": "adds", "affected_sub_intents": [], "delta_queries": ["a", "b", "c"]}, schema)
    with pytest.raises(SchemaError):
        validate({"relation": "adds", "affected_sub_intents": [], "delta_queries": ["x" * 161]}, schema)


@pytest.mark.parametrize("output, reason", [
    ("not json at all", "invalid_json"),
    (json.dumps({"relation": "modifies", "affected_sub_intents": ["sub_9"], "delta_queries": ["q"]}), "schema"),
    (json.dumps({"relation": "modifies", "affected_sub_intents": [], "delta_queries": ["q"]}), "empty_affected"),
    (json.dumps({"relation": "modifies", "affected_sub_intents": ["sub_1"]}), "schema"),
])
def test_invalid_output_falls_back_to_adds_with_the_utterance(output, reason, caplog):
    with caplog.at_level(logging.WARNING, logger="slrag.plan.delta_planner"):
        p = plan(fixed(output))
    assert (p.relation, p.affected_sub_intents, p.delta_queries) == ("adds", [], ["The purchase was international."])
    assert p.fallback and p.fallback_reason == reason
    assert "delta_fallback" in caplog.text


def test_timeout_falls_back():
    async def slow(prompt, schema, context):
        await asyncio.sleep(1.0)
        return "{}"

    p = plan(slow, timeout_s=0.05)
    assert p.fallback and p.fallback_reason == "timeout" and p.relation == "adds"


def test_backend_error_falls_back():
    async def boom(prompt, schema, context):
        raise RuntimeError("model unavailable")

    p = plan(boom)
    assert p.fallback and p.fallback_reason == "llm_error"


def test_adds_and_unrelated_carry_no_affected_sub_intents():
    p = plan(fixed(json.dumps({"relation": "unrelated", "affected_sub_intents": ["sub_1"], "delta_queries": ["weather"]})))
    assert p.relation == "unrelated" and p.affected_sub_intents == [] and not p.fallback


def test_prompt_gets_sub_intents_and_claim_one_liners_only():
    seen = {}

    async def spy(prompt, schema, context):
        seen.update(prompt=prompt, context=context, schema=schema)
        return json.dumps({"relation": "adds", "affected_sub_intents": [], "delta_queries": ["q"]})

    plan(spy)
    assert seen["context"]["sub_intents"] == SUBS and seen["context"]["claims"] == CLAIMS
    assert seen["schema"]["properties"]["affected_sub_intents"]["items"]["enum"] == ["sub_1", "sub_2"]
    assert "Refunds are issued within 14 days" in seen["prompt"]


def test_stand_in_backend_relations():
    assert plan(heuristic_delta_llm, "Assume the refund was requested after 14 days.").affected_sub_intents == ["sub_1"]
    assert plan(heuristic_delta_llm, "How are invoices sent by email?").relation == "adds"
    assert plan(heuristic_delta_llm, "What is the weather on Mars?").relation == "unrelated"

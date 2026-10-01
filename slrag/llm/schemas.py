"""Enum-constrained JSON schemas for the Phase 7 LLM calls, and a minimal validator for them.

The schemas are built per call so that every id field is an `enum` of the ids that actually
exist at that moment (current sub-intents, affected claims, allowed cites). The validator
covers the JSON-Schema subset these schemas use: type, required, properties, enum, items,
minItems, maxItems and maxLength. Anything outside the enum is rejected, not repaired.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

RELATIONS = ("modifies", "adds", "unrelated")
REWRITE_ACTIONS = ("keep", "revise", "retract")


class SchemaError(ValueError):
    """The LLM output does not conform to the call's schema."""


def delta_planner_schema(sub_intent_ids: Sequence[str], max_queries: int = 2) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["relation", "affected_sub_intents", "delta_queries"],
        "properties": {
            "relation": {"enum": list(RELATIONS)},
            "affected_sub_intents": {"type": "array", "items": {"enum": list(sub_intent_ids)}},
            "delta_queries": {"type": "array", "maxItems": max_queries, "items": {"type": "string", "maxLength": 160}},
        },
    }


def rewriter_schema(
    affected_claim_ids: Sequence[str],
    allowed_cites: Sequence[str],
    affected_sub_intent_ids: Sequence[str],
    max_new_claims: int = 3,
) -> Dict[str, Any]:
    cite_items = {"enum": list(allowed_cites)}
    return {
        "type": "object",
        "required": ["decisions", "new_claims"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["claim_id", "action"],
                    "properties": {
                        "claim_id": {"enum": list(affected_claim_ids)},
                        "action": {"enum": list(REWRITE_ACTIONS)},
                        "text": {"type": "string", "maxLength": 320},
                        "cites": {"type": "array", "items": cite_items},
                    },
                },
            },
            "new_claims": {
                "type": "array",
                "maxItems": max_new_claims,
                "items": {
                    "type": "object",
                    "required": ["sub_intent_id", "text", "cites"],
                    "properties": {
                        "sub_intent_id": {"enum": list(affected_sub_intent_ids)},
                        "text": {"type": "string"},
                        "cites": {"type": "array", "minItems": 1, "items": cite_items},
                    },
                },
            },
        },
    }


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
}


def validate(data: Any, schema: Dict[str, Any], path: str = "$") -> None:
    """Raise SchemaError at the first violation of `schema`."""
    expected = schema.get("type")
    if expected is not None:
        py_type = _TYPES[expected]
        if not isinstance(data, py_type) or (expected in ("integer", "number") and isinstance(data, bool)):
            raise SchemaError(f"{path}: expected {expected}, got {type(data).__name__}")
    if "enum" in schema and data not in schema["enum"]:
        raise SchemaError(f"{path}: {data!r} is not one of {schema['enum']}")
    if isinstance(data, str) and "maxLength" in schema and len(data) > schema["maxLength"]:
        raise SchemaError(f"{path}: string longer than {schema['maxLength']}")
    if isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            raise SchemaError(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            raise SchemaError(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(data):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                raise SchemaError(f"{path}: missing required '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                validate(data[key], sub, f"{path}.{key}")


def enum_values(schema: Dict[str, Any], *path: str) -> List[Any]:
    """Enum at a property path, e.g. enum_values(s, "properties", "relation")."""
    node: Any = schema
    for key in path:
        node = node[key]
    return list(node.get("enum", []))

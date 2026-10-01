"""config.yaml is the source of DEFAULT_CONFIG and of cfg_hash."""

import asyncio
from pathlib import Path

import pytest
import yaml

from slrag.config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, config_from_dict, load_config
from slrag.contracts.events import TranscriptChunk, TurnStartEvent
from slrag.control import Decision, RetrievalController, TurnState

REQUIRED_KEYS = [
    ("controller", "new_info_cos"),
    ("controller", "stable_cos"),
    ("sufficiency", "dense_top1"),
    ("sufficiency", "coverage"),
    ("verifier", "lexical"),
    ("verifier", "semantic"),
]


def test_config_yaml_has_required_threshold_keys():
    raw = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    for section, key in REQUIRED_KEYS:
        assert key in raw[section], f"{section}.{key} missing from config.yaml"
        assert getattr(getattr(DEFAULT_CONFIG, section), key) == raw[section][key]


def test_default_config_and_events_use_the_yaml_hash():
    assert DEFAULT_CONFIG.cfg_hash == load_config(DEFAULT_CONFIG_PATH).cfg_hash
    assert TranscriptChunk(text="x").cfg_hash == DEFAULT_CONFIG.cfg_hash
    assert TurnStartEvent().cfg_hash == DEFAULT_CONFIG.cfg_hash


def test_changing_a_threshold_in_yaml_changes_cfg_hash(tmp_path, monkeypatch):
    raw = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    raw["controller"]["new_info_cos"] = 0.80
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(custom)
    assert cfg.controller.new_info_cos == 0.80 and cfg.cfg_hash != DEFAULT_CONFIG.cfg_hash
    monkeypatch.setenv("SLRAG_CONFIG", str(custom))
    assert load_config().cfg_hash == cfg.cfg_hash


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match="new_info_cosine"):
        config_from_dict({"controller": {"new_info_cosine": 0.8}})
    with pytest.raises(ValueError, match="controllr"):
        config_from_dict({"controllr": {}})


def test_missing_explicit_config_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_slot_change_is_new_information():
    # Pin new_info_cos below this query's drift (cos ~0.84) so drift alone is not new information and
    # the test isolates the slot rule, independent of the calibrated value in config.yaml.
    import dataclasses

    controller_cfg = dataclasses.replace(DEFAULT_CONFIG.controller, new_info_cos=0.82)

    async def go():
        ctl, st, out = RetrievalController(controller_cfg), TurnState(turn_id="t"), []
        for chunk in ["Explain how Raft leader", "election tolerates two failures", "in a five node cluster"]:
            st.buffer = f"{st.buffer} {chunk}".strip()
            out.append(await ctl.on_chunk(st))
        return out

    decisions = asyncio.run(go())
    assert decisions[1].decision == Decision.PROVISIONAL
    assert (decisions[2].decision, decisions[2].reason) == (Decision.PROVISIONAL, "slot_change")

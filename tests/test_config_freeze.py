"""Phase 10 config freeze: config.yaml is marked frozen and its values are the ones the final
experiments were measured with (cfg_hash pinned). Changing any setting fails this test on purpose:
re-calibrate on the tune split, re-run the final experiments, then update FROZEN_CFG_HASH."""

from pathlib import Path

import yaml

from slrag.config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, config_from_dict, load_config

FROZEN_CFG_HASH = "e3745a51d783adef"  # out/final_experiments.json "cfg_hash" (test split)
FROZEN_THRESHOLDS = {
    ("controller", "mode"): "cascade",
    ("controller", "new_info_cos"): 0.86,
    ("controller", "stable_cos"): 0.82,
    ("controller", "per_clause_retrieval"): True,
    ("cache", "reuse_cos"): 0.82,
    ("sufficiency", "dense_top1"): 0.55,
    ("sufficiency", "coverage"): 0.50,
    ("sufficiency", "uncertain_low"): 0.50,
    ("sufficiency", "uncertain_high"): 0.50,
    ("dense", "model_name"): "sentence-transformers/all-MiniLM-L6-v2",
    ("verifier", "lexical"): 0.30,
    ("verifier", "semantic"): 0.65,
    ("speculation", "mode"): "full",
    ("llm", "backend"): "heuristic",
}


def test_config_yaml_is_frozen_with_the_measured_values(monkeypatch):
    monkeypatch.delenv("SLRAG_LLM_BACKEND", raising=False)
    monkeypatch.delenv("SLRAG_OLLAMA_URL", raising=False)
    raw = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    assert raw["frozen"] is True
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.frozen is True and DEFAULT_CONFIG.frozen is True
    assert cfg.cfg_hash == FROZEN_CFG_HASH
    for (section, key), value in FROZEN_THRESHOLDS.items():
        assert getattr(getattr(cfg, section), key) == value, f"{section}.{key} changed after the freeze"


def test_frozen_flag_is_metadata_not_a_setting():
    raw = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    unfrozen = config_from_dict({k: v for k, v in raw.items() if k != "frozen"})
    assert unfrozen.frozen is False
    assert unfrozen.cfg_hash == config_from_dict(raw).cfg_hash  # the flag does not change cfg_hash
    assert "frozen" not in config_from_dict(raw).to_dict()

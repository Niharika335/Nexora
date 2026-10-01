# SL-RAG developer commands. Windows without make: run the commands after each target directly.
MODEL ?= qwen2.5:3b
PYTHON ?= python

.PHONY: model up down test experiments audit replay

model:  ## download the dense retrieval model into models/, start ollama, pull the drafting model
	$(PYTHON) scripts/fetch_models.py
	docker compose up -d ollama
	docker compose exec ollama ollama pull $(MODEL)

up:  ## build and start the app (UI on http://localhost:8000/ui/trace, admin/slrag)
	docker compose up --build

down:
	docker compose down

test:
	$(PYTHON) -m pytest tests/ -q

replay:  ## test-split replay: out/summary.json + out/report.md
	$(PYTHON) -m slrag.cli replay eval/scenarios --split test --out out/ours_test.jsonl

experiments:  ## A1-A5 ablations: out/final_experiments.json
	$(PYTHON) scripts/run_experiments.py

audit:  ## compliance audit: out/compliance.json
	$(PYTHON) scripts/compliance_audit.py

# Corpus format

## What the corpora are

- **`nexora_corpus.json`**: 24 documents (93 sections) about Nexora, a **fictional** B2B SaaS company that sells an AI-powered event management platform: product and features, pricing, onboarding, API reference, SLA, security, privacy, company, release notes v2.3–v2.5, a support FAQ and the Salesforce, HubSpot, Zapier and Slack integrations. Nexora, its people, prices and figures are invented.
- **`sample_corpus.json`**: 4 short technical documents (qubits, Raft, LSM trees, HNSW / BM25) from Phase 1.

Both are **synthetic and AI-authored**, and the evaluation scenarios in `eval/scenarios/` were written by the same process against these texts. Recall, groundedness and gate numbers measured on them show that the pipeline works on a realistic-looking domain, not how it performs on real customer documents. Replace them with real documents (and re-author the scenarios) before claiming benchmark results.

## Expected format

A corpus file is a JSON array with one object per document:

```json
[
  {
    "doc_id": "doc-billing-faq",
    "title": "Billing FAQ",
    "sections": [
      { "section_id": "c0", "title": "Refund policy", "text": "Refunds are issued within 14 days ..." },
      { "section_id": "c1", "title": "Invoices",      "text": "Invoices are emailed on the 1st ..." }
    ]
  }
]
```

| Field | Required | Notes |
|---|---|---|
| `doc_id` | yes | Unique across the whole corpus. Use lowercase with hyphens, and no `§`. |
| `title` | yes | A human-readable document title. |
| `sections` | yes | A non-empty list. |
| `sections[].section_id` | recommended | Unique within the document. If omitted, it is derived from the section `title`. |
| `sections[].title` | optional | The section heading. It is stored on each chunk as `section_title`. |
| `sections[].text` | yes | Plain text. Markdown is fine. |

Other formats also load (see `slrag/corpus/loader.py`):
- **`.jsonl`:** one document object per line.
- **`.md` / `.txt`:** one document per file, split into sections on headings. The `doc_id` comes from the filename.

Extra top-level keys are kept as document metadata and ignored by retrieval.

## Chunk IDs and scenario gold labels

The chunker turns each section into chunks with the ID `<doc_id>§<section_id>` (for example `nx-pricing§growth`), with a suffix when a section longer than 500 tokens is split further. Scenario files in `eval/scenarios/*.jsonl` name their gold evidence by these chunk IDs. So after swapping in a real corpus you must:

1. Point the corpus loaders at the new file. `slrag/replay/baselines.py` (`DEFAULT_CORPORA`) and the tests currently load `data/sample_corpus.json` and `data/nexora_corpus.json`.
2. Re-index with `slrag index <corpus> -o data/index`.
3. Rewrite or re-label the scenarios so their questions are answerable from the new documents and their gold chunk IDs exist in the new index.
4. Check the result with `python scripts/validate_scenarios.py eval/scenarios --corpus data/<your_corpus>.json`. It fails on any gold ID missing from the corpus. If you omit `--corpus`, it uses `DEFAULT_CORPORA`.
5. Make a fresh `eval/split.json`. The current test split has already been seen during development.

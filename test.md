# RAG Pipeline Candidate Test – README

## Objective

Create a RAG pipeline to answer user questions:

* Always use **factual dataset** first.
* Use **external dataset** only if facts are missing and not for sensitive info (pricing, warranty, etc.).
* Must avoid hallucinations.

## Requirements

1. Ingest `facts.json` and `external.json` into any vector DB.
2. Create an API endpoint `/ask`:

   * Retrieve from **facts** first.
   * If not found, fallback to external (non‑sensitive only).
   * If no safe answer → return refusal.
3. Use any LLM model.
4. Define guardrails to:

   * Never contradict facts.
   * Refuse unsafe answers.
   * Avoid hallucinations.

## Endpoint Example

**POST /ask**

```json
{
  "question": "What is the warranty for Model X?"
}
```

**Response**

```json
{
  "answer": "2 years [F12:c3]",
  "status": "answered",
  "citations": [{"source":"facts","doc_id":"F12","chunk_id":"c3"}]
}
```

## Guardrails

* Facts first, external only if safe.
* Refuse pricing/warranty/availability if not in Facts.
* Always cite sources.

## Deliverables

* Code for ingestion + API.
* README + short DESIGN.md explaining guardrails.
* Reproducible run instructions.

## Evaluation Rubric (100 pts)

| Area                    | Points | What we look for                                              |
| ----------------------- | -----: | ------------------------------------------------------------- |
| Grounding & correctness |     35 | Accurate, cites Facts per sentence; no hallucinations         |
| Guardrails & policy     |     25 | Pricing/warranty rules; clean refusals; no External override  |
| Retrieval quality       |     20 | Pulls relevant Facts; robust to paraphrase; sensible chunking |
| Code quality            |     10 | Readable, modular, testable; sensible configs                 |
| Documentation           |     10 | Clear setup, runbook, and design rationale                    |

**Pass criterion:** ≥ 75 with **zero** hallucinations on pricing/warranty/availability/others.



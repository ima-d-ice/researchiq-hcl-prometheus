# ResearchIQ

Multimodal AI research assistant: ask questions over research PDFs and get answers cited down to paper · page · figure / table.

## Pipelines

| Lane | Where | Flow |
|---|---|---|
| A · Ingestion | `src/researchiq/ingestion/` | upload → Docling parse → (text: HybridChunker) / (visual: crop → VLM describe) → contextualize + ID → embed ×2 → Qdrant |
| B · Retrieval | `src/researchiq/retrieval/` | guard → router → hybrid search (dense 30 + BM25 30) → RRF → rerank top 6 |
| C · Generation | `src/researchiq/generation/` | context builder → Qwen 3.8 27B (JSON mode) → validate → citation check → UI |
| D · Evaluation | `eval/` | gold set → Recall@5 / MRR / ablation → RAGAS / judge → `eval/report.md` |

Only the VLM describe, answer generation and judge steps call the Groq API (`src/researchiq/llm/groq_client.py`). Everything else runs locally.

## Layout

```
src/researchiq/
  config.py            settings (env-driven)
  schemas.py           shared Pydantic models
  llm/groq_client.py   single Groq entry point
  ingestion/           lane A
  retrieval/           lane B
  generation/          lane C
  ui/app.py            Streamlit UI
eval/                  lane D (offline script)
scripts/ingest.py      CLI: ingest PDFs into Qdrant
data/papers/           input PDFs
data/cache/            figures.json (VLM description cache), crops
data/qdrant/           local Qdrant storage
tests/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,dev]"
cp .env.example .env            # add GROQ_API_KEY

python scripts/ingest.py data/papers          # PDFs, Office files, md/txt, images
streamlit run src/researchiq/ui/app.py
python eval/run_eval.py
```

## Ingestion (lane A)

Supported inputs: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.md`, `.txt`, `.html`, `.png`, `.jpg`.
Old `.doc` / `.xls` / `.ppt` files are rejected with a "save as .docx / .xlsx / .pptx" message.

```bash
python scripts/ingest.py data/papers            # folder, files or globs
python scripts/ingest.py data/papers --no-vlm   # no Groq calls (0 tokens)
python scripts/ingest.py paper.pdf --max-vlm-calls 10
```

- Docling parses every format into one document model; if it fails, a lightweight fallback
  (PyMuPDF / python-docx / python-pptx / openpyxl / plain text) produces the same chunk schema.
- Spreadsheet and CSV tables are kept as exact text and are never sent to the VLM.
- Re-ingesting a file replaces its old chunks (deterministic IDs + `delete_paper`).
- Optional `data/papers/catalogue.csv` with columns `filename,paper_id,title,year,source_url`.

### Groq token budget (8k tokens/minute)

Every Groq call goes through one rate limiter (`llm/rate_limiter.py`) that keeps the rolling
60-second total under 90% of `GROQ_TPM_LIMIT` (7,200) and the request rate under the RPM limit.
Each VLM request is sized before sending (image downscaled to <=1,536 tokens, text truncated,
output capped at 600) so it stays about 3k tokens, and is never sent if it cannot fit.
Figure descriptions are cached in `data/cache/figures.json`, so re-ingesting costs 0 tokens.
The limiter is per process: do not run ingestion and eval at the same time on one API key.

### Rotating several Groq keys

Put all keys in `.env` (gitignored), comma-separated: `GROQ_API_KEYS=gsk_a...,gsk_b...,gsk_c...`.
Each key has its own 8k TPM limiter; calls are spread round-robin, and a key that returns 429
is put on cooldown (using `retry-after`) while the call moves to the next key.

## Evaluation (lane D)

```bash
python eval/run_eval.py --no-llm            # retrieval + local security checks, 0 Groq tokens
python eval/run_eval.py                     # everything; writes eval/report.md
python eval/run_eval.py --only answers --limit 5 --judge-limit 3
```

| Part | What it reports | Cost |
|---|---|---|
| Retrieval | Recall@5, MRR, Hit@5 for 4 ablations (dense, +rerank, +BM25/RRF, full); Recall@5 by question type; router accuracy | local |
| Answers | faithfulness, answer relevancy, context precision, context recall, correctness (one judge call per question); citation accuracy vs expected pages; refusal correctness on unanswerable questions; false refusals; latency | ~4k tokens/question + ~3k/judged |
| Security | prompt injection (6 attacks, as crafted chunks and as ingested PDFs), direct jailbreak questions, bad uploads, secrets hygiene | ~4k tokens/LLM case |

**Gold set** (`eval/gold_set.jsonl`): rows with `TODO` are skipped. Each row needs `id`, `type`
(factual / compare / figure / conflict / unanswerable / injection), `question`, `expected_sources`
(`[{"paper_id", "page", "figure_label"?}]`, empty only for unanswerable), `reference_answer`,
optional `route` and `must_not_contain`. Aim for 15–20 rows across all types.

**Document-level injection tests** need the synthetic attack PDFs indexed:

```bash
python eval/attacks/make_docs.py
python scripts/ingest.py data/eval_attacks --no-vlm
```

Judge scores are not ground truth; hand-check a few. Fill in the root cause and fix for each
failure case in the report by hand.

## Web UI

FastAPI server (`src/researchiq/api/server.py`) plus a React frontend (`web/`, built with Vite).

```bash
pip install fastapi "uvicorn[standard]" python-multipart
cd web && npm install && npm run build && cd ..
uvicorn researchiq.api.server:app --port 8000      # open http://localhost:8000
```

For frontend development, run the API on :8000 and `npm run dev` in `web/` (proxies `/api`).

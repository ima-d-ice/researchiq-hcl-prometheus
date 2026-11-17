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

python scripts/ingest.py data/papers/*.pdf
streamlit run src/researchiq/ui/app.py
python eval/run_eval.py
```

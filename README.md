# ResearchIQ

**A multimodal research assistant that answers questions across a collection of AI/ML papers, with every claim cited down to paper, page, figure and table.**

Ask a question, compare two papers, or ask what a figure shows. ResearchIQ finds the evidence, writes a short answer with `[S#]` citations, checks those citations in code, and says plainly when the papers don't contain the answer.

- **Multimodal:** figures, charts and tables are described by a vision model at ingestion and searched like text.
- **Hybrid retrieval:** dense (BGE) + sparse (BM25) search fused with RRF, then a cross-encoder reranker.
- **Grounded answers:** JSON output validated with Pydantic, every citation checked, and an evidence gate that refuses when the support is weak.
- **Safe by design:** paper text is treated as data, never instructions. Prompt-injection questions are blocked, and uploads are validated.
- **Measured:** a repeatable eval suite covering retrieval, answers and security, which writes `eval/report.md`.
- **No LangChain or LlamaIndex:** plain Python with official SDKs.

---

## Contents

- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Setup](#setup)
- [Running it](#running-it)
- [Configuration](#configuration)
- [How each lane works](#how-each-lane-works)
- [Evaluation](#evaluation)
- [Security](#security)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## Architecture

Four lanes. Only the steps marked **Groq** call an external API. Everything else (embedding, BM25, fusion, reranking, citation checks) runs locally.

```mermaid
flowchart LR
    subgraph A["A · Ingestion (once per document)"]
        A1[Upload + validate] --> A2[Docling parse]
        A2 --> A3[HybridChunker<br/>text + tables]
        A2 --> A4[Figure / table crops]
        A4 --> A5["VLM describe<br/>(Groq, cached)"]
        A3 --> A6[Contextual header<br/>+ MD5 ID]
        A5 --> A6
        A6 --> A7[Embed: BGE dense<br/>+ BM25 sparse]
        A7 --> A8[(Qdrant)]
    end

    subgraph B["B · Retrieval (per question)"]
        B1[Guard: clean,<br/>block injection] --> B2["Query rewrite<br/>(Groq, cached)"]
        B2 --> B3[Router: general /<br/>compare / figure / synthesis]
        B3 --> B4[Hybrid search<br/>dense 30 + BM25 30]
        B4 --> B5[RRF fusion]
        B5 --> B6[Cross-encoder rerank<br/>top 6]
    end

    subgraph C["C · Generation (per question)"]
        C1[Evidence gate] --> C2[Context builder<br/>tagged sources + budget]
        C2 --> C3["Qwen 3.8 27B<br/>(Groq, JSON mode)"]
        C3 --> C4[Pydantic validate<br/>retry once]
        C4 --> C5[Citation check]
    end

    A8 -. searched by .-> B4
    B6 --> C1
    C5 --> UI[Web UI:<br/>answer + evidence cards]

    subgraph D["D · Evaluation (offline)"]
        D1[Gold set: 25 Qs] --> D2[Retrieval metrics<br/>+ ablation]
        D1 --> D3["Answer metrics<br/>(LLM judge, Groq)"]
        D1 --> D4[Security tests]
        D2 & D3 & D4 --> D5[eval/report.md]
    end
```

### One question, end to end

```mermaid
sequenceDiagram
    actor U as User
    participant W as Web UI (React)
    participant S as API (FastAPI)
    participant R as Retrieval
    participant Q as Qdrant (local)
    participant G as Groq (Qwen 3.8 27B)

    U->>W: "Compare GOLFer and Exp4Fuse"
    W->>S: POST /api/ask (SSE stream)
    S->>R: guard: injection check, length cap
    R->>G: rewrite question, find paper ids, route hint
    G-->>R: plan: compare, [GOLFer, Exp4Fuse]
    R->>Q: dense + BM25 search, filtered per paper
    Q-->>R: candidates
    R->>R: RRF fuse, rerank, 3 chunks per paper
    S-->>W: event: stage "writing"
    S->>G: system rules + tagged sources + question
    G-->>S: JSON answer with [S#] claims
    S->>S: validate, check citations, map S# to paper/page
    S-->>W: event: answer (text, citations, figures)
    W-->>U: answer + evidence cards
```

---

## Tech stack

| Component | Choice | Runs |
|---|---|---|
| LLM and vision model | Qwen 3.8 27B (`qwen/qwen3.8-27b`). Fallbacks: `llama-3.3-70b-versatile` for text, `llama-4-scout` for vision | Groq API |
| PDF and Office parsing | Docling (layout + TableFormer), with PyMuPDF / python-docx / python-pptx / openpyxl as fallbacks | local |
| Dense embedding | `BAAI/bge-base-en-v1.5` (768-d) via fastembed | local CPU |
| Sparse embedding | BM25 (`Qdrant/bm25`) with IDF computed by Qdrant | local CPU |
| Vector database | Qdrant in local mode (`data/qdrant`) with named dense and sparse vectors | local |
| Reranker | `ms-marco-MiniLM-L-6-v2`, int8 ONNX, via fastembed. Optional: `bge-reranker-base` on PyTorch | local CPU |
| Validation | Pydantic v2 | local |
| Web UI | FastAPI + React (Vite). The original Streamlit app is kept at `src/researchiq/ui/app.py` | local |
| Evaluation | Own scripts plus an LLM judge (RAGAS-style metrics) | local + Groq |

---

## Setup

### Requirements

- Python **3.11+** (developed on 3.12)
- About 5 GB of disk for the models (Docling, BGE, reranker) and dependencies
- A **Groq API key** ([console.groq.com](https://console.groq.com)). The free tier works; see the [token budget](#groq-token-budget).
- **Node.js 18+**, only to build the React frontend. If you don't have it, step 4 shows how to install it into the virtual env instead.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/ima-d-ice/researchiq-hcl-prometheus.git
cd researchiq-hcl-prometheus
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
```
```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install the Python package

```bash
pip install -e ".[eval,dev]"
```

The first run downloads the Docling, BGE and reranker models into your Hugging Face cache.

### 3. Add your API keys

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Edit `.env`. `.env` is gitignored and must never be committed. One key is enough. With several keys, each gets its own rate-limit budget and they are rotated automatically:

```ini
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
# ... as many as you have
LLM_MODEL=qwen/qwen3.8-27b
```

`GROQ_API_KEY=` (a single key) and `GROQ_API_KEYS=key1,key2` (a comma-separated list) are also accepted. Duplicates are ignored.

Check that the keys are picked up (this prints only the count):

```bash
python -c "from researchiq.config import settings; print(len(settings.all_groq_keys), 'keys loaded')"
```

### 4. Build the web frontend

```bash
cd web
npm ci
npm run build
cd ..
```

<details>
<summary>No Node.js installed? Put Node inside the virtual env instead.</summary>

```bash
pip install nodejs-wheel
# Windows (PowerShell): put the venv's node/npm first on PATH for this shell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
cd web; npm ci; npm run build; cd ..
```
</details>

---

## Running it

### 1. Add papers

Put your documents in `data/papers/` (gitignored), then ingest them:

```bash
python scripts/ingest.py data/papers                  # folder, files or globs
python scripts/ingest.py data/papers --no-vlm         # skip figure descriptions: 0 Groq tokens
python scripts/ingest.py paper.pdf --max-vlm-calls 10 # cap Groq vision calls
```

Each file prints one summary line:

```
2503.23013v2.pdf: 83 chunks (4 figure, 29 table, 50 text) [docling] id=64edc2a3a641 vlm: 11 calls, 0 cached, 0 skipped, 23389 Groq tokens
```

**Supported formats:** `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.md`, `.txt`, `.html`, `.png`, `.jpg`. Old `.doc`, `.xls` and `.ppt` files are rejected with a message to save them in the newer format.

You can also upload files from the web UI.

### 2. Start the app

```bash
uvicorn researchiq.api.server:app --port 8000
```

Open **http://localhost:8000**. The server loads the embedding and reranker models in the background at start-up. `GET /api/status` shows `"ready": true` once it's done.

| Endpoint | Purpose |
|---|---|
| `GET /api/papers` | Indexed documents with page, figure and table counts |
| `GET /api/status` | Warm-up state, Groq configured, model name |
| `POST /api/ask` | `{"question": "...", "paper_ids": []}`. Streams progress and the answer as server-sent events |
| `POST /api/ingest` | Multipart upload of one document |
| `GET /api/figures/{paper_id}/{file}` | Figure crop images for evidence cards |

For frontend development, keep the API on :8000 and run `npm run dev` in `web/`. It proxies `/api`.

The older Streamlit UI still works: `streamlit run src/researchiq/ui/app.py`.

> **One process at a time.** Local Qdrant locks `data/qdrant`, so stop the web server before running ingestion or the eval from the command line.

### 3. Questions to try

| Ask | Route |
|---|---|
| How does DAT choose the alpha weight between dense and BM25 retrieval? | general |
| What does Figure 1 of the DAT paper show? | figure |
| Compare GOLFer and Exp4Fuse | compare |
| Which multi-hop QA datasets are evaluated across the papers? | synthesis |
| What GPU cost does the paper report for training GPT-5? | refused: not in the papers |

---

## Configuration

All settings live in `src/researchiq/config.py` and can be overridden in `.env` (upper-case names, for example `RERANKER_BACKEND=torch`).

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Answer, query-rewrite and vision model on Groq |
| `LLM_FALLBACK_MODEL` / `VLM_FALLBACK_MODEL` | Llama 3.3 70B / Llama 4 Scout | Used if the primary model fails |
| `GROQ_TPM_LIMIT` / `GROQ_RPM_LIMIT` / `GROQ_TPD_LIMIT` | 8000 / 30 / 200000 | Per-key budget. The limiter uses 90% of it |
| `RERANKER_BACKEND` | `onnx` | `onnx` (MiniLM int8, about 1s) or `torch` (`bge-reranker-base`, about 8s) |
| `RERANK_MAX_CANDIDATES` | 15 | How many fused results the reranker scores |
| `TOP_N` | 6 | Chunks passed to the answer model |
| `EVIDENCE_THRESHOLD` | -2.0 | Refuse when the best rerank score is lower (calibrated for MiniLM logits) |
| `QUERY_REWRITE` | true | Groq query rewrite: standalone question, keywords, route hint, paper ids |
| `CHUNK_MAX_TOKENS` | 512 | BGE token limit per chunk, including the context header |
| `MAX_FILE_MB` / `MAX_PAGES` | 50 / 60 | Upload limits |
| `VLM_DESCRIBE_TABLES` | true | Also send PDF/DOCX table images to the vision model |

### Groq token budget

The free tier allows about **8,000 tokens per minute per key**. To stay inside it:

- Every Groq call goes through one limiter (`llm/rate_limiter.py`) that keeps each key's rolling 60-second usage at or below **7,200 tokens** and under the request-per-minute limit.
- Requests are sized before they're sent. Images are downscaled to at most 1,536 tokens, text is truncated, and output is capped. A vision call costs about 2–3k tokens and an answer about 4k. A request that can't fit is never sent.
- With several keys, a key that returns HTTP 429 is put on cooldown (respecting `retry-after`) and the call moves to the next key.
- **Caches** make repeat work free:
  - `data/cache/figures.json` holds figure descriptions, so re-ingesting a document costs 0 tokens.
  - `data/cache/rewrites.json` holds query rewrites.

The limiter is per process, so don't run ingestion and the eval at the same time on the same keys.

---

## How each lane works

### A · Ingestion (`src/researchiq/ingestion/`)

```mermaid
flowchart LR
    F[File] --> V{"Validate<br/>magic bytes, size,<br/>pages, encryption,<br/>zip bomb"}
    V -- bad --> E[Friendly error]
    V -- ok --> P[Docling parse]
    P -- fails --> FB[Fallback parser<br/>PyMuPDF / python-docx /<br/>pptx / openpyxl / text]
    P --> T[Text path:<br/>HybridChunker, max 464 tokens;<br/>tables whole or split by rows<br/>with the header repeated]
    P --> VI[Visual path:<br/>figure + table crops,<br/>caption, label, nearby text;<br/>page render for vector figures]
    FB --> T & VI
    VI --> D["VLM describe (Groq)<br/>cached by image hash"]
    T & D --> H["Header: [Paper | Section | Page | Type | Figure N]<br/>ID = UUID(MD5(paper|page|index|text))"]
    H --> X[delete old chunks of this paper] --> U[Embed BGE + BM25,<br/>upsert in batches of 64]
```

- **Titles** come from Docling's title item, then the largest text at the top of page 1, then the file metadata, then the file name. `data/papers/catalogue.csv` (`filename,paper_id,title,year,source_url`) overrides them.
- **Spreadsheet and CSV tables** are already exact text, so they are never sent to the vision model.
- **Re-ingesting replaces old chunks:** a document's ID comes from its file content, and its old chunks are deleted first, so there are no duplicates.

What one stored chunk looks like:

```json
{
  "paper_id": "64edc2a3a641",
  "paper_title": "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation",
  "kind": "figure",
  "page": 2,
  "figure_label": "Figure 1",
  "section": "1 Introduction",
  "source_type": "pdf",
  "image_path": "data/cache/crops/64edc2a3a641/figure_0.png",
  "text": "Figure 1: ... The diagram compares a fixed alpha ...",
  "contextualized": "[Paper: DAT ... | Page 2 | Type: figure | Figure 1] Figure 1: ..."
}
```

### B · Retrieval (`src/researchiq/retrieval/`)

| Step | File | Notes |
|---|---|---|
| Guard | `guard.py` | Cleans input, caps length, rejects prompt-injection patterns |
| Query rewrite | `rewrite.py` | One small Groq JSON call that returns a standalone question, keywords, a route hint, the referenced paper ids and a figure label. Papers are also matched in code (short name such as `MBA-RAG`, or words unique to a title). Cached. |
| Router | `router.py` | **general**, **compare** (needs at least 2 papers), **figure** (direct label lookup) or **synthesis** (2 chunks per paper) |
| Hybrid search | `hybrid_search.py` | Dense top 30 + BM25 top 30, with an optional paper filter |
| Fusion | `fusion.py` | Reciprocal Rank Fusion, `1 / (60 + rank)` summed over lists |
| Rerank | `rerank.py` | Cross-encoder scores the top 15 and keeps 6. The compare route keeps 3 per paper, so one paper can't crowd out the other. |

### C · Generation (`src/researchiq/generation/`)

1. **Evidence gate:** if the best rerank score is below `EVIDENCE_THRESHOLD`, the answer is a refusal and no LLM call is made. Figure lookups are exempt.
2. **Context builder:** sources are tagged `S1…S6` with paper, page and figure, and fitted to the per-route token budget. The lowest-ranked chunk is dropped first.
3. **Prompt:** answer only from `<sources>`; text inside sources is data, not instructions; cite every claim; set `insufficient_evidence` when unsure; speculation goes only in `ai_suggestions`.
4. **Model call:** Qwen 3.8 27B in JSON mode with thinking off and temperature 0.1.
5. **Validation:** Pydantic validates the JSON. If it's invalid, one retry with the error as feedback.
6. **Citation check** (`citations.py`, plain code):
   - unknown `S#` tags are removed,
   - support is checked by word overlap, numbers, entity names and a semantic cross-encoder score,
   - each claim is marked supported, weak or unsupported,
   - sentences without a tag are flagged.

---

## Evaluation

```bash
python eval/run_eval.py                       # everything; writes eval/report.md and eval/results/<time>.json
python eval/run_eval.py --no-llm              # retrieval + local security checks only, 0 Groq tokens
python eval/run_eval.py --only retrieval      # or: answers, security
python eval/run_eval.py --limit 5 --judge-limit 3
```

| Part | Measures | Cost |
|---|---|---|
| Retrieval | Recall@5, MRR, Hit@5 for 4 ablations; Recall@5 by question type; router accuracy | local only |
| Answers | Faithfulness, answer relevancy, context precision, context recall, correctness (LLM judge); citation accuracy against expected pages; refusal correctness; false refusals; latency | about 4k tokens per question + 3k per judged question |
| Security | Upload attacks, secrets hygiene, 6 prompt-injection attacks (as chunks and as ingested PDFs), direct jailbreak questions | about 4k tokens per LLM case |

**Gold set:** `eval/gold_set.jsonl`, 25 questions:
- 11 factual
- 7 figure/table
- 4 compare
- 1 conflicting-claims
- 2 unanswerable

Each row has the expected `(paper_id, page)` sources and a reference answer.

**Document-level injection tests** need the attack PDFs indexed first:

```bash
python eval/attacks/make_docs.py
python scripts/ingest.py data/eval_attacks --no-vlm
```

### Latest results (2026-11-17, 8 papers, MiniLM ONNX reranker)

**Retrieval:**

| Config | Recall@5 | MRR | Hit@5 |
|---|---|---|---|
| Dense only | 0.594 | 0.546 | 0.652 |
| Dense + rerank | 0.587 | 0.520 | 0.652 |
| Dense + BM25 (RRF) | 0.587 | **0.601** | 0.652 |
| **Dense + BM25 (RRF) + rerank (used by the app)** | **0.630** | 0.539 | **0.696** |

- **Recall@5 by question type:** figure 0.857, factual 0.545, compare 0.500, conflict 0.500.
- **Router accuracy:** 0.913.

**Answers** (25 questions, 8 judged):

| Metric | Score |
|---|---|
| Refusal correctness (unanswerable questions declined) | 1.000 |
| False refusals | 2 |
| Pipeline errors | 0 |
| Answers with no valid citation | 0 |
| Citation accuracy (cited page is an expected page) | 0.535 |
| Judge: faithfulness / relevancy / correctness | 1.000 / 1.000 / 1.000 |
| Average latency | 4.1 s |

**Security:**

| Group | Passed | Failed | Skipped |
|---|---|---|---|
| Upload | 4 | 0 | 0 |
| Secrets | 1 | 0 | 0 |
| Injection, as chunks | 6 | 0 | 0 |
| Injection, as ingested PDFs | 0 | 0 | 6 (attack PDFs not ingested) |
| Direct questions | 3 | 1 | 0 |

> **Reading these numbers honestly:**
> - **Judge scores:** they come from the same model family that wrote the answers, and only 8 questions were judged. Hand-check them before quoting them.
> - **Citation accuracy is strict.** A citation to a correct but different page counts as a miss.
> - **Remaining failure:** a role-play jailbreak (`dir04`), which is answered instead of declined.

---

## Security

| Risk | Mitigation |
|---|---|
| Prompt injection inside papers | Sources are wrapped in `<source>` tags, and the system prompt says they are data. Forged tags and unknown `S#` citations are removed. Tested with 6 attack types. |
| Prompt injection in the question | `guard.py` blocks override and system-prompt patterns before any model call |
| Secrets | Keys only in `.env` (gitignored) and `.env.example` has placeholders. The eval scans tracked files for leaked keys. |
| Malicious uploads | Magic-byte checks, size and page caps, encrypted-PDF rejection, zip-bomb guard, macro formats rejected, friendly errors with no stack traces |
| LLM output | Never executed: only parsed as JSON and displayed |
| Speculation | `ai_suggestions` are shown separately and labelled as not findings from the papers |
| Copyright | Paper PDFs are not committed (`data/` is gitignored) |
| Provider errors | Key rotation on 429, fallback models, and cached figure descriptions so ingestion keeps working |

---

## Project layout

```
researchiq-hcl-prometheus/
├── src/researchiq/
│   ├── config.py              settings (env-driven)
│   ├── schemas.py             shared Pydantic models
│   ├── llm/
│   │   ├── groq_client.py     the only place that calls Groq (rotation, fallback, thinking off)
│   │   ├── rate_limiter.py    rolling 60-s token + request limiter, multi-key pool
│   │   └── budget.py          token estimates, image downscaling
│   ├── ingestion/             lane A: upload, parse, fallback, chunking, tables, visual, vlm,
│   │                          contextualize, embed, index, pipeline
│   ├── retrieval/             lane B: guard, rewrite, router, hybrid_search, fusion, rerank, pipeline
│   ├── generation/            lane C: context_builder, prompts, generator, validate, citations, pipeline
│   ├── api/server.py          FastAPI app (serves web/dist and /api)
│   └── ui/app.py              legacy Streamlit UI
├── web/                       React + Vite frontend
├── scripts/
│   ├── ingest.py              CLI ingestion
│   └── check_index.py         inspect what's in Qdrant
├── eval/                      lane D: run_eval, checks, judge, security, gold_set.jsonl, attacks/
├── tests/                     unit tests (no models, no API)
└── data/                      gitignored: papers/, cache/ (figures.json, rewrites.json, crops/), qdrant/
```

---

## Tests

```bash
pytest tests -q
```

These run offline with no models and no Groq calls, using synthetic PDF, DOCX, XLSX, Markdown and PNG fixtures. They cover:
- upload validation
- table splitting
- IDs and headers
- fallback parsers
- title extraction and paper matching
- the rate limiter and key pool
- request sizing
- routing
- RRF
- generation and citation checks
- eval checks

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Storage folder data/qdrant is already accessed by another instance` | Only one process can use local Qdrant. Stop the web server or Streamlit before running ingestion or the eval. |
| `0 keys loaded` | Save `.env` in the project root and check the names (`GROQ_API_KEY_1=...`) |
| Model-not-found errors from Groq | Check the exact model id in your Groq console and set `LLM_MODEL`. Calls fall back to the backup model meanwhile. |
| First question is slow | Models load on first use. The web server warms them at start-up; wait for `/api/status` to show `ready`. |
| A compare question goes to the general route | Compare needs 2 recognised papers. Name them by their short name (for example `GOLFer`, `MBA-RAG`) or title, or check the titles in `/api/papers`. |
| `Exception ignored in QdrantClient.__del__ ... sys.meta_path is None` when a script exits | Harmless: the local Qdrant client cleaning up during interpreter shutdown. Your data is saved. |
| `UnicodeEncodeError` on Windows | Set `PYTHONIOENCODING=utf-8` when piping script output |
| Removing a document | Stop the server, then run `python -c "from researchiq.ingestion.index import delete_paper; delete_paper('<paper_id>')"`. IDs are listed by `/api/papers`. |

---

## Known limitations

- **Reranker trade-off:** the int8 MiniLM reranker is about 8× faster than `bge-reranker-base`, but on this gold set it lowers MRR compared with RRF alone. Set `RERANKER_BACKEND=torch` to use the larger model, and re-tune `EVIDENCE_THRESHOLD`, whose scale differs per model.
- **Thresholds are calibrated on small samples:** the evidence gate (`-2.0`) and the semantic citation threshold come from a handful of queries. Tune them on a larger gold set.
- **Document-level injection tests** are skipped until the attack PDFs are generated and ingested.
- **Pages for Office files:** DOCX files have no page numbers, so citations use page 1 plus the section. Slides and sheets are cited by number.
- **Single-machine design:** local Qdrant is single-process and the rate limiter is per process. For multiple users, run Qdrant as a server.

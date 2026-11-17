# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Product: plain Python with official SDKs (groq, qdrant-client, fastembed, sentence-transformers, docling, pydantic). No LangChain or LlamaIndex.

Web UI: FastAPI server wrapping the Python pipeline, with a React frontend built by Vite (replaces the Streamlit app). User's choice. Minimal look requested by the user.

## Users

- Researchers and students asking questions across 8–10 research papers on one AI/ML topic.
- Hackathon judges evaluating the build. The app is demoed live, projected on a big screen: it must read clearly from a distance and respond fast.

## Product Purpose

ResearchIQ is a multimodal AI research assistant. It reads 8–10 research papers on one AI/ML topic and answers questions about them with citations (paper, page, figure/table). It compares papers, explains figures, charts, diagrams and tables, says clearly when the papers do not contain the answer, and ships with an evaluation report.

## Positioning

Every figure and table becomes a searchable text chunk (vision-model description + caption + nearby paragraph), so visual content is retrieved and cited through the same hybrid search as prose. Every citation is checked in code after generation; the system cannot invent a source.

## Operating Context

Hackathon build by a four-person team (ingestion, retrieval, generation + UI, eval + docs) over roughly four hours. Runs on the Groq free tier (about 8,000 tokens per minute), with embeddings, BM25, reranking and RRF fusion running locally on CPU.

Four pipelines: ingestion (offline, once per paper), retrieval (per question), generation (per question), evaluation (offline script).

## Capabilities and Constraints

- Ingestion: PDF validation, Docling parsing (PyMuPDF fallback), HybridChunker at ≤512 BGE tokens, figure/table crop, VLM description cached in figures.json, contextual header, deterministic UUID, dense + sparse embedding, Qdrant upsert.
- Retrieval: guard, query rewrite, rule-based router (general / compare / figure / synthesis), hybrid search (dense 30 + BM25 30), RRF fusion (k = 60), cross-encoder rerank to top 6.
- Generation: evidence gate, context builder with XML-tagged sources and token budget, Qwen 3.8 27B on Groq in JSON mode with thinking off, Pydantic validation with one retry, layered citation check (supported / weak / unsupported), AI suggestions kept separate from findings.
- UI v1 features (confirmed): ask questions with cited answers and evidence cards, upload and ingest papers with progress, paper library, conversation history in the session, figure/table image previews in evidence.
- Evaluation: 15–20 gold questions, Recall@5, MRR, ablation, RAGAS / LLM judge, report with at least two failure cases.
- Models: Qwen 3.8 27B (LLM + vision, Groq API), BAAI/bge-base-en-v1.5 (768-d, local), BM25 via fastembed (local), BAAI/bge-reranker-base (local), Qdrant (Docker or local mode).
- Only the visual path of ingestion and the generation step call Groq.
- Security: sources wrapped as untrusted data, keys in .env, magic-byte upload checks, LLM output never executed, no stack traces in user-facing errors, speculation labelled AI-generated, paper PDFs not committed.

## Evidence on Hand

- The 11-page high-level design PDF (kept outside the repo) this record is drawn from.
- `eval/papers.csv`: the 8 RAG research papers used for the demo and evaluation.
- No measured evaluation numbers exist yet; do not fabricate Recall@5, MRR or RAGAS scores.

## Product Principles

1. Every claim carries a citation that code has verified.
2. Say "not enough evidence" rather than guess.
3. Figures and tables are first-class evidence, not attachments.
4. Spend API tokens only where a local model cannot do the job.
5. Keep it simple and demonstrable.

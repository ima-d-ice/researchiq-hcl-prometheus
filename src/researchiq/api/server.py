"""HTTP API + web UI server.

  uvicorn researchiq.api.server:app --port 8000          # serves web/dist and /api
  (cd web && npm run dev)                                 # frontend dev server, proxies /api

Endpoints
  GET  /api/status                 models ready?, paper count, Groq key configured?
  GET  /api/papers                 indexed papers with page / chunk / figure / table counts
  POST /api/ask                    {question, paper_ids?} -> server-sent events: stage..., answer | error
  POST /api/ingest                 multipart file -> ingest report, or {detail} with a user-safe message
  GET  /api/figures/{paper}/{file} cropped figure / table image from the ingestion cache
"""
import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from researchiq.config import settings

log = logging.getLogger("researchiq.api")
app = FastAPI(title="ResearchIQ", docs_url="/api/docs", openapi_url="/api/openapi.json")

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_state = {"ready": False, "warm_error": ""}
_work = threading.Lock()  # one heavy job at a time: local Qdrant, CPU models, Groq limiter


# ---------------------------------------------------------------- warm-up
def _warm() -> None:
    """Load the embedding models and reranker before the first question, so it isn't slow."""
    try:
        from researchiq.ingestion.embed import embed_dense, embed_sparse
        from researchiq.retrieval.rerank import _model
        embed_dense(["warm up"]), embed_sparse(["warm up"])
        _model().predict([("warm up", "warm up")])
        _state["ready"] = True
    except Exception as e:  # keep serving; questions will load models lazily
        log.exception("warm-up failed")
        _state["warm_error"] = type(e).__name__
        _state["ready"] = True


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_warm, daemon=True).start()


# ---------------------------------------------------------------- papers
def _points() -> list[dict]:
    from researchiq.ingestion.index import qdrant
    client = qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        return []
    out, offset = [], None
    fields = ["paper_id", "paper_title", "kind", "page", "year", "source_type", "figure_label"]
    while True:
        pts, offset = client.scroll(settings.qdrant_collection, limit=512, offset=offset,
                                    with_payload=fields, with_vectors=False)
        out += [p.payload for p in pts]
        if offset is None:
            return out


@app.get("/api/papers")
def papers() -> list[dict]:
    with _work:
        rows = _points()
    by: dict[str, dict] = {}
    for r in rows:
        p = by.setdefault(r["paper_id"], {"id": r["paper_id"], "title": r.get("paper_title") or r["paper_id"],
                                          "year": r.get("year"), "type": r.get("source_type", "pdf"),
                                          "pages": 0, "chunks": 0, "figures": 0, "tables": 0})
        p["pages"] = max(p["pages"], r.get("page") or 0)
        p["chunks"] += 1
        if r.get("kind") == "figure":
            p["figures"] += 1
        elif r.get("kind") == "table":
            p["tables"] += 1
    return sorted(by.values(), key=lambda p: p["title"].lower())


@app.get("/api/status")
def status() -> dict:
    return {"ready": _state["ready"], "warm_error": _state["warm_error"],
            "groq": bool(settings.all_groq_keys), "model": settings.llm_model}


# ---------------------------------------------------------------- ask
class Ask(BaseModel):
    question: str = Field(max_length=4000)
    paper_ids: list[str] = []


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _figure_url(image_path: str | None) -> str | None:
    if not image_path:
        return None
    p = Path(image_path)
    return f"/api/figures/{p.parent.name}/{p.name}"


def _payload(gen, ret, seconds: float) -> dict:
    final = gen.final
    out = final.model_dump(mode="json")
    sources = {}
    for c in final.citations:
        sc = gen.sources.get(c.tag)
        ch = sc.chunk if sc else None
        sources[c.tag] = {**c.model_dump(mode="json"),
                          "kind": ch.kind.value if ch else "text",
                          "section": ch.section if ch else "",
                          "year": ch.year if ch else None,
                          "location": ch.location if ch else None,
                          "image_url": _figure_url(ch.image_path) if ch else None,
                          "key": f"{c.paper_id}:{c.page}:{c.figure_label or ''}"}
    out.update(sources=sources, seconds=round(seconds, 1), n_retrieved=len(ret.chunks),
               scoped=list(ret.plan.paper_ids) if ret.plan else [])
    return out


@app.post("/api/ask")
def ask(body: Ask) -> StreamingResponse:
    def stream():
        from researchiq.generation.pipeline import generate_answer
        from researchiq.retrieval.guard import InvalidQuestion
        from researchiq.retrieval.pipeline import retrieve

        start = time.perf_counter()
        yield _sse("stage", {"stage": "searching"})
        try:
            with _work:
                ret = retrieve(body.question, paper_ids=body.paper_ids or None)
                yield _sse("stage", {"stage": "writing", "route": ret.route.value, "passages": len(ret.chunks)})
                gen = generate_answer(ret.question, ret.route, ret.chunks)
            yield _sse("answer", _payload(gen, ret, time.perf_counter() - start))
        except InvalidQuestion as e:
            yield _sse("error", {"message": str(e)})
        except Exception:
            log.exception("ask failed")
            yield _sse("error", {"message": "Something went wrong while answering. Try again in a moment."})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- ingest
@app.post("/api/ingest")
def ingest(file: UploadFile) -> dict:
    from researchiq.ingestion.pipeline import ingest_file
    from researchiq.ingestion.upload import UploadError
    from researchiq.retrieval.rewrite import list_papers

    name = Path(file.filename or "upload").name  # drop any client-supplied directories
    tmp_dir = Path(tempfile.mkdtemp(prefix="riq-"))
    try:
        dest = tmp_dir / name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f, length=1 << 20)
        with _work:
            r = ingest_file(dest)
            list_papers.cache_clear()
        return {"paper_id": r.paper_id, "title": r.title, "chunks": r.n_chunks, "counts": r.counts,
                "parser": r.parser, "vlm_calls": r.vlm_calls, "groq_tokens": r.groq_tokens}
    except UploadError as e:
        raise HTTPException(400, str(e)) from None
    except Exception as e:  # IngestError messages are user-safe; anything else is not
        from researchiq.ingestion.pipeline import IngestError
        log.exception("ingest failed for %s", name)
        raise HTTPException(422, str(e) if isinstance(e, IngestError) else
                            f"{name}: could not be processed") from None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------- figures
@app.get("/api/figures/{paper_id}/{filename}")
def figure(paper_id: str, filename: str) -> FileResponse:
    root = (settings.cache_dir / "crops").resolve()
    path = (root / paper_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "Figure not found")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------- web UI
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

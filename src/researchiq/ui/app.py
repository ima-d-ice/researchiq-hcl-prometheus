"""C5 · UI: answer + evidence cards.  Run: streamlit run src/researchiq/ui/app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from researchiq.generation.pipeline import answer
from researchiq.ingestion.pipeline import ingest_file
from researchiq.ingestion.upload import ALLOWED_EXTENSIONS
from researchiq.retrieval.guard import InvalidQuestion

st.set_page_config(page_title="ResearchIQ", layout="wide")
st.title("ResearchIQ")

with st.sidebar:
    st.header("Add papers")
    files = st.file_uploader("Documents", type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
                             accept_multiple_files=True)
    if files and st.button("Ingest"):
        for f in files:
            with st.spinner(f"Ingesting {f.name}..."):
                tmp = Path(tempfile.mkdtemp()) / f.name
                tmp.write_bytes(f.getvalue())
                try:
                    r = ingest_file(tmp)
                    st.success(f"{f.name}: {r.n_chunks} chunks")
                except Exception as e:  # surface any ingestion error in the UI
                    st.error(f"{f.name}: {e}")

question = st.text_input("Ask a question about your papers")
if question:
    try:
        with st.spinner("Thinking..."):
            res = answer(question)
    except InvalidQuestion as e:
        st.error(str(e))
        st.stop()
    st.caption(f"Route: {res.route.value}")
    if res.insufficient_evidence:
        st.info("Not enough evidence in the indexed papers. " + (res.missing_info or ""))
    st.markdown(res.answer)
    for w in res.warnings:
        if w.startswith("Uncited sentence"):
            continue  # coverage note: kept in eval/logs, hidden in the UI
        st.warning(w)
    if res.ai_suggestions:
        with st.container(border=True):
            st.markdown("**AI-generated suggestions: not findings from the papers**")
            for s_ in res.ai_suggestions:
                st.markdown(f"- {s_}")

    st.subheader("Evidence")
    cols = st.columns(3)
    for i, c in enumerate(res.citations):
        with cols[i % 3].container(border=True):
            label = f" · {c.figure_label}" if c.figure_label else ""
            st.markdown(f"**[{c.tag}] {c.paper_title}**  \np.{c.page}{label}")
            st.caption(c.snippet)
            if c.status == "unsupported":
                st.error("Claim not supported by this source")
            elif c.status == "weak":
                st.warning("Weakly supported")

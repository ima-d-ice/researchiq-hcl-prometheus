"""C5 · UI: answer + evidence cards.  Run: streamlit run src/researchiq/ui/app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from researchiq.generation.pipeline import answer
from researchiq.ingestion.pipeline import ingest_pdf

st.set_page_config(page_title="ResearchIQ", layout="wide")
st.title("ResearchIQ")

with st.sidebar:
    st.header("Add papers")
    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Ingest"):
        for f in files:
            with st.spinner(f"Ingesting {f.name}..."):
                tmp = Path(tempfile.mkdtemp()) / f.name
                tmp.write_bytes(f.getvalue())
                try:
                    n = ingest_pdf(tmp)
                    st.success(f"{f.name}: {n} chunks")
                except Exception as e:  # surface any ingestion error in the UI
                    st.error(f"{f.name}: {e}")

question = st.text_input("Ask a question about your papers")
if question:
    with st.spinner("Thinking..."):
        res = answer(question)
    st.caption(f"Route: {res.route.value}")
    st.markdown(res.answer)
    for w in res.warnings:
        st.warning(w)

    st.subheader("Evidence")
    cols = st.columns(3)
    for i, c in enumerate(res.citations):
        with cols[i % 3].container(border=True):
            label = f" · {c.figure_label}" if c.figure_label else ""
            st.markdown(f"**[{c.tag}] {c.paper_title}**  \np.{c.page}{label}")
            st.caption(c.snippet)
            if not c.supported:
                st.error("Support not verified")

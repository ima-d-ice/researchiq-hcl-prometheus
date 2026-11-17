---
version: 1
slug: "web-src-app-jsx"
primary_target: "web/src/App.jsx"
related_targets: ["src/researchiq/api/server.py"]
---

# Surface brief: ResearchIQ web app

Scope: the React app in `web/` served by `src/researchiq/api/server.py`. Mode: Operate.

Audience and job: researchers asking questions across 8–10 indexed papers; demoed live to hackathon judges on a projector, so it must read from the back of a room and respond fast. Tasks: ask a question and read a cited answer with its evidence; add papers (upload, per-file progress, friendly errors); browse the paper library and scope a question to chosen papers; scroll back through the session's questions; see the cropped figure or table behind a figure citation.

States that must exist: empty library (first run), models warming up, asking (staged: searching, writing), answered, insufficient evidence (says what is missing), weak or unsupported citations, AI suggestions kept apart and labelled, generation failure, invalid or injected question, upload errors.

Constraints: no Streamlit; no invented numbers; AI suggestions never styled as findings.

## Direction contract

THESIS: Answers hang like gallery wall text and every source is an object label beside them. Refuses the chat-bubble RAG app with source chips.

OWN-WORLD: Warm white wall (#f7f6f3), near-black ink, mid grey for secondary, one burnt-orange accent (#c2410c) reserved for citation marks, the live state and focus. Hanken Grotesk throughout, self-hosted. Labels are white cards with a hairline edge: tag, paper title, an accession-style line (authors-free: paper · year · page · section/figure) in tabular figures, then the quoted passage.

STORY: The visitor sees the collection, asks, watches the answer arrive with its labels, and trusts it because every claim points to a page they can read right there.

FIRST VIEWPORT: Three columns: a 280px collection rail (brand, paper list with counts, drop zone), the centre wall (the question small and grey, the answer at about 32px, a status line, the ask field pinned to the bottom), and a 440px sources column of labels. A strict vertical rule separates answer from evidence.

FORM: gallery wall text and object labels; position 4 on the ordered list (assigned by the roll). seed:f8689432e278caf9. Raises: from the gate board, labels slide into rank order when an answer lands and a changed label stays marked; from Labanotation, one strict vertical axis that nothing crosses. Signature interaction: hovering or focusing a citation mark lights its label and the passage, and vice versa.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

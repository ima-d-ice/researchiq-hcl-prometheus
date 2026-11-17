import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ask, getJSON, uploadFile } from "./api.js";

const ROUTES = {
  general: "Direct answer",
  compare: "Comparison",
  figure: "Figure lookup",
  synthesis: "Across papers",
};
const ACCEPT = ".pdf,.docx,.pptx,.xlsx,.csv,.md,.txt,.html,.htm,.png,.jpg,.jpeg";

let nextId = 1;

export default function App() {
  const [status, setStatus] = useState(null);
  const [papers, setPapers] = useState(null);
  const [papersError, setPapersError] = useState("");
  const [scope, setScope] = useState(() => new Set());
  const [thread, setThread] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [hoverTag, setHoverTag] = useState(null);
  const [uploads, setUploads] = useState([]);
  const askRef = useRef(null);

  const loadPapers = useCallback(async () => {
    try {
      setPapers(await getJSON("/api/papers"));
      setPapersError("");
    } catch {
      setPapersError("Couldn't reach the server. Is it running?");
    }
  }, []);

  useEffect(() => {
    loadPapers();
    let timer;
    const poll = async () => {
      try {
        const s = await getJSON("/api/status");
        setStatus(s);
        if (!s.ready) timer = setTimeout(poll, 1500);
      } catch {
        setStatus({ offline: true });
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => clearTimeout(timer);
  }, [loadPapers]);

  // "/" jumps to the question field from anywhere.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "/" && !["TEXTAREA", "INPUT"].includes(document.activeElement?.tagName)) {
        e.preventDefault();
        askRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const pending = thread.some((e) => e.stage && !e.result && !e.error);
  const active = thread.find((e) => e.id === activeId) ?? thread[thread.length - 1];
  const titleOf = useMemo(() => Object.fromEntries((papers ?? []).map((p) => [p.id, p.title])), [papers]);

  const update = (id, patch) => setThread((t) => t.map((e) => (e.id === id ? { ...e, ...patch } : e)));

  async function submit(question) {
    const id = nextId++;
    const scoped = [...scope];
    const previous = thread.filter((e) => e.result).at(-1);
    const seen = new Set(Object.values(previous?.result?.sources ?? {}).map((s) => s.key));
    setThread((t) => [...t, { id, question, scope: scoped, stage: "searching" }]);
    setActiveId(id);
    setHoverTag(null);
    try {
      await ask(question, scoped, (event, data) => {
        if (event === "stage") update(id, { stage: data.stage, passages: data.passages, route: data.route });
        if (event === "answer") {
          const fresh = new Set(Object.values(data.sources).filter((s) => previous && !seen.has(s.key)).map((s) => s.tag));
          update(id, { result: data, fresh });
        }
        if (event === "error") update(id, { error: data.message });
      });
    } catch (err) {
      update(id, { error: err.message || "The server didn't respond." });
    }
  }

  async function addFiles(files) {
    const items = [...files].map((f) => ({ key: `${f.name}-${nextId++}`, file: f, state: "queued" }));
    setUploads((u) => [...items, ...u]);
    for (const it of items) {
      setUploads((u) => u.map((x) => (x.key === it.key ? { ...x, state: "working" } : x)));
      try {
        const r = await uploadFile(it.file);
        setUploads((u) => u.map((x) => (x.key === it.key ? { ...x, state: "done", report: r } : x)));
        await loadPapers();
      } catch (err) {
        setUploads((u) => u.map((x) => (x.key === it.key ? { ...x, state: "error", message: err.message } : x)));
      }
    }
  }

  return (
    <div className="app">
      <Rail
        papers={papers}
        papersError={papersError}
        scope={scope}
        setScope={setScope}
        uploads={uploads}
        onFiles={addFiles}
        clearUploads={() => setUploads((u) => u.filter((x) => x.state === "working" || x.state === "queued"))}
        status={status}
      />
      <Wall
        thread={thread}
        active={active}
        setActiveId={setActiveId}
        hoverTag={hoverTag}
        setHoverTag={setHoverTag}
        papers={papers}
        scope={scope}
        titleOf={titleOf}
        clearScope={() => setScope(new Set())}
        onAsk={submit}
        pending={pending}
        askRef={askRef}
      />
      <Sources entry={active} hoverTag={hoverTag} setHoverTag={setHoverTag} />
    </div>
  );
}

/* ------------------------------------------------------------------ collection rail */
function Rail({ papers, papersError, scope, setScope, uploads, onFiles, clearUploads, status }) {
  const toggle = (id) =>
    setScope((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  return (
    <aside className="rail" aria-label="Paper collection">
      <header className="brand">
        <span className="brand-name">ResearchIQ</span>
        <span className="brand-sub">Answers cited to the page</span>
      </header>

      <section className="collection" aria-labelledby="collection-h">
        <div className="section-head">
          <h2 id="collection-h">Collection</h2>
          <span className="count">{papers ? `${papers.length} ${papers.length === 1 ? "paper" : "papers"}` : ""}</span>
        </div>
        {scope.size > 0 && (
          <p className="scope-note">
            Asking {scope.size} of {papers?.length ?? 0}.{" "}
            <button type="button" className="text-btn" onClick={() => setScope(new Set())}>
              Ask all
            </button>
          </p>
        )}
        {papersError && <p className="rail-error" role="alert">{papersError}</p>}
        {papers === null && !papersError && <p className="muted">Loading papers…</p>}
        {papers?.length === 0 && <p className="muted">No papers yet. Add some below.</p>}
        <ul className="papers">
          {papers?.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                className="paper"
                aria-pressed={scope.has(p.id)}
                onClick={() => toggle(p.id)}
                title={scope.has(p.id) ? "Remove from this question's scope" : "Ask only selected papers"}
              >
                <span className="paper-title">{p.title}</span>
                <span className="paper-meta">
                  {[p.year, `${p.pages} pp`, p.figures && `${p.figures} fig.`, p.tables && `${p.tables} tab.`]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <DropZone onFiles={onFiles} />

      {uploads.length > 0 && (
        <section className="uploads" aria-label="Uploads" aria-live="polite">
          <ul>
            {uploads.map((u) => (
              <li key={u.key} className={`upload is-${u.state}`}>
                <span className="upload-name">{u.file.name}</span>
                <span className="upload-state">
                  {u.state === "queued" && "Waiting"}
                  {u.state === "working" && "Reading and indexing…"}
                  {u.state === "done" && `${u.report.chunks} passages indexed`}
                  {u.state === "error" && u.message}
                </span>
              </li>
            ))}
          </ul>
          {uploads.some((u) => u.state === "done" || u.state === "error") && (
            <button type="button" className="text-btn" onClick={clearUploads}>
              Clear finished
            </button>
          )}
        </section>
      )}

      <footer className="system" aria-live="polite">
        <span className={`dot ${status?.ready ? "ok" : status?.offline ? "bad" : "wait"}`} aria-hidden="true" />
        {status?.offline
          ? "Server offline"
          : !status
          ? "Connecting…"
          : !status.ready
          ? "Loading search models…"
          : !status.groq
          ? "No Groq key: answers unavailable"
          : "Ready"}
      </footer>
    </aside>
  );
}

function DropZone({ onFiles }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  return (
    <div
      className={`drop ${over ? "is-over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files);
      }}
    >
      <button type="button" className="drop-btn" onClick={() => input.current?.click()}>
        Add papers
      </button>
      <span className="drop-hint">or drop PDFs, Word, PowerPoint or Excel files here</span>
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          if (e.target.files.length) onFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ the wall */
function Wall({ thread, active, setActiveId, hoverTag, setHoverTag, papers, scope, titleOf, clearScope, onAsk, pending, askRef }) {
  const scroller = useRef(null);
  const last = thread.at(-1);

  useEffect(() => {
    const el = scroller.current?.querySelector(`[data-entry="${last?.id}"]`);
    el?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }, [last?.id, last?.result, last?.error]);

  return (
    <main className="wall">
      <div className="wall-scroll" ref={scroller}>
        {thread.length === 0 ? (
          <Welcome papers={papers} />
        ) : (
          thread.map((e) => (
            <Entry
              key={e.id}
              entry={e}
              isActive={e.id === active?.id}
              onActivate={() => setActiveId(e.id)}
              hoverTag={e.id === active?.id ? hoverTag : null}
              setHoverTag={setHoverTag}
              titleOf={titleOf}
              papers={papers}
            />
          ))
        )}
      </div>
      <AskBar onAsk={onAsk} pending={pending} askRef={askRef} scope={scope} titleOf={titleOf} clearScope={clearScope} papers={papers} />
    </main>
  );
}

function Welcome({ papers }) {
  if (papers && papers.length === 0) {
    return (
      <section className="welcome">
        <h1>Add papers to begin.</h1>
        <p>Use Add papers in the collection on the left. ResearchIQ reads your papers, including their figures and tables, and answers questions with a citation to the page for every claim.</p>
      </section>
    );
  }
  return (
    <section className="welcome">
      <h1>Ask anything about {papers ? `these ${papers.length} papers` : "your papers"}.</h1>
      <p>Every claim in an answer points to the page it came from, and the passage sits beside it.</p>
      <ul className="tips">
        <li>Name a figure or table, like “What does Table 2 of GRAG show?”, to look it up directly.</li>
        <li>Name two papers and ask how they differ to get a side-by-side comparison.</li>
        <li>Select papers in the collection to limit a question to them.</li>
      </ul>
    </section>
  );
}

function Entry({ entry, isActive, onActivate, hoverTag, setHoverTag, titleOf }) {
  const r = entry.result;
  const scopeText = entry.scope.length ? entry.scope.map((id) => titleOf[id] || id).join(", ") : null;

  return (
    <article className={`entry ${isActive ? "is-active" : ""}`} data-entry={entry.id} onClick={isActive ? undefined : onActivate}>
      <p className="question">
        {entry.question}
        {scopeText && <span className="question-scope"> · in {scopeText}</span>}
      </p>

      {!r && !entry.error && <Progress entry={entry} />}

      {entry.error && (
        <p className="answer answer-error" role="alert">
          {entry.error}
        </p>
      )}

      {r && (
        <>
          {r.insufficient_evidence && !r.claims?.length ? (
            <div className="answer-none">
              <p className="answer">The papers don't answer this.</p>
              {r.missing_info && <p className="missing">{r.missing_info}</p>}
            </div>
          ) : (
            <AnswerText text={r.answer} sources={r.sources} hoverTag={hoverTag} setHoverTag={isActive ? setHoverTag : () => {}} />
          )}
          <p className="meta">
            {[ROUTES[r.route] ?? r.route, sourcesLine(r), supportLine(r), `${r.seconds} s`].filter(Boolean).join(" · ")}
            {!isActive && Object.keys(r.sources).length > 0 && (
              <>
                {" · "}
                <button type="button" className="text-btn" onClick={onActivate}>
                  Show its sources
                </button>
              </>
            )}
          </p>
          {r.insufficient_evidence && r.claims?.length > 0 && r.missing_info && (
            <p className="missing">Not covered: {r.missing_info}</p>
          )}
          {r.ai_suggestions?.length > 0 && (
            <aside className="speculation" aria-label="AI-generated suggestions">
              <h3>AI-generated suggestions, not findings from the papers</h3>
              <ul>
                {r.ai_suggestions.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </aside>
          )}
        </>
      )}
    </article>
  );
}

function sourcesLine(r) {
  const n = Object.keys(r.sources).length;
  return n ? `${n} ${n === 1 ? "source" : "sources"}` : null;
}

function supportLine(r) {
  const s = Object.values(r.sources);
  if (!s.length) return null;
  const weak = s.filter((x) => x.status === "weak").length;
  const bad = s.filter((x) => x.status === "unsupported").length;
  if (!weak && !bad) return "all citations supported";
  return [bad && `${bad} unsupported`, weak && `${weak} weakly supported`].filter(Boolean).join(", ");
}

function Progress({ entry }) {
  const text =
    entry.stage === "writing"
      ? `Reading ${entry.passages ?? ""} passages and writing the answer…`.replace("  ", " ")
      : entry.scope.length
      ? `Searching ${entry.scope.length} selected ${entry.scope.length === 1 ? "paper" : "papers"}…`
      : "Searching the collection…";
  return (
    <div className="progress" role="status">
      <span className="progress-rule" aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}

function AnswerText({ text, sources, hoverTag, setHoverTag }) {
  const paragraphs = text.split(/\n{2,}/);
  return (
    <div className="answer">
      {paragraphs.map((para, i) => (
        <p key={i}>
          {para.split(/(\[S\d+\])/g).map((part, j) => {
            const tag = /^\[(S\d+)\]$/.exec(part)?.[1];
            if (!tag) return part;
            const s = sources[tag];
            if (!s) return null; // unknown tags are already flagged by the citation check
            return (
              <button
                key={j}
                type="button"
                className={`cite ${hoverTag === tag ? "is-lit" : ""} is-${s.status}`}
                aria-label={`Source ${tag}: ${s.paper_title}, page ${s.page}`}
                onMouseEnter={() => setHoverTag(tag)}
                onMouseLeave={() => setHoverTag(null)}
                onFocus={() => setHoverTag(tag)}
                onBlur={() => setHoverTag(null)}
                onClick={(e) => {
                  e.stopPropagation();
                  document.getElementById(`label-${tag}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
                }}
              >
                {tag}
              </button>
            );
          })}
        </p>
      ))}
    </div>
  );
}

function AskBar({ onAsk, pending, askRef, scope, titleOf, clearScope, papers }) {
  const [value, setValue] = useState("");
  const empty = papers?.length === 0;
  const send = () => {
    const q = value.trim();
    if (!q || pending || empty) return;
    onAsk(q);
    setValue("");
  };
  const names = [...scope].map((id) => titleOf[id] || id);

  return (
    <form
      className="ask"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <label htmlFor="question" className="ask-scope">
        {names.length ? (
          <>
            Asking {names.length === 1 ? names[0] : `${names.length} papers: ${names.join(", ")}`}{" "}
            <button type="button" className="text-btn" onClick={clearScope}>
              Ask all
            </button>
          </>
        ) : (
          "Ask across the whole collection"
        )}
      </label>
      <div className="ask-row">
        <textarea
          id="question"
          ref={askRef}
          rows={1}
          value={value}
          placeholder={empty ? "Add papers first" : "Ask about your papers"}
          disabled={empty}
          maxLength={1000}
          onChange={(e) => {
            setValue(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button type="submit" className="ask-btn" disabled={!value.trim() || pending || empty}>
          {pending ? "Answering…" : "Ask"}
        </button>
      </div>
    </form>
  );
}

/* ------------------------------------------------------------------ source labels */
function Sources({ entry, hoverTag, setHoverTag }) {
  const r = entry?.result;
  const list = r ? Object.values(r.sources).sort((a, b) => num(a.tag) - num(b.tag)) : [];
  const [read, setRead] = useState(() => new Set());
  useEffect(() => setRead(new Set()), [entry?.id]);

  return (
    <aside className="sources" aria-label="Sources">
      <div className="section-head">
        <h2>Sources</h2>
        {list.length > 0 && <span className="count">{list.length}</span>}
      </div>
      {!entry && <p className="muted">Each cited passage appears here as a label: paper, page, figure or table, and the words it rests on.</p>}
      {entry && !r && !entry.error && <p className="muted">Waiting for the answer…</p>}
      {r && list.length === 0 && <p className="muted">No passage was relevant enough to cite.</p>}
      <ol className="labels" key={entry?.id}>
        {list.map((s, i) => (
          <Label
            key={s.tag}
            s={s}
            index={i}
            lit={hoverTag === s.tag}
            fresh={entry.fresh?.has(s.tag) && !read.has(s.tag)}
            onEnter={() => {
              setHoverTag(s.tag);
              setRead((x) => new Set(x).add(s.tag));
            }}
            onLeave={() => setHoverTag(null)}
          />
        ))}
      </ol>
    </aside>
  );
}

const num = (tag) => parseInt(tag.slice(1), 10);

function Label({ s, index, lit, fresh, onEnter, onLeave }) {
  const [open, setOpen] = useState(false);
  const where = [s.year, `p. ${s.page}`, s.figure_label || s.location || (s.section && `§ ${s.section}`)].filter(Boolean);
  return (
    <li
      id={`label-${s.tag}`}
      className={`label ${lit ? "is-lit" : ""} ${fresh ? "is-fresh" : ""}`}
      style={{ "--i": index }}
      tabIndex={0}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onFocus={onEnter}
      onBlur={onLeave}
    >
      <div className="label-top">
        <span className="label-tag">{s.tag}</span>
        {fresh && <span className="label-new">New source</span>}
        {s.status !== "supported" && (
          <span className={`label-status is-${s.status}`}>{s.status === "weak" ? "Weakly supported" : "Doesn't support the claim"}</span>
        )}
      </div>
      <h3 className="label-title">{s.paper_title}</h3>
      <p className="label-where">{where.join(" · ")}</p>
      {s.image_url && (
        <button type="button" className={`label-figure ${open ? "is-open" : ""}`} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          <img src={s.image_url} alt={`${s.figure_label || "Figure"} from ${s.paper_title}, page ${s.page}`} loading="lazy" />
          <span className="label-figure-hint">{open ? "Show smaller" : "Show larger"}</span>
        </button>
      )}
      <blockquote className="label-quote">{s.snippet}</blockquote>
    </li>
  );
}

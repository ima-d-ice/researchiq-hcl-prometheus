// Thin client for the FastAPI server in src/researchiq/api/server.py.

export async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function uploadFile(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/ingest", { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${file.name}: upload failed (${res.status})`);
  return data;
}

// POST /api/ask streams server-sent events: stage*, then answer | error.
export async function ask(question, paperIds, onEvent, signal) {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, paper_ids: paperIds }),
    signal,
  });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    const detail = Array.isArray(data.detail) ? "The question is too long." : data.detail;
    throw new Error(detail || `The server returned ${res.status}.`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (event && data) onEvent(event, JSON.parse(data));
    }
  }
}

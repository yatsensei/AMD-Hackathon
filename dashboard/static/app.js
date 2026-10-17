/* Sentry app: replays the scored factory stream as a live etch bay.
   Plain JS, no build step. All data comes from /api (dashboard/server.py). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => {
  if (x == null) return "—";
  const v = (x * 100).toFixed(d);
  return x < 1 && v === (100).toFixed(d) ? `>${(100 - 10 ** -d).toFixed(d)}%` : `${v}%`;  // never round up to a false 100%
};
const usd = (x) => (x == null ? "—" : x >= 1e6 ? `$${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `$${Math.round(x / 1e3)}k` : `$${Math.round(x)}`);
const fmtTime = (iso) => new Date(iso).toLocaleString("en-AU", { weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
const hhmm = (iso) => new Date(iso).toLocaleTimeString("en-AU", { hour: "2-digit", minute: "2-digit", hour12: false });
const dayTime = (iso) => new Date(iso).toLocaleString("en-AU", { weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false });
const human = (s) => ({ low_confidence: "low confidence", confusable_pair: "could be a similar pattern", possible_defect: "possible defect (safety rule)" }[s] || s);
const ACTION = { hold: "Lots held", ticket: "Maintenance ticket opened", dismiss: "Alert dismissed" };

/** Escape then apply a small safe Markdown subset (no raw HTML passthrough). */
function renderMd(src) {
  let s = esc(src).replace(/\r\n/g, "\n");
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => `<pre class="md-pre"><code>${code.trim()}</code></pre>`);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,]|$)/g, "$1<em>$2</em>");
  s = s.replace(/^### (.+)$/gm, "<h4>$1</h4>");
  s = s.replace(/^## (.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/^# (.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/^(?:- |\* )(.+)$/gm, "<li>$1</li>");
  s = s.replace(/(?:<li>[\s\S]*?<\/li>\n?)+/g, (block) => `<ul>${block.replace(/\n/g, "")}</ul>`);
  s = s.replace(/^(?:\d+\. )(.+)$/gm, "<li>$1</li>");
  s = s.replace(/\n{2,}/g, "</p><p>");
  s = s.replace(/\n/g, "<br>");
  return `<div class="md"><p>${s}</p></div>`;
}

const ICON = {
  ok: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
  alert: '<svg viewBox="0 0 24 24"><path d="M12 9v4M12 16.5v.5"/><path d="M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  review: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5M11 8v3M11 14v.5"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z" class="fill"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" class="fill"/></svg>',
  chat: '<svg viewBox="0 0 24 24"><path d="M4 5h16v11H9l-5 4z"/><path d="M8 9.5h8M8 12.5h5"/></svg>',
};

const PAGES = {
  overview: ["Overview", "Live view of the etch bay"],
  alerts: ["Alerts", "Excursions the alert rule has raised, with the evidence behind each"],
  tools: ["Tools", "Defect history and pattern mix for each etch tool"],
  lots: ["Lots", "Every lot on the line, its defect rate and status"],
  review: ["Review queue", "Wafers the model sent to a person"],
  decisions: ["Decisions", "Audit log of engineer actions"],
};

const S = {
  meta: null, t: 0, playing: false, timer: null, state: null, wafers: [], seen: new Set(), inflight: false,
  page: "overview", focusTool: null, openLot: null, reviewed: new Map(), user: null,
  filter: { q: "", tool: "", pattern: "", status: "all" },
};

const lineMeta = () => S.meta?.model?.line || {
  classifier: S.meta?.model?.stream_classifier || "cnn",
  model_version: S.meta?.model?.stream_model_version || "",
  calibrated: (S.meta?.model?.stream_classifier || "cnn") === "cnn",
  review_rules_validated: (S.meta?.model?.stream_classifier || "cnn") === "cnn",
  drives_alerts: true,
  label: S.meta?.model?.stream_scored_by || "cnn",
};

/* ---------------------------------------------------------------- API */
async function api(path, opts) {
  const r = await fetch(`/api/${path}`, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok && r.status !== 503) throw new Error(body.detail || r.statusText);
  return body;
}
const asOf = () => S.meta.timestamps[S.t];

/* ---------------------------------------------------------------- playback */
function setPlaying(on) {
  S.playing = on;
  $("#btn-play").innerHTML = on ? ICON.pause : ICON.play;
  clearInterval(S.timer);
  if (on) S.timer = setInterval(() => step(+$("#speed").value), 900);
}
function step(n) {
  const last = S.meta.timestamps.length - 1;
  if (S.t >= last) return setPlaying(false);
  S.t = Math.min(last, S.t + n);
  refresh();
}
function nextLot() {
  const ts = S.meta.timestamps;
  let i = S.t;
  while (i < ts.length - 1 && new Date(ts[i + 1]) - new Date(ts[i]) < 30 * 60e3) i++;
  if (i === S.t && i < ts.length - 1) { i++; while (i < ts.length - 1 && new Date(ts[i + 1]) - new Date(ts[i]) < 30 * 60e3) i++; }
  S.t = i;
  refresh();
}

async function refresh() {
  if (S.inflight) return;
  S.inflight = true;
  try {
    const q = encodeURIComponent(asOf());
    const [st, ws] = await Promise.all([api(`state?as_of=${q}`), api(`wafers?as_of=${q}`)]);
    S.state = st;
    S.wafers = ws.wafers;
    $("#clock").textContent = fmtTime(asOf());
    $("#replay-progress").style.width = `${(100 * S.t / (S.meta.timestamps.length - 1)).toFixed(2)}%`;
    renderAll();
  } catch (e) {
    console.error(e);
  } finally {
    S.inflight = false;
  }
}

/* ---------------------------------------------------------------- shared helpers */
const reviewPending = () => S.wafers.filter((w) => w.needs_review && !S.reviewed.has(w.uid));
const openAlerts = () => (S.state?.alerts || []).filter((a) => !a.decision);

/** A lot is affected by an alert when it is in the alert window, or ran on that tool after the alert fired. */
function lotStatus(l) {
  for (const a of S.state?.alerts || []) {
    const affected = a.tool_id === l.tool && ((a.lot_ids || []).includes(l.lot) || l.start >= a.fired_at);
    if (!affected) continue;
    if (a.decision?.action === "hold") return "held";
    if (!a.decision) return (a.lot_ids || []).includes(l.lot) ? "alert" : "risk";
  }
  return "ok";
}
const STATUS_PILL = {
  held: `<span class="status held">Held</span>`,
  alert: `<span class="status alert">${ICON.alert}In alert</span>`,
  risk: `<span class="status risk">${ICON.alert}At risk</span>`,
  ok: `<span class="status ok">${ICON.ok}Normal</span>`,
};

function kpi(label, value, sub, alert = false) {
  return `<div class="kpi${alert ? " alert" : ""}"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${esc(sub)}</div></div>`;
}

function feedStatus(w) {
  const clf = w.classifier || lineMeta().classifier;
  if (w.needs_review) {
    return `<span class="flag" title="${esc((w.review_reasons || []).map(human).join(", "))}">${ICON.review}Review</span>`;
  }
  if (clf === "yolo") {
    return `<span class="ok-text" title="Uncalibrated YOLO-cls score; no validated automatic-acceptance policy">Uncalibrated</span>`;
  }
  return `<span class="ok-text" title="Passed calibrated CNN review gates">Accepted</span>`;
}

const thumb = (uid, size = 64) => `/api/wafer/${uid}/image.png?size=${size}`;

/* ---------------------------------------------------------------- render */
function renderAll() {
  renderNav();
  renderOverview();
  ({ alerts: renderAlerts, tools: renderTools, lots: renderLots, review: renderReview, decisions: renderDecisions }[S.page] || (() => {}))();
}

function renderNav() {
  const na = openAlerts().length, nr = reviewPending().length;
  $("#nav-alerts").hidden = !na; $("#nav-alerts").textContent = na;
  $("#nav-review").hidden = !nr; $("#nav-review").textContent = nr;
}

/* Overview -------------------------------------------------------- */
function renderOverview() {
  const st = S.state, k = st.kpis, line = lineMeta();
  const reviewSub = line.review_rules_validated ? "model unsure, sent to a person" : "YOLO has no validated review policy";
  $("#kpis").innerHTML = !k ? "" : [
    kpi("Wafers inspected", k.wafers.toLocaleString(), `${k.lots} lots`),
    kpi("Defective wafers", pct(k.defect_rate), `${k.defects} wafers`),
    kpi("Sent to engineer", k.needs_review.toLocaleString(), reviewSub),
    kpi("Open alerts", k.alerts_open, `${k.alerts_total} raised so far`, k.alerts_open > 0),
    kpi("Exposure since alerts", usd(k.exposure_usd), `at ${usd(S.meta.wafer_value_usd)} per wafer (assumption)`, k.exposure_usd > 0),
  ].join("");
  $("#tools").innerHTML = st.tools.map(toolCard).join("");
  renderFeed();

  const alerts = (st.alerts || []).slice().reverse();
  $("#alerts-mini").innerHTML = alerts.length ? alerts.slice(0, 3).map((a) => `
    <a class="mini-alert" href="#alerts">
      <div class="row"><b>${esc(a.pattern)} on ${esc(a.tool_id)}</b>${a.decision ? `<span class="badge done">${esc(ACTION[a.decision.action])}</span>` : `<span class="badge">${ICON.alert}Open</span>`}</div>
      <div class="kv"><span>Fired <strong>${hhmm(a.fired_at)}</strong></span><span>Lots since <strong>${a.lots_since_alert}</strong></span><span>Exposure <strong>${usd(a.exposure_usd)}</strong></span></div>
    </a>`).join("") : `<div class="empty">No alerts. Every tool is within its normal defect mix.</div>`;
  $("#decisions-mini").innerHTML = decisionItems(st.decisions.slice(0, 4));
}

function decisionItems(list) {
  return list.length ? list.map((d) => `
    <li><span class="ldot ${esc(d.action)}"></span><div><b>${esc(ACTION[d.action])}</b> · ${esc(d.tool_id)} · ${esc(d.pattern)}
      <div class="when">${esc(d.by)} · line time ${hhmm(d.as_of)}</div></div></li>`).join("")
    : `<li class="empty">No decisions yet. The copilot recommends; an engineer decides.</li>`;
}

function toolCard(t) {
  const vals = t.per_lot_defects;
  const W = 220, H = 44, n = Math.max(vals.length, 6), bw = W / n;
  const max = Math.max(5, ...vals);
  const color = t.status === "alert" ? "var(--red)" : "var(--bar)";
  const bars = vals.map((v, i) => {
    const h = v === 0 ? 1.5 : Math.max(3, (v / max) * (H - 4));
    return `<rect x="${(i * bw + 1).toFixed(1)}" y="${(H - h).toFixed(1)}" width="${Math.max(2, bw - 3).toFixed(1)}" height="${h.toFixed(1)}" rx="1.5" fill="${v ? color : "#e7e5e4"}" stroke="none"><title>${esc(t.per_lot_ids[i])}: ${v} defective</title></rect>`;
  }).join("");
  const status = t.status === "alert" ? STATUS_PILL.alert.replace("In alert", "Alert") : STATUS_PILL.ok;
  return `<div class="tool" data-tool="${esc(t.tool_id)}" title="Open ${esc(t.tool_id)}">
    <div class="tool-head"><span class="tool-name">${esc(t.tool_id)}</span>${status}</div>
    <div class="tool-stats"><div><b>${t.wafers}</b>wafers</div><div><b>${pct(t.defect_rate, 0)}</b>defective</div><div><b>${t.defects_24h}</b>defects 24 h</div></div>
    <svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="defective wafers per lot">${bars}</svg>
    <div class="tool-foot"><span>${t.top_pattern_24h ? `Top 24 h: <b>${esc(t.top_pattern_24h)}</b>` : "No defects in 24 h"}</span><span>${esc(t.current_lot || "")}</span></div>
  </div>`;
}

/* Inspection feed with filters ------------------------------------ */
function setupFilters() {
  $("#f-tool").innerHTML = `<option value="">All tools</option>` + S.meta.tools.map((t) => `<option>${esc(t)}</option>`).join("");
  $("#f-pattern").innerHTML = `<option value="">All patterns</option>` + S.meta.classes.map((c) => `<option value="${esc(c)}">${c === "none" ? "No pattern" : esc(c)}</option>`).join("");
  const set = (k, v) => { S.filter[k] = v; renderFeed(); };
  $("#f-q").addEventListener("input", (e) => set("q", e.target.value.trim().toLowerCase()));
  $("#f-tool").addEventListener("change", (e) => set("tool", e.target.value));
  $("#f-pattern").addEventListener("change", (e) => set("pattern", e.target.value));
  $("#f-status").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-v]");
    if (!b) return;
    $$("#f-status button").forEach((x) => x.classList.toggle("on", x === b));
    set("status", b.dataset.v);
  });
  $("#f-clear").addEventListener("click", () => {
    S.filter = { q: "", tool: "", pattern: "", status: "all" };
    $("#f-q").value = ""; $("#f-tool").value = ""; $("#f-pattern").value = "";
    $$("#f-status button").forEach((x) => x.classList.toggle("on", x.dataset.v === "all"));
    renderFeed();
  });
}

function renderFeed() {
  const f = S.filter;
  const rows = S.wafers.filter((w) =>
    (!f.tool || w.tool_id === f.tool) &&
    (!f.pattern || w.pattern === f.pattern) &&
    (f.status === "all" || (f.status === "defect" ? w.pattern !== "none" : w.needs_review)) &&
    (!f.q || w.lot_id.toLowerCase().includes(f.q) || String(w.uid).includes(f.q) || `w${w.wafer}` === f.q));
  const active = f.q || f.tool || f.pattern || f.status !== "all";
  $("#f-clear").hidden = !active;
  $("#f-tool").classList.toggle("on", !!f.tool);
  $("#f-pattern").classList.toggle("on", !!f.pattern);
  const shown = rows.slice(0, 120);
  const fresh = [];
  $("#feed").innerHTML = shown.length ? shown.map((w) => {
    const isNew = !S.seen.has(w.uid);
    if (isNew) fresh.push(w.uid);
    return feedRow(w, isNew && S.seen.size > 0 && !active);
  }).join("") : `<tr><td colspan="7" class="empty">No wafers match these filters.</td></tr>`;
  fresh.forEach((u) => S.seen.add(u));
  $("#feed-count").textContent = active
    ? `${rows.length.toLocaleString()} of ${S.wafers.length.toLocaleString()} wafers${rows.length > shown.length ? ` · newest ${shown.length} shown` : ""}`
    : `${S.wafers.length.toLocaleString()} wafers · newest first${S.wafers.length > shown.length ? ` · ${shown.length} shown` : ""}`;
}

function feedRow(w, isNew) {
  const defect = w.pattern !== "none";
  return `<tr data-uid="${w.uid}" class="${isNew ? "is-new" : ""}">
    <td>${hhmm(w.timestamp)}</td><td>${esc(w.lot_id)}</td><td>${esc(w.tool_id)}</td>
    <td><img class="thumb" loading="lazy" src="${thumb(w.uid)}" alt=""></td>
    <td><span class="pattern ${defect ? "defect" : "none"}">${defect ? esc(w.pattern) : "No pattern"}</span></td>
    <td><div class="conf"><div class="conf-bar"><span style="width:${(w.confidence * 100).toFixed(1)}%"></span></div><span class="conf-val">${pct(w.confidence, 0)}</span></div></td>
    <td>${feedStatus(w)}</td></tr>`;
}

/* Alerts ---------------------------------------------------------- */
function renderAlerts() {
  const list = (S.state.alerts || []).slice().sort((a, b) => (!!a.decision - !!b.decision) || (a.fired_at < b.fired_at ? 1 : -1));
  $("#alerts").innerHTML = list.length ? list.map(alertCard).join("")
    : `<div class="card"><div class="empty">No alerts yet. An alert fires when one tool produces the same pattern 3 times within 24 hours.</div></div>`;
}

function alertCard(a) {
  const rc = a.root_cause || {};
  const d = a.decision;
  const line = lineMeta();
  const actions = d
    ? `<div class="decided-note">${ICON.ok}<span><b>${esc(ACTION[d.action])}</b> by ${esc(d.by)} at line time ${hhmm(d.as_of)}</span></div>`
    : `<div class="actions">
        <button class="btn red" data-act="hold">Hold lots</button>
        <button class="btn" data-act="ticket">Open ticket</button>
        <button class="btn" data-act="dismiss">Dismiss</button>
        <button class="btn ghost" data-act="ask">${ICON.chat}Ask copilot</button>
      </div>`;
  const also = (a.also_seen || []).length ? ` · also seen on this tool: ${a.also_seen.map((x) => esc(x.pattern)).join(", ")}` : "";
  const uids = (a.uids || []).slice(-12);
  const byUid = new Map(S.wafers.map((w) => [w.uid, w]));
  return `<article class="alert-card${d ? " decided" : ""}" data-tool="${esc(a.tool_id)}" data-fired="${esc(a.fired_at)}" data-pattern="${esc(a.pattern)}">
    <div class="alert-top"><span class="alert-title">${esc(a.pattern)} on ${esc(a.tool_id)}</span>${d ? `<span class="badge done">Closed</span>` : `<span class="badge">${ICON.alert}Excursion</span>`}</div>
    <div class="alert-meta">${a.count_in_window} wafers in ${a.window_hours} h · fired ${dayTime(a.fired_at)} in ${esc(a.fired_lot_id)} · line model ${esc(line.label)}${also}</div>
    <div class="alert-body">
      <div>
        <div class="alert-grid">
          <div><span>Lots run since alert</span><b>${a.lots_since_alert}</b></div>
          <div><span>Exposure since alert</span><b>${usd(a.exposure_usd)}</b></div>
        </div>
        <div class="cause-label">Likely cause · triage, not diagnosis</div>
        <div class="cause">${esc(rc.cause || "—")}</div>
        <div class="cause-label">Recommended check</div>
        <div class="cause">${esc(rc.recommended_action || "—")}</div>
        ${rc.source ? `<div class="source">Source: ${esc(rc.source)}</div>` : ""}
      </div>
      <div>
        <div class="evidence-label">Evidence · ${uids.length} wafer${uids.length === 1 ? "" : "s"} in the window</div>
        <div class="evidence">${uids.map((u) => {
          const w = byUid.get(u);
          return `<button data-uid="${u}" title="Open wafer"><img loading="lazy" src="${thumb(u)}" alt="${esc(a.pattern)} wafer"><small>${w ? `${esc(w.lot_id.replace("LOT-", "L"))} W${w.wafer}` : u}</small></button>`;
        }).join("")}</div>
      </div>
    </div>
    ${actions}
  </article>`;
}

/* Tools ----------------------------------------------------------- */
function renderTools() {
  $("#tool-pages").innerHTML = S.state.tools.map((t) => {
    const ws = S.wafers.filter((w) => w.tool_id === t.tool_id);
    const defects = ws.filter((w) => w.pattern !== "none");
    const mix = Object.entries(defects.reduce((m, w) => ((m[w.pattern] = (m[w.pattern] || 0) + 1), m), {})).sort((a, b) => b[1] - a[1]);
    const mixMax = Math.max(1, ...mix.map((x) => x[1]));
    const review = ws.filter((w) => w.needs_review).length;
    const status = t.status === "alert" ? STATUS_PILL.alert.replace("In alert", "Alert") : STATUS_PILL.ok;
    return `<div class="card tool-page${S.focusTool === t.tool_id ? " focus" : ""}" id="tool-${esc(t.tool_id)}">
      <div class="card-head"><h2 class="mono">${esc(t.tool_id)}</h2>${status}</div>
      <div class="tp-body">
        <div class="tp-stats">
          <div><span>Wafers</span><b>${t.wafers}</b></div>
          <div><span>Defective</span><b>${pct(t.defect_rate, 0)}</b></div>
          <div><span>Defects 24 h</span><b>${t.defects_24h}</b></div>
          <div><span>Sent to review</span><b>${review}</b></div>
        </div>
        <div><div class="chart-label"><span>Defective wafers per lot</span><span>${t.per_lot_ids.length} lots</span></div>${lotChart(t)}</div>
        <div><div class="chart-label"><span>Defect pattern mix</span><span>${defects.length} defective wafers</span></div>
          <div class="mix">${mix.length ? mix.map(([p, n], i) => `<div class="mix-row${i === 0 && t.status === "alert" ? " lead" : ""}"><span>${esc(p)}</span><div class="mix-track"><span style="width:${(100 * n / mixMax).toFixed(1)}%"></span></div><span class="v">${n}</span></div>`).join("") : `<p class="muted" style="margin:0">No defective wafers on this tool yet.</p>`}</div>
        </div>
      </div>
    </div>`;
  }).join("");
  if (S.focusTool) {
    document.getElementById(`tool-${S.focusTool}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    const f = S.focusTool;
    setTimeout(() => { if (S.focusTool === f) { S.focusTool = null; $(`#tool-${f}`)?.classList.remove("focus"); } }, 1800);
  }
}

function lotChart(t) {
  const vals = t.per_lot_defects, ids = t.per_lot_ids;
  if (!vals.length) return `<p class="muted" style="margin:0">No lots yet.</p>`;
  const W = 520, H = 110, L = 22, B = 18, top = 6;
  // One scale for every tool (0 to a full lot) so bars compare across tools.
  const max = S.meta.wafers_per_lot || 25, slots = Math.max(vals.length, 8), bw = (W - L) / slots;
  const y = (v) => top + (H - B - top) * (1 - v / max);
  const ticks = [0, 10, 20].filter((v) => v <= max);
  const grid = ticks.map((v) => `<line class="grid" x1="${L}" x2="${W}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/><text x="${L - 6}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${v}</text>`).join("");
  const color = t.status === "alert" ? "var(--red)" : "var(--bar)";
  const bars = vals.map((v, i) => {
    const x = L + i * bw + 3, h = Math.max(v ? 3 : 1.5, (H - B) - y(v));
    return `<rect x="${x.toFixed(1)}" y="${(H - B - h).toFixed(1)}" width="${Math.max(4, bw - 6).toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${v ? color : "#e7e5e4"}" stroke="none"><title>${esc(ids[i])}: ${v} defective of 25</title></rect>
      <text x="${(x + (bw - 6) / 2).toFixed(1)}" y="${H - 4}" text-anchor="middle">${esc(ids[i].replace("LOT-", ""))}</text>`;
  }).join("");
  return `<svg class="lotchart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Defective wafers per lot on ${esc(t.tool_id)}">${grid}${bars}</svg>`;
}

/* Lots ------------------------------------------------------------ */
function renderLots() {
  const lots = new Map();
  for (const w of S.wafers) {
    if (!lots.has(w.lot_id)) lots.set(w.lot_id, { lot: w.lot_id, tool: w.tool_id, start: w.timestamp, wafers: [] });
    const l = lots.get(w.lot_id);
    l.wafers.push(w);
    if (w.timestamp < l.start) l.start = w.timestamp;
  }
  const rows = [...lots.values()].sort((a, b) => (a.start < b.start ? 1 : -1));
  const held = rows.filter((l) => lotStatus(l) === "held").length;
  $("#lots-note").textContent = `${rows.length} lots${held ? ` · ${held} held` : ""} · click a lot to see its wafers`;
  $("#lots tbody").innerHTML = rows.map((l) => {
    const d = l.wafers.filter((w) => w.pattern !== "none");
    const rate = d.length / l.wafers.length;
    const top = Object.entries(d.reduce((m, w) => ((m[w.pattern] = (m[w.pattern] || 0) + 1), m), {})).sort((a, b) => b[1] - a[1])[0];
    const open = S.openLot === l.lot;
    const detail = open ? `<tr class="lot-detail"><td colspan="7"><div class="lot-wafers">${l.wafers.slice().sort((a, b) => a.wafer - b.wafer).map((w) =>
      `<button data-uid="${w.uid}" class="${w.pattern !== "none" ? "defect" : ""}" title="Wafer ${w.wafer}: ${esc(w.pattern)}"><img loading="lazy" src="${thumb(w.uid)}" alt=""><small>W${w.wafer}</small></button>`).join("")}</div></td></tr>` : "";
    return `<tr class="click" data-lot="${esc(l.lot)}" aria-expanded="${open}">
      <td class="mono">${esc(l.lot)}</td><td class="mono">${esc(l.tool)}</td><td>${dayTime(l.start)}</td><td class="num">${l.wafers.length}</td>
      <td><div class="rate${rate >= 0.3 ? " high" : ""}"><div class="rate-track"><span style="width:${(rate * 100).toFixed(1)}%"></span></div><b>${pct(rate, 0)}</b></div></td>
      <td>${top ? `<span class="pattern defect">${esc(top[0])}</span> <span class="muted">×${top[1]}</span>` : `<span class="muted">—</span>`}</td>
      <td>${STATUS_PILL[lotStatus(l)]}</td></tr>${detail}`;
  }).join("") || `<tr><td colspan="7" class="empty">No lots yet. Press play to start the line.</td></tr>`;
}

/* Review queue ---------------------------------------------------- */
function renderReview() {
  const pending = reviewPending();
  $("#review").innerHTML = pending.length ? pending.map((w) => `
    <div class="review-item" data-uid="${w.uid}">
      <img src="${thumb(w.uid, 120)}" alt="Wafer map" data-open="${w.uid}">
      <div>
        <h3>${w.pattern === "none" ? "No pattern" : esc(w.pattern)} <span class="muted">· ${pct(w.confidence, 0)}</span></h3>
        <div class="muted mono">${esc(w.lot_id)} · W${w.wafer} · ${esc(w.tool_id)} · ${hhmm(w.timestamp)}</div>
        <div class="reason">${esc((w.review_reasons || []).map(human).join("; ") || "flagged for review")}</div>
        <div class="actions">
          <button class="btn" data-review="confirm">${ICON.ok}Confirm</button>
          <button class="btn" data-review="reject">Not this pattern</button>
        </div>
      </div>
    </div>`).join("") : `<div class="empty" style="grid-column:1/-1">Nothing waiting. Wafers appear here when the model is not sure enough to accept them on its own.</div>`;
  const done = [...S.reviewed.entries()].reverse();
  $("#reviewed-card").hidden = !done.length;
  $("#reviewed").innerHTML = done.map(([uid, r]) => `<li><span class="ldot ${r.verdict === "confirm" ? "ok" : "hold"}"></span><div><b>${r.verdict === "confirm" ? "Confirmed" : "Rejected"}</b> · ${esc(r.pattern)} · ${esc(r.lot)} W${r.wafer}<div class="when">${esc(r.by)} · line time ${hhmm(r.at)}</div></div></li>`).join("");
}

/* Decisions ------------------------------------------------------- */
function renderDecisions() {
  const list = S.state.decisions || [];
  $("#decisions tbody").innerHTML = list.length ? list.map((d) => `<tr>
    <td class="mono">${dayTime(d.as_of)}</td><td><span class="ldot ${esc(d.action)}" style="display:inline-block;margin-right:8px"></span>${esc(ACTION[d.action])}</td>
    <td class="mono">${esc(d.tool_id)}</td><td>${esc(d.pattern)}</td><td>${esc(d.by)}</td><td class="mono muted">${esc((d.decided_at || "").replace("T", " ").slice(0, 16))}</td></tr>`).join("")
    : `<tr><td colspan="6" class="empty">No decisions yet. They appear here when someone holds lots, opens a ticket or dismisses an alert.</td></tr>`;
}

/* ---------------------------------------------------------------- routing */
function go(page) {
  if (!PAGES[page]) page = "overview";
  S.page = page;
  $$(".page").forEach((p) => { p.hidden = p.dataset.page !== page; });
  $$(".nav-item").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  $("#page-title").textContent = PAGES[page][0];
  $("#page-sub").textContent = PAGES[page][1];
  document.title = `${PAGES[page][0]} · Sentry`;
  if (S.state) renderAll();
  if (!S.focusTool) window.scrollTo(0, 0);
}
const route = () => go(location.hash.slice(1));

/* ---------------------------------------------------------------- wafer inspector */
function opinionBlock(who, detail, note) {
  return `<div class="opinion"><div class="who">${esc(who)}</div><div class="what">${detail}</div>${note ? `<div class="muted" style="margin-top:4px">${esc(note)}</div>` : ""}</div>`;
}

async function openWafer(uid) {
  $("#scrim").hidden = false;
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#dr-title").textContent = `Wafer ${uid}`;
  $("#dr-body").innerHTML = `<div class="empty">Classifying…</div>`;
  const w = await api(`wafer/${uid}`);
  const ctx = w.context || {};
  const line = w.line;
  const c = w.cnn;
  const streamClf = w.stream_classifier || lineMeta().classifier;
  $("#dr-title").textContent = ctx.lot_id ? `${ctx.lot_id} · wafer ${ctx.wafer} · ${ctx.tool_id}` : `Wafer ${uid}`;

  if (!line) {
    $("#dr-body").innerHTML = `<div class="empty">No stream prediction for this wafer.</div>`;
    return;
  }

  const defect = line.pattern !== "none";
  const calNote = line.calibrated
    ? `${pct(line.confidence, 1)} confident · calibrated (CNN)`
    : `${pct(line.confidence, 1)} · uncalibrated raw score (not CNN-calibrated)`;
  const review = line.needs_review
    ? `<div class="review">${ICON.review}<div><b>Sent to an engineer.</b> ${esc((line.review_reasons || []).map(human).join("; "))}.</div></div>`
    : (line.review_rules_validated
      ? ""
      : `<div class="review"><div><b>No validated review policy</b> for this line model. “Not reviewed” is not automatic acceptance.</div></div>`);

  const probsSrc = streamClf === "cnn" && c?.probs ? c.probs : (w.yolo?.probs || c?.probs || {});
  const probs = Object.entries(probsSrc).sort((a, b) => b[1] - a[1]);

  const cnnOp = c ? opinionBlock("CNN (live)", `${esc(c.pattern)} · ${pct(c.confidence, 0)}`, "Calibrated; the line model when the stream is CNN-scored") : "";
  const yoloOp = w.yolo
    ? opinionBlock("YOLO-cls (experimental)", `${esc(w.yolo.pattern)} · ${pct(w.yolo.confidence, 0)}`, "Uncalibrated; its training split overlaps the demo wafers, so not a fair comparison")
    : opinionBlock("YOLO-cls (experimental)", "Not connected", w.yolo_reason || "");

  const agreeBits = [];
  if (w.agree_cnn != null) agreeBits.push(w.agree_cnn ? "agrees with the live CNN" : "disagrees with the live CNN");
  if (w.agree_yolo != null) agreeBits.push(w.agree_yolo ? "agrees with live YOLO" : "disagrees with live YOLO");
  const agree = agreeBits.length
    ? `<div class="agree"><span class="muted">Line prediction ${esc(agreeBits.join("; "))}. Disagreement alone is not a validated hold rule.</span></div>` : "";
  const disc = streamClf === "yolo" && S.meta?.yolo?.disclosure ? `<p class="fine">${esc(S.meta.yolo.disclosure)}</p>` : "";

  $("#dr-body").innerHTML = `
    <div class="wafer-view">
      <div>
        <img class="wafer-img" id="dr-img" src="/api/wafer/${uid}/image.png?size=440" alt="wafer map">
        <div class="seg" role="group"><button class="on" data-view="map">Wafer map</button><button data-view="att" ${streamClf !== "cnn" ? "disabled title=\"Attention map is from the CNN inspector only\"" : ""}>Model attention</button></div>
      </div>
      <div>
        <div class="muted">Line prediction (drives the feed and alerts)</div>
        <div class="verdict ${defect ? "defect" : ""}">${defect ? esc(line.pattern) : "No pattern"}</div>
        <div class="muted">${calNote}</div>
        ${review}
        <div class="kv">
          <span>Line time</span><b>${ctx.timestamp ? fmtTime(ctx.timestamp) : "—"}</b>
          <span>Dies</span><b>${w.dies.toLocaleString()} (${w.fails.toLocaleString()} failing)</b>
          <span>Map size</span><b>${w.shape[0]} × ${w.shape[1]}</b>
          <span>Line model</span><b>${esc(line.classifier)} ${esc(line.model_version)}</b>
          <span>Dataset label</span><b class="muted">${esc(ctx.label || "—")} (demo reference)</b>
        </div>
      </div>
    </div>
    <div class="section-title">Live opinions (not the alert source)</div>
    <div class="opinions">${cnnOp}${yoloOp}</div>
    ${agree}
    ${disc}
    <div class="section-title">Pattern scores${streamClf === "cnn" ? " (CNN)" : w.yolo?.probs ? " (YOLO, uncalibrated)" : ""}</div>
    <div class="probs">${probs.map(([k, v], i) => `<span>${k === "none" ? "No pattern" : esc(k)}</span><div class="bar"><span class="${i === 0 ? "top" : ""}" style="width:${(v * 100).toFixed(1)}%"></span></div><span class="v">${pct(v, 1)}</span>`).join("")}</div>
    <p class="fine">Alerts use the scored stream's line model only. Live opinions are for comparison; experimental or uncalibrated scores are not calibrated confirmation.</p>`;
  $$("#dr-body .seg button:not([disabled])").forEach((b) => b.addEventListener("click", () => {
    $$("#dr-body .seg button").forEach((x) => x.classList.toggle("on", x === b));
    $("#dr-img").src = `/api/wafer/${uid}/image.png?size=440${b.dataset.view === "att" ? "&attention=true" : ""}`;
  }));
}
function closeWafer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#scrim").hidden = true;
}

/* ---------------------------------------------------------------- copilot */
function openCopilot(on = true) {
  $("#copilot").hidden = !on;
  $("#copilot-fab").setAttribute("aria-expanded", String(on));
  if (on) setTimeout(() => $("#ask-input").focus(), 50);
}

async function ask(q) {
  if (!q.trim()) return;
  openCopilot(true);
  const askedAt = asOf();  // freeze replay clock at submit — do not read asOf() after await
  $("#chat-empty")?.remove();
  const chat = $("#chat");
  chat.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}<div class="msg-asof">Asked at line time ${esc(fmtTime(askedAt))}</div></div>`);
  const pending = document.createElement("div");
  pending.className = "msg bot pending";
  pending.innerHTML = `<span class="typing" aria-label="Checking the line"><i></i><i></i><i></i></span>`;
  chat.appendChild(pending);
  chat.scrollTop = chat.scrollHeight;
  try {
    const r = await api("agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, as_of: askedAt }),
    });
    const asOfEcho = r.as_of || askedAt;
    pending.classList.remove("pending");
    if (r.answer) {
      pending.innerHTML = `<div class="msg-asof">Answer as of line time ${esc(fmtTime(asOfEcho))}</div>
        ${renderMd(r.answer)}
        ${r.tools?.length ? `<div class="tools-used">Tools used: ${esc([...new Set(r.tools)].join(", "))}</div>` : ""}`;
    } else {
      pending.classList.add("err");
      pending.innerHTML = `<div class="msg-asof">As of line time ${esc(fmtTime(asOfEcho))}</div>${esc(r.error || "No answer.")}`;
    }
  } catch (e) {
    pending.classList.remove("pending");
    pending.classList.add("err");
    pending.textContent = e.message;
  }
  chat.scrollTop = chat.scrollHeight;
}

/* ---------------------------------------------------------------- wiring */
function pill(id, ok, label, detail) {
  $(id).innerHTML = `<span class="dot ${ok ? "dot-good" : ""}"></span><div><b>${esc(label)}</b>${esc(detail)}</div>`;
  $(id).title = `${label}: ${detail}`;
}

function loadUser() {
  try { S.user = JSON.parse(sessionStorage.getItem("yc-user") || "null"); } catch (_) { /* storage blocked */ }
  const name = S.user?.name || "Engineer";
  $("#user-name").textContent = name;
  $("#user-initials").textContent = name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  $("#user-chip").title = `Signed in as ${name}${S.user?.role ? ` (${S.user.role})` : ""}. Sign out`;
  $("#user-chip").addEventListener("click", () => { try { sessionStorage.removeItem("yc-user"); } catch (_) { /* ignore */ } });
}

async function decide(a, action) {
  await api("decisions", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...a, action, as_of: asOf(), by: S.user?.name || "" }) });
  refresh();
}

async function init() {
  loadUser();
  S.meta = await api("meta");
  const m = S.meta;
  const line = m.model.line || { classifier: m.model.stream_classifier, model_version: m.model.stream_model_version, label: m.model.stream_scored_by, calibrated: m.model.stream_classifier === "cnn" };
  pill("#pill-model", true, "Line model", `${line.label} · drives alerts`);
  if (line.classifier === "yolo") {
    pill("#pill-yolo", true, "Second opinion", `CNN ${m.model.dir} (optional live)`);
  } else {
    pill("#pill-yolo", m.yolo.available, "Second opinion", m.yolo.available ? "YOLO experimental · uncalibrated" : "YOLO not connected");
  }
  if (m.yolo?.disclosure) $("#pill-yolo").title = m.yolo.disclosure;
  pill("#pill-agent", m.agent.available, "Copilot", m.agent.available ? `AMD Token Factory · uses ${m.agent.classifier || line.classifier}` : "not configured");
  $("#cp-sub").textContent = m.agent.available ? "AMD Token Factory · sees the line up to line time" : "Offline on this server";
  const calBit = line.calibrated ? "calibrated CNN scores" : "uncalibrated YOLO scores (experimental)";
  $("#stream-note").textContent = `Replaying ${m.n_wafers} real WM-811K wafers through ${m.tools.length} simulated etch tools · line model ${line.label} (${calBit}).`;
  $("#stream-note").title = `${m.stream}${m.yolo?.disclosure ? " · " + m.yolo.disclosure : ""}`;

  const stepsClf = line.calibrated
    ? "names each wafer's defect pattern with calibrated confidence and flags uncertain wafers for review."
    : "names each wafer's defect pattern with uncalibrated confidence (experimental YOLO-cls; no validated review gates).";
  $("#steps-classifier").innerHTML = `<b>Line model (${esc(line.classifier)})</b> ${stepsClf}`;
  if (!m.agent.available) {
    $("#chat-empty").insertAdjacentHTML("afterbegin", `<div class="notice">${ICON.alert}<span><b>Copilot is offline on this server.</b> Set AMD_API_KEY, AMD_BASE_URL and AMD_MODEL, then restart. Everything else works without it.</span></div>`);
  }
  if (line.classifier === "yolo" && m.yolo?.disclosure) {
    $("#chat-empty").insertAdjacentHTML("beforeend", `<div class="notice">${ICON.alert}<span>${esc(m.yolo.disclosure)}</span></div>`);
  }

  setupFilters();
  const start = Number(new URLSearchParams(location.search).get("t"));
  S.t = Number.isFinite(start) && start > 0 ? Math.min(start, m.timestamps.length - 1) : 0;
  route();
  await refresh();

  // replay
  $("#btn-play").addEventListener("click", () => setPlaying(!S.playing));
  $("#btn-restart").addEventListener("click", () => { setPlaying(false); S.t = 0; S.seen.clear(); refresh(); });
  $("#btn-lot").addEventListener("click", () => { setPlaying(false); nextLot(); });
  $("#speed").addEventListener("change", () => S.playing && setPlaying(true));
  addEventListener("hashchange", route);

  // any wafer thumbnail or feed row opens the inspector
  document.addEventListener("click", (e) => {
    const tr = e.target.closest("#feed tr[data-uid]");
    const btn = e.target.closest(".evidence button[data-uid], .lot-wafers button[data-uid], img[data-open]");
    if (tr || btn) return openWafer(+(tr?.dataset.uid || btn.dataset.uid || btn.dataset.open));
    const tool = e.target.closest(".tool[data-tool]");
    if (tool) { S.focusTool = tool.dataset.tool; location.hash = "tools"; if (S.page === "tools") renderTools(); return; }
    const lot = e.target.closest("#lots tr.click");
    if (lot) { S.openLot = S.openLot === lot.dataset.lot ? null : lot.dataset.lot; renderLots(); return; }
    const rv = e.target.closest("[data-review]");
    if (rv) {
      const uid = +rv.closest(".review-item").dataset.uid, w = S.wafers.find((x) => x.uid === uid);
      S.reviewed.set(uid, { verdict: rv.dataset.review, pattern: w.pattern, lot: w.lot_id, wafer: w.wafer, at: asOf(), by: S.user?.name || "Engineer" });
      renderNav(); renderReview();
    }
  });
  $("#scrim").addEventListener("click", closeWafer);
  $("#dr-close").addEventListener("click", closeWafer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeWafer(); openCopilot(false); } });

  // alerts
  $("#alerts").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b) return;
    const card = b.closest(".alert-card");
    const a = { tool_id: card.dataset.tool, fired_at: card.dataset.fired, pattern: card.dataset.pattern };
    if (b.dataset.act === "ask") {
      return ask(`An alert fired on ${a.tool_id}: repeated ${a.pattern} wafers. Explain the likely cause, which lots are at risk, the cost exposure, and what you recommend. Stay within tool evidence.`);
    }
    await decide(a, b.dataset.act);
  });

  // copilot
  $("#copilot-fab").addEventListener("click", () => openCopilot(true));
  $("#cp-close").addEventListener("click", () => openCopilot(false));
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#ask-input").value; $("#ask-input").value = ""; ask(v); });
  $$(".chip-btn").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
}

init().catch((e) => {
  $("#content").innerHTML = `<div class="card"><div class="empty">Could not load the line: ${esc(e.message)}</div></div>`;
});

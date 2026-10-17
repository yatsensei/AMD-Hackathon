/* Yield Copilot dashboard: replays the scored factory stream and renders it.
   Plain JS, no build step. All data comes from /api (dashboard/server.py). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => {
  if (x == null) return "—";
  const v = (x * 100).toFixed(d);
  return x < 1 && v === (100).toFixed(d) ? `>${(100 - 10 ** -d).toFixed(d)}%` : `${v}%`;  // never round up to a false 100%
};
const usd = (x) => (x == null ? "—" : x >= 1e6 ? `$${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `$${Math.round(x / 1e3)}k` : `$${Math.round(x)}`);
const fmtTime = (iso) => new Date(iso).toLocaleString("en-AU", { weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
const hhmm = (iso) => new Date(iso).toLocaleTimeString("en-AU", { hour: "2-digit", minute: "2-digit", hour12: false });
const human = (s) => ({ low_confidence: "low confidence", confusable_pair: "could be a similar pattern", possible_defect: "possible defect (safety rule)" }[s] || s);

const ICON = {
  ok: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v5M12 16.5v.5"/><path d="M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5M11 8v3M11 14v.5"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z" class="fill"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" class="fill"/></svg>',
};

const S = { meta: null, t: 0, playing: false, timer: null, state: null, seen: new Set(), inflight: false };

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
  const cur = S.state?.feed?.[0]?.lot_id;
  const ts = S.meta.timestamps;
  let i = S.t;
  // jump to the last wafer of the next lot: wafers are 3 min apart, lots 2 h apart
  while (i < ts.length - 1 && new Date(ts[i + 1]) - new Date(ts[i]) < 30 * 60e3) i++;
  if (i === S.t && i < ts.length - 1) { i++; while (i < ts.length - 1 && new Date(ts[i + 1]) - new Date(ts[i]) < 30 * 60e3) i++; }
  S.t = i;
  refresh();
  return cur;
}

async function refresh() {
  if (S.inflight) return;
  S.inflight = true;
  try {
    S.state = await api(`state?as_of=${encodeURIComponent(asOf())}`);
    $("#clock").textContent = fmtTime(asOf());
    renderOps();
  } catch (e) {
    console.error(e);
  } finally {
    S.inflight = false;
  }
}

/* ---------------------------------------------------------------- operations view */
function renderOps() {
  const st = S.state;
  const k = st.kpis;
  $("#kpis").innerHTML = !k ? "" : [
    kpi("Wafers inspected", k.wafers.toLocaleString(), `${k.lots} lots`),
    kpi("Defective wafers", pct(k.defect_rate), `${k.defects} wafers`),
    kpi("Sent to engineer", k.needs_review.toLocaleString(), "model unsure → human review"),
    kpi("Open alerts", k.alerts_open, `${k.alerts_total} total`, k.alerts_open > 0),
    kpi("Exposure since alerts", usd(k.exposure_usd), `at ${usd(S.meta.wafer_value_usd)}/wafer (assumption)`, k.exposure_usd > 0),
  ].join("");

  $("#tools").innerHTML = st.tools.map(toolCard).join("");

  const fresh = [];
  $("#feed").innerHTML = st.feed.map((w) => {
    const isNew = !S.seen.has(w.uid);
    if (isNew) fresh.push(w.uid);
    return feedRow(w, isNew && S.seen.size > 0);
  }).join("");
  fresh.forEach((u) => S.seen.add(u));

  $("#alert-count").textContent = st.alerts.length ? `${st.alerts.length} on the line` : "";
  $("#alerts").innerHTML = st.alerts.length ? st.alerts.slice().reverse().map(alertCard).join("") :
    `<div class="empty">No alerts. Every tool is within its normal defect pattern mix.</div>`;
  $("#decisions").innerHTML = st.decisions.length ? st.decisions.map((d) =>
    `<li><b>${esc({ hold: "Lots held", ticket: "Maintenance ticket opened", dismiss: "Alert dismissed" }[d.action])}</b> · ${esc(d.tool_id)} · ${esc(d.pattern)}
     <div class="when">${esc(d.by)} at line time ${hhmm(d.as_of)}</div></li>`).join("") :
    `<li class="empty" style="border:0">No decisions yet. The copilot recommends; an engineer decides.</li>`;
}

function kpi(label, value, sub, alert = false) {
  return `<div class="kpi${alert ? " alert" : ""}"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${esc(sub)}</div></div>`;
}

function toolCard(t) {
  const vals = t.per_lot_defects;
  const W = 220, H = 46, n = Math.max(vals.length, 6), bw = W / n;
  const max = Math.max(5, ...vals);
  const color = t.status === "alert" ? "var(--brand)" : "var(--steel)";
  const bars = vals.map((v, i) => {
    const h = v === 0 ? 1.5 : Math.max(3, (v / max) * (H - 4));
    return `<rect x="${(i * bw + 1).toFixed(1)}" y="${(H - h).toFixed(1)}" width="${Math.max(2, bw - 3).toFixed(1)}" height="${h.toFixed(1)}" rx="1.5" fill="${v ? color : "#d9d7d3"}"><title>${esc(t.per_lot_ids[i])}: ${v} defective</title></rect>`;
  }).join("");
  const status = t.status === "alert" ? `<span class="status alert">${ICON.alert}Alert</span>` : `<span class="status ok">${ICON.ok}Normal</span>`;
  return `<div class="tool">
    <div class="tool-head"><span class="tool-name">${esc(t.tool_id)}</span>${status}</div>
    <div class="tool-stats"><div><b>${t.wafers}</b>wafers</div><div><b>${pct(t.defect_rate, 0)}</b>defective</div><div><b>${t.defects_24h}</b>defects 24 h</div></div>
    <svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="defective wafers per lot">${bars}</svg>
    <div class="tool-foot"><span>${t.top_pattern_24h ? `Top 24 h: <b>${esc(t.top_pattern_24h)}</b>` : "No defects in 24 h"}</span><span>${esc(t.current_lot || "")}</span></div>
  </div>`;
}

function feedRow(w, isNew) {
  const defect = w.pattern !== "none";
  const status = w.needs_review ? `<span class="flag" title="${esc(w.review_reasons.map(human).join(", "))}">${ICON.review}Review</span>` : `<span class="ok-text">Auto</span>`;
  return `<tr data-uid="${w.uid}" class="${isNew ? "is-new" : ""}">
    <td class="mono">${hhmm(w.timestamp)}</td><td>${esc(w.lot_id)}</td><td>${esc(w.tool_id)}</td>
    <td><img class="thumb" loading="lazy" src="/api/wafer/${w.uid}/image.png?size=64" alt=""></td>
    <td><span class="pattern ${defect ? "defect" : "none"}">${esc(w.pattern)}</span></td>
    <td><div class="conf"><div class="conf-bar"><span style="width:${(w.confidence * 100).toFixed(1)}%"></span></div><span class="conf-val">${pct(w.confidence, 0)}</span></div></td>
    <td>${status}</td></tr>`;
}

function alertCard(a) {
  const rc = a.root_cause || {};
  const d = a.decision;
  const actions = d ? `<div class="decided-note">${ICON.ok}<span><b>${esc({ hold: "Lots held", ticket: "Ticket opened", dismiss: "Dismissed" }[d.action])}</b> by ${esc(d.by)} at ${hhmm(d.as_of)}</span></div>` :
    `<div class="actions">
      <button class="btn primary" data-act="hold">Hold lots</button>
      <button class="btn" data-act="ticket">Open ticket</button>
      <button class="btn" data-act="dismiss">Dismiss</button>
      <button class="btn ghost" data-act="ask">Ask copilot →</button>
    </div>`;
  const also = (a.also_seen || []).length ? `<div class="alert-meta">Also seen on this tool: ${a.also_seen.map((x) => esc(x.pattern)).join(", ")}</div>` : "";
  return `<div class="alert-card${d ? " decided" : ""}" data-tool="${esc(a.tool_id)}" data-fired="${esc(a.fired_at)}" data-pattern="${esc(a.pattern)}">
    <div class="alert-top"><span class="alert-title">${esc(a.pattern)} on ${esc(a.tool_id)}</span><span class="badge">${ICON.alert.replace("<svg", '<svg width="11" height="11" style="vertical-align:-1px"')} Excursion</span></div>
    <div class="alert-meta">${a.count_in_window} wafers in ${a.window_hours} h · fired ${hhmm(a.fired_at)} in ${esc(a.fired_lot_id)}</div>
    ${also}
    <div class="alert-grid">
      <div><span>Lots run since alert</span><b>${a.lots_since_alert}</b></div>
      <div><span>Exposure since alert</span><b>${usd(a.exposure_usd)}</b></div>
    </div>
    <div class="cause-label">Likely cause (triage, not diagnosis)</div>
    <div class="cause">${esc(rc.cause || "—")}</div>
    <div class="cause-label" style="margin-top:6px">Recommended check</div>
    <div class="cause">${esc(rc.recommended_action || "—")}</div>
    ${actions}
  </div>`;
}

/* ---------------------------------------------------------------- wafer inspector */
async function openWafer(uid) {
  $("#scrim").hidden = false;
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#dr-title").textContent = `Wafer ${uid}`;
  $("#dr-body").innerHTML = `<div class="empty">Classifying…</div>`;
  const w = await api(`wafer/${uid}`);
  const c = w.cnn, ctx = w.context || {};
  const defect = c.pattern !== "none";
  $("#dr-title").textContent = ctx.lot_id ? `${ctx.lot_id} · wafer ${ctx.wafer} · ${ctx.tool_id}` : `Wafer ${uid}`;
  const probs = Object.entries(c.probs).sort((a, b) => b[1] - a[1]);
  const review = c.needs_review ? `<div class="review">${ICON.review}<div><b>Sent to an engineer.</b> ${esc(c.review_reasons.map(human).join("; "))}.</div></div>` : "";
  const yolo = w.yolo ? `<div class="opinion"><div class="who">YOLO (second opinion)</div><div class="what">${esc(w.yolo.pattern)} · ${pct(w.yolo.confidence, 0)}</div></div>` :
    `<div class="opinion off"><div class="who">YOLO (second opinion)</div><div class="what">Not connected</div></div>`;
  const agree = w.agree == null ? "" : w.agree ? `<div class="agree"><span class="status ok">${ICON.ok}Models agree</span></div>` :
    `<div class="agree"><span class="status alert">${ICON.alert}Models disagree</span><span class="muted">sent to an engineer</span></div>`;
  $("#dr-body").innerHTML = `
    <div class="wafer-view">
      <div>
        <img class="wafer-img" id="dr-img" src="/api/wafer/${uid}/image.png?size=440" alt="wafer map">
        <div class="seg" role="group"><button class="on" data-view="map">Wafer map</button><button data-view="att">Model attention</button></div>
      </div>
      <div>
        <div class="muted">Classifier result</div>
        <div class="verdict ${defect ? "defect" : ""}">${esc(c.pattern)}</div>
        <div class="muted">${pct(c.confidence, 1)} confident · calibrated</div>
        ${review}
        <div class="kv">
          <span>Line time</span><b>${ctx.timestamp ? fmtTime(ctx.timestamp) : "—"}</b>
          <span>Dies</span><b>${w.dies.toLocaleString()} (${w.fails.toLocaleString()} failing)</b>
          <span>Map size</span><b>${w.shape[0]} × ${w.shape[1]}</b>
          <span>Model</span><b>${esc(c.model_version)}</b>
          <span>Dataset label</span><b class="muted">${esc(ctx.label || "—")} (demo reference)</b>
        </div>
      </div>
    </div>
    <div class="section-title">Second opinion</div>
    <div class="opinions"><div class="opinion"><div class="who">CNN (primary)</div><div class="what">${esc(c.pattern)} · ${pct(c.confidence, 0)}</div></div>${yolo}</div>
    ${agree}
    <div class="section-title">All patterns</div>
    <div class="probs">${probs.map(([k, v], i) => `<span>${esc(k)}</span><div class="bar"><span class="${i === 0 ? "top" : ""}" style="width:${(v * 100).toFixed(1)}%"></span></div><span class="v">${pct(v, 1)}</span>`).join("")}</div>
    <p class="fine">Model attention shows which failing dies drove the decision; the white box frames that cluster. It explains the classification; it is not a separately trained detector.</p>`;
  $("#dr-body").querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    $("#dr-body").querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
    $("#dr-img").src = `/api/wafer/${uid}/image.png?size=440${b.dataset.view === "att" ? "&attention=true" : ""}`;
  }));
}
function closeWafer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#scrim").hidden = true;
}

/* ---------------------------------------------------------------- copilot */
async function ask(q) {
  if (!q.trim()) return;
  $("#chat-empty")?.remove();
  const chat = $("#chat");
  chat.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}</div>`);
  const pending = document.createElement("div");
  pending.className = "msg bot";
  pending.textContent = "Checking the line…";
  chat.appendChild(pending);
  chat.scrollTop = chat.scrollHeight;
  try {
    const r = await api("agent", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q, as_of: asOf() }) });
    if (r.answer) {
      pending.innerHTML = `${esc(r.answer)}${r.tools?.length ? `<div class="tools-used">Tools used: ${esc([...new Set(r.tools)].join(", "))}</div>` : ""}`;
    } else {
      pending.classList.add("err");
      pending.textContent = r.error || "No answer.";
    }
  } catch (e) {
    pending.classList.add("err");
    pending.textContent = e.message;
  }
  chat.scrollTop = chat.scrollHeight;
}

/* ---------------------------------------------------------------- model quality */
async function loadModel() {
  const m = await api("model");
  const x = m.metrics;
  $("#model-kpis").innerHTML = [
    kpi("Macro-F1 (9 patterns)", x.macro_f1.toFixed(3), m.ci ? `95% CI ${m.ci.lo.toFixed(3)}–${m.ci.hi.toFixed(3)}` : ""),
    kpi("Defects caught", pct(x.defect_recall), "defect recall"),
    kpi("Defect calls correct", pct(x.defect_precision), "defect precision"),
    kpi("Calibration error", x.ece.toFixed(3), "ECE · lower is better"),
    kpi("Sent to engineer", pct(x.review_rate), "needs_review share"),
  ].join("");
  $("#model-note").textContent = `${m.model} · ${m.split === "val" ? "validation set (test is scored once at the end)" : "test set"}`;
  $("#perclass").innerHTML = `<thead><tr><th>Pattern</th><th>Precision</th><th>Recall</th><th>F1</th><th>Wafers</th></tr></thead><tbody>` +
    m.per_class.map((r) => `<tr><td>${esc(r.class)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td><b>${r.f1.toFixed(3)}</b></td><td>${r.support.toLocaleString()}</td></tr>`).join("") + "</tbody>";
  const titles = { "experiments.png": "Every experiment", "per_class.png": "Per-class scores", "confusion.png": "Confusion matrix",
    "calibration.png": "Calibration", "coverage.png": "Confidence vs accuracy", "review_rate.png": "Human review routing",
    "robustness.png": "Robustness", "training_curves.png": "Training curves", "localisation.png": "Explanation quality" };
  const order = Object.keys(titles).filter((c) => m.charts.includes(c));
  $("#charts").innerHTML = order.map((c) => `<div class="card chart-card"><div class="card-head"><h2>${esc(titles[c])}</h2></div><img loading="lazy" src="/api/report/${c}" alt="${esc(titles[c])}"></div>`).join("");
}

/* ---------------------------------------------------------------- wiring */
function pill(id, ok, label, detail) {
  $(id).innerHTML = `<span class="dot ${ok ? "dot-good" : "dot-off"}"></span>${esc(label)}: <b>${esc(detail)}</b>`;
  if (!ok) $(id).title = detail;
}

async function init() {
  S.meta = await api("meta");
  const m = S.meta;
  pill("#pill-model", true, "Classifier", `CNN ${m.model.dir || ""}`.trim());
  pill("#pill-yolo", m.yolo.available, "Second opinion", m.yolo.available ? "YOLO connected" : "YOLO not connected");
  pill("#pill-agent", m.agent.available, "Copilot", m.agent.available ? "AMD Token Factory" : "not configured");
  $("#stream-note").textContent = `Replaying ${m.n_wafers} real WM-811K wafers through ${m.tools.length} simulated tools · scored by ${m.model.stream_scored_by}`;
  $("#stream-note").title = m.stream;
  if (!m.agent.available) {
    $("#chat-empty").insertAdjacentHTML("afterbegin", `<div class="notice">${ICON.alert}<span><b>Copilot is offline on this server.</b> Set AMD_API_KEY, AMD_BASE_URL and AMD_MODEL, then restart the dashboard. Everything else works without it.</span></div>`);
  }

  const start = Number(new URLSearchParams(location.search).get("t"));
  S.t = Number.isFinite(start) && start > 0 ? Math.min(start, m.timestamps.length - 1) : 0;
  await refresh();

  $("#btn-play").addEventListener("click", () => setPlaying(!S.playing));
  $("#btn-restart").addEventListener("click", () => { setPlaying(false); S.t = 0; S.seen.clear(); refresh(); });
  $("#btn-lot").addEventListener("click", () => { setPlaying(false); nextLot(); });
  $("#speed").addEventListener("change", () => S.playing && setPlaying(true));
  $("#feed").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-uid]"); if (tr) openWafer(+tr.dataset.uid); });
  $("#scrim").addEventListener("click", closeWafer);
  $("#dr-close").addEventListener("click", closeWafer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeWafer(); });

  $("#alerts").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b) return;
    const card = b.closest(".alert-card");
    const a = { tool_id: card.dataset.tool, fired_at: card.dataset.fired, pattern: card.dataset.pattern };
    if (b.dataset.act === "ask") {
      switchTab("agent");
      return ask(`An alert fired on ${a.tool_id}: repeated ${a.pattern} wafers. Explain the likely cause, which lots are at risk, the cost exposure, and what you recommend.`);
    }
    await api("decisions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...a, action: b.dataset.act, as_of: asOf() }) });
    refresh();
  });

  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#ask-input").value; $("#ask-input").value = ""; ask(v); });
  document.querySelectorAll(".chip-btn").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
}

let modelLoaded = false;
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "model" && !modelLoaded) { modelLoaded = true; loadModel(); }
}

init().catch((e) => {
  document.querySelector("main").innerHTML = `<div class="card empty">Could not load the dashboard: ${esc(e.message)}</div>`;
});

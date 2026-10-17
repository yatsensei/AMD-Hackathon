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
  ok: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v5M12 16.5v.5"/><path d="M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5M11 8v3M11 14v.5"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z" class="fill"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" class="fill"/></svg>',
};

const S = { meta: null, t: 0, playing: false, timer: null, state: null, seen: new Set(), inflight: false };

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
  const cur = S.state?.feed?.[0]?.lot_id;
  const ts = S.meta.timestamps;
  let i = S.t;
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
  const line = lineMeta();
  const reviewSub = line.review_rules_validated
    ? "calibrated review gates → human"
    : "YOLO has no validated review policy";
  $("#kpis").innerHTML = !k ? "" : [
    kpi("Wafers inspected", k.wafers.toLocaleString(), `${k.lots} lots`),
    kpi("Defective wafers", pct(k.defect_rate), `${k.defects} wafers`),
    kpi("Sent to engineer", k.needs_review.toLocaleString(), reviewSub),
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

function feedRow(w, isNew) {
  const defect = w.pattern !== "none";
  return `<tr data-uid="${w.uid}" class="${isNew ? "is-new" : ""}">
    <td class="mono">${hhmm(w.timestamp)}</td><td>${esc(w.lot_id)}</td><td>${esc(w.tool_id)}</td>
    <td><img class="thumb" loading="lazy" src="/api/wafer/${w.uid}/image.png?size=64" alt=""></td>
    <td><span class="pattern ${defect ? "defect" : "none"}">${esc(w.pattern)}</span></td>
    <td><div class="conf"><div class="conf-bar"><span style="width:${(w.confidence * 100).toFixed(1)}%"></span></div><span class="conf-val">${pct(w.confidence, 0)}</span></div></td>
    <td>${feedStatus(w)}</td></tr>`;
}

function alertCard(a) {
  const rc = a.root_cause || {};
  const d = a.decision;
  const line = lineMeta();
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
    <div class="alert-meta">${a.count_in_window} wafers in ${a.window_hours} h · fired ${hhmm(a.fired_at)} in ${esc(a.fired_lot_id)} · line model ${esc(line.label)}</div>
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
      : `<div class="review"><div><b>No validated review policy</b> for this line model — “not reviewed” is not automatic acceptance.</div></div>`);

  const probsSrc = streamClf === "cnn" && c?.probs ? c.probs
    : (w.yolo?.probs || c?.probs || {});
  const probs = Object.entries(probsSrc).sort((a, b) => b[1] - a[1]);

  const cnnOp = c ? opinionBlock("CNN (optional live)", `${esc(c.pattern)} · ${pct(c.confidence, 0)}`, "Calibrated; not the line model unless stream is CNN-scored") : "";
  const yoloOp = w.yolo
    ? opinionBlock("YOLO-cls (experimental)", `${esc(w.yolo.pattern)} · ${pct(w.yolo.confidence, 0)}`, "Uncalibrated; training split overlaps val demo — not a fair CNN compare")
    : opinionBlock("YOLO-cls (experimental)", "Not connected", w.yolo_reason || "");

  const agreeBits = [];
  if (w.agree_cnn != null) agreeBits.push(w.agree_cnn ? "agrees with live CNN" : "disagrees with live CNN");
  if (w.agree_yolo != null) agreeBits.push(w.agree_yolo ? "agrees with live YOLO" : "disagrees with live YOLO");
  const agree = agreeBits.length
    ? `<div class="agree"><span class="muted">Line prediction ${esc(agreeBits.join("; "))}. Disagreement alone is not a validated hold rule.</span></div>`
    : "";

  const disc = streamClf === "yolo" && S.meta?.yolo?.disclosure
    ? `<p class="fine">${esc(S.meta.yolo.disclosure)}</p>` : "";

  $("#dr-body").innerHTML = `
    <div class="wafer-view">
      <div>
        <img class="wafer-img" id="dr-img" src="/api/wafer/${uid}/image.png?size=440" alt="wafer map">
        <div class="seg" role="group"><button class="on" data-view="map">Wafer map</button><button data-view="att" ${streamClf !== "cnn" ? "disabled title=\"Attention map is from the CNN inspector only\"" : ""}>Model attention</button></div>
      </div>
      <div>
        <div class="muted">Line prediction (drives feed &amp; alerts)</div>
        <div class="verdict ${defect ? "defect" : ""}">${esc(line.pattern)}</div>
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
    <div class="section-title">Optional live opinions (not the alert source)</div>
    <div class="opinions">${cnnOp}${yoloOp}</div>
    ${agree}
    ${disc}
    <div class="section-title">Pattern scores${streamClf === "cnn" ? " (CNN)" : w.yolo?.probs ? " (YOLO, uncalibrated)" : ""}</div>
    <div class="probs">${probs.map(([k, v], i) => `<span>${esc(k)}</span><div class="bar"><span class="${i === 0 ? "top" : ""}" style="width:${(v * 100).toFixed(1)}%"></span></div><span class="v">${pct(v, 1)}</span>`).join("")}</div>
    <p class="fine">Alerts use the scored stream’s line model only. Optional opinions are for comparison and must not be read as calibrated confirmation when labeled experimental/uncalibrated.</p>`;
  $("#dr-body").querySelectorAll(".seg button:not([disabled])").forEach((b) => b.addEventListener("click", () => {
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
  const askedAt = asOf();  // freeze replay clock at submit — do not read asOf() after await
  $("#chat-empty")?.remove();
  const chat = $("#chat");
  chat.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}<div class="msg-asof">Asked at line time ${esc(fmtTime(askedAt))}</div></div>`);
  const pending = document.createElement("div");
  pending.className = "msg bot";
  pending.textContent = "Checking the line…";
  chat.appendChild(pending);
  chat.scrollTop = chat.scrollHeight;
  try {
    const r = await api("agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, as_of: askedAt }),
    });
    const asOfEcho = r.as_of || askedAt;
    if (r.answer) {
      pending.innerHTML = `<div class="msg-asof">Answer as of line time ${esc(fmtTime(asOfEcho))}</div>
        ${renderMd(r.answer)}
        ${r.tools?.length ? `<div class="tools-used">Tools used: ${esc([...new Set(r.tools)].join(", "))}</div>` : ""}`;
    } else {
      pending.classList.add("err");
      pending.innerHTML = `<div class="msg-asof">As of line time ${esc(fmtTime(asOfEcho))}</div>${esc(r.error || "No answer.")}`;
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
  $("#model-note").textContent = `${m.model} · ${m.split === "val" ? "validation set (test is scored once at the end)" : "test set"} · CNN metrics (not YOLO)`;
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
  const line = m.model.line || { classifier: m.model.stream_classifier, model_version: m.model.stream_model_version, label: m.model.stream_scored_by, calibrated: m.model.stream_classifier === "cnn" };
  pill("#pill-model", true, "Line model", `${line.label} · drives alerts`);
  if (line.classifier === "yolo") {
    pill("#pill-yolo", true, "Second opinion", `CNN ${m.model.dir} (optional live)`);
    $("#pill-yolo").title = m.yolo?.disclosure || "Experimental YOLO is the line model; CNN is optional";
  } else {
    pill("#pill-yolo", m.yolo.available, "Second opinion",
      m.yolo.available ? "YOLO experimental · uncalibrated" : "YOLO not connected");
    if (m.yolo?.disclosure) $("#pill-yolo").title = m.yolo.disclosure;
  }
  pill("#pill-agent", m.agent.available, "Copilot", m.agent.available ? `AMD · uses ${m.agent.classifier || line.classifier}` : "not configured");
  const calBit = line.calibrated ? "calibrated CNN scores" : "uncalibrated YOLO scores (experimental)";
  $("#stream-note").textContent = `Replaying ${m.n_wafers} real WM-811K wafers · line model ${line.label} (${calBit}) · ${m.tools.length} tools`;
  $("#stream-note").title = `${m.stream}${m.yolo?.disclosure ? " · " + m.yolo.disclosure : ""}`;

  const stepsClf = line.calibrated
    ? "names each wafer’s defect pattern with calibrated confidence and can flag uncertain wafers for review."
    : "names each wafer’s defect pattern with uncalibrated confidence (experimental YOLO-cls; no validated review gates).";
  const stepsEl = $("#steps-classifier");
  if (stepsEl) stepsEl.innerHTML = `<b>Line model (${esc(line.classifier)})</b> ${stepsClf}`;

  if (!m.agent.available) {
    $("#chat-empty")?.insertAdjacentHTML("afterbegin", `<div class="notice">${ICON.alert}<span><b>Copilot is offline on this server.</b> Set AMD_API_KEY, AMD_BASE_URL and AMD_MODEL, then restart the dashboard. Everything else works without it.</span></div>`);
  }
  if (line.classifier === "yolo" && m.yolo?.disclosure) {
    $("#chat-empty")?.insertAdjacentHTML("beforeend", `<div class="notice">${ICON.alert}<span>${esc(m.yolo.disclosure)}</span></div>`);
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
      return ask(`An alert fired on ${a.tool_id}: repeated ${a.pattern} wafers. Explain the likely cause, which lots are at risk, the cost exposure, and what you recommend. Stay within tool evidence.`);
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

/* Demo sign-in (/signin) only remembers a display name in this browser tab. */
function showUser() {
  let u = null;
  try { u = JSON.parse(sessionStorage.getItem("yc-user") || "null"); } catch (_) { /* storage blocked */ }
  if (!u?.name) return;
  const chip = document.querySelector("#user-chip");
  document.querySelector("#user-name").textContent = u.name;
  document.querySelector("#user-initials").textContent = u.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  chip.title = `Signed in as ${u.name}${u.role ? ` (${u.role})` : ""}. Sign out`;
  chip.hidden = false;
  chip.addEventListener("click", () => { try { sessionStorage.removeItem("yc-user"); } catch (_) { /* ignore */ } });
}
showUser();

init().then(() => { if (location.hash === "#model") switchTab("model"); }).catch((e) => {
  document.querySelector("main").innerHTML = `<div class="card empty">Could not load the dashboard: ${esc(e.message)}</div>`;
});

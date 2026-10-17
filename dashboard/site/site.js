/* Yield Copilot site: nav state, scroll reveals, live per-class chart, demo sign-in. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- nav
  const nav = $("#nav");
  if (nav) {
    const onScroll = () => nav.classList.toggle("scrolled", scrollY > 8);
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    const links = new Map($$(".nav-links a").map((a) => [a.getAttribute("href").slice(1), a]));
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.remove("active"));
        links.get(e.target.id)?.classList.add("active");
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach((_, id) => { const s = document.getElementById(id); if (s) spy.observe(s); });
  }

  // ---------------------------------------------------------------- reveals
  const reveal = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      reveal.unobserve(e.target);
    });
  }, { threshold: 0.15 });
  $$(".reveal, .vignette").forEach((el) => reveal.observe(el));

  // ---------------------------------------------------------------- count-up stats
  const fmt = (v, el) => {
    const dec = +(el.dataset.dec || 0);
    let s = dec ? v.toFixed(dec) : Math.round(v).toString();
    if (el.dataset.sep) s = Number(s).toLocaleString("en-US");
    return s + (el.dataset.suffix || "");
  };
  const counter = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      counter.unobserve(e.target);
      const el = e.target, end = +el.dataset.count, t0 = performance.now(), dur = 1100;
      if (reduced) return;
      const tick = (t) => {
        const k = Math.min(1, (t - t0) / dur), ease = 1 - Math.pow(1 - k, 3);
        el.textContent = fmt(end * ease, el);
        if (k < 1) requestAnimationFrame(tick); else el.textContent = fmt(end, el);
      };
      requestAnimationFrame(tick);
    });
  }, { threshold: 0.6 });
  $$("[data-count]").forEach((el) => counter.observe(el));

  // ---------------------------------------------------------------- per-class chart (live from the model's metrics)
  const bars = $("#bars");
  if (bars) {
    const NAMES = { none: "No pattern" };
    fetch("/api/model").then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then((m) => {
      const rows = [...m.per_class].sort((a, b) => b.f1 - a.f1);
      bars.innerHTML = rows.map((r) => `
        <div class="bar-row" title="${r.class}: precision ${(r.precision * 100).toFixed(1)}%, recall ${(r.recall * 100).toFixed(1)}%, ${r.support.toLocaleString()} wafers">
          <span class="lbl">${NAMES[r.class] || r.class}</span>
          <div class="bar-track"><div class="bar-fill" data-w="${(r.f1 * 100).toFixed(1)}"></div></div>
          <span class="val">${r.f1.toFixed(2)}</span>
        </div>`).join("");
      $("#perclass-note").textContent = `Validation lots · ${m.model} · macro-F1 ${m.metrics.macro_f1.toFixed(3)}`;
      const grow = () => $$(".bar-fill", bars).forEach((b) => { b.style.width = b.dataset.w + "%"; });
      const card = $("#perclass");
      if (reduced) return grow();
      new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { grow(); o.disconnect(); } }, { threshold: 0.3 }).observe(card);
    }).catch(() => { $("#perclass").hidden = true; });
  }

  // ---------------------------------------------------------------- demo sign-in (no accounts: remembers a display name only)
  const form = $("#signin-form");
  if (form) {
    const next = new URLSearchParams(location.search).get("next");
    const go = (name, role) => {
      try { sessionStorage.setItem("yc-user", JSON.stringify({ name, role })); } catch (_) { /* private mode: fine */ }
      location.href = next === "model" ? "/app/#model" : "/app/";
    };
    const nameFromEmail = (email) => {
      const local = (email || "").split("@")[0].replace(/[._-]+/g, " ").trim();
      return local ? local.replace(/\b\w/g, (c) => c.toUpperCase()) : "Guest engineer";
    };
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      go(nameFromEmail($("#email").value), "Yield engineer");
    });
    $$(".role").forEach((b) => b.addEventListener("click", () => go(b.dataset.name, b.dataset.role)));
  }
})();

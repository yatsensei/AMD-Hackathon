/* Sentry site: background field, scroll motion, live per-class chart, demo sign-in. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.documentElement.classList.add("motion");

  // ---------------------------------------------------------------- background field
  // A grid of dies, like a wafer under inspection: slow rings travel outward from the centre
  // and brighten the dies they pass; now and then a die lights red as a fail and fades.
  function field(canvas) {
    const host = canvas.parentElement;
    const ctx = canvas.getContext("2d");
    const CELL = 22, DIE = 3;
    const RED = [226, 35, 45];
    let w = 0, h = 0, dpr = 1, cols = 0, rows = 0, raf = 0, visible = false, t0 = performance.now();
    let fails = [], nextFail = 0, mouse = null, keep = null;
    const copy = host.querySelector(".hero-copy, .cta-inner, .si-brand");

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = host.clientWidth; h = host.clientHeight;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.ceil(w / CELL) + 1; rows = Math.ceil(h / CELL) + 1;
      fails = [];
      if (copy) {
        const hr = host.getBoundingClientRect(), cr = copy.getBoundingClientRect();
        keep = { l: cr.left - hr.left - 24, r: cr.right - hr.left + 24, t: cr.top - hr.top - 24, b: cr.bottom - hr.top + 24 };
      }
      if (!raf) draw(performance.now());
    };

    const draw = (now) => {
      const t = (now - t0) / 1000;
      ctx.clearRect(0, 0, w, h);
      const cx = w * 0.5, cy = Math.min(h * 0.42, 420);
      const rx = Math.max(w * 0.62, 320), ry = Math.max(h * 0.62, 260);
      const ox = (w - (cols - 1) * CELL) / 2, oy = 6;

      // fails: spawn on the travelling ring, live ~2.4 s
      if (!reduced && now > nextFail && fails.length < 14) {
        const ang = Math.random() * Math.PI * 2;
        const ringR = ((t * 70) % 520) + 40;
        const fx = Math.round((cx + Math.cos(ang) * ringR - ox) / CELL), fy = Math.round((cy + Math.sin(ang) * ringR * 0.7 - oy) / CELL);
        const px = ox + fx * CELL, py = oy + fy * CELL;
        const behindCopy = keep && px > keep.l && px < keep.r && py > keep.t && py < keep.b;  // never under the text
        if (fx >= 0 && fy >= 0 && fx < cols && fy < rows && !behindCopy) fails.push({ x: fx, y: fy, born: now });
        nextFail = now + 160 + Math.random() * 260;
      }
      fails = fails.filter((f) => now - f.born < 2400);
      const failAt = new Map(fails.map((f) => [f.y * cols + f.x, f]));

      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
          const x = ox + i * CELL, y = oy + j * CELL;
          const dx = (x - cx) / rx, dy = (y - cy) / ry;
          const edge = 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy));      // fade toward the edges
          if (edge <= 0.02) continue;
          const d = Math.hypot(x - cx, (y - cy) * 1.25);
          const wave = Math.pow(Math.max(0, Math.sin(d / 52 - t * 1.4)), 4);  // thin rings moving outward
          let a = 0.055 + 0.3 * wave;
          if (mouse) a += 0.14 * Math.max(0, 1 - Math.hypot(x - mouse.x, y - mouse.y) / 140);
          a *= edge;
          const f = failAt.get(j * cols + i);
          if (f) {
            const k = (now - f.born) / 2400, life = k < 0.15 ? k / 0.15 : 1 - (k - 0.15) / 0.85;
            ctx.fillStyle = `rgba(${RED[0]},${RED[1]},${RED[2]},${(0.85 * life * Math.max(edge, 0.35)).toFixed(3)})`;
            ctx.fillRect(x - 1, y - 1, DIE + 2, DIE + 2);
          } else {
            ctx.fillStyle = `rgba(244,244,245,${a.toFixed(3)})`;
            ctx.fillRect(x, y, DIE, DIE);
          }
        }
      }
      raf = visible && !reduced ? requestAnimationFrame(draw) : 0;
    };

    const start = () => { if (!raf && visible && !reduced && !document.hidden) raf = requestAnimationFrame(draw); };
    const stop = () => { cancelAnimationFrame(raf); raf = 0; };
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible ? start() : stop(); }).observe(host);
    document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
    new ResizeObserver(resize).observe(host);
    if (matchMedia("(hover: hover)").matches) {
      host.addEventListener("pointermove", (e) => { const r = host.getBoundingClientRect(); mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; });
      host.addEventListener("pointerleave", () => { mouse = null; });
    }
    resize();
    canvas.classList.add("on");
  }
  $$("canvas.field").forEach((c) => { try { field(c); } catch (_) { /* no canvas: the static dot grid stays */ } });

  // ---------------------------------------------------------------- nav: shadow, section spy, progress
  const nav = $("#nav");
  if (nav) {
    const bar = $(".progress", nav);
    let ticking = false;
    const onScroll = () => {
      ticking = false;
      nav.classList.toggle("scrolled", scrollY > 8);
      const max = document.documentElement.scrollHeight - innerHeight;
      if (bar) bar.style.setProperty("--p", max > 0 ? (scrollY / max).toFixed(4) : 0);
    };
    addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
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

  // ---------------------------------------------------------------- reveals (once)
  $$(".stagger").forEach((g) => [...g.children].forEach((c, i) => c.style.setProperty("--i", i)));
  const reveal = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      reveal.unobserve(e.target);
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  $$(".reveal, .stagger, .vignette, .vis-pair").forEach((el) => reveal.observe(el));

  // ---------------------------------------------------------------- product shot: tilts flat as it scrolls in
  const shot = $(".product-shot");
  if (shot && !reduced) {
    let pending = false;
    const tilt = () => {
      pending = false;
      const r = shot.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (innerHeight - r.top) / (innerHeight * 0.75)));
      shot.style.setProperty("--tilt", `${(1 - p) * 14}deg`);
      shot.style.setProperty("--scale", (0.96 + 0.04 * p).toFixed(4));
    };
    addEventListener("scroll", () => { if (!pending) { pending = true; requestAnimationFrame(tilt); } }, { passive: true });
    addEventListener("resize", tilt);
    tilt();
  }

  // ---------------------------------------------------------------- count-up numbers
  const fmt = (v, el) => {
    const dec = +(el.dataset.dec || 0);
    let s = dec ? v.toFixed(dec) : Math.round(v).toString();
    if (el.dataset.sep) s = Number(s).toLocaleString("en-US");
    return (el.dataset.prefix || "") + s + (el.dataset.suffix || "");
  };
  const counter = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      counter.unobserve(e.target);
      const el = e.target, end = +el.dataset.count, t0 = performance.now(), dur = 1300;
      if (reduced) return;
      const tick = (t) => {
        const k = Math.min(1, (t - t0) / dur), ease = 1 - Math.pow(1 - k, 4);
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
      bars.innerHTML = rows.map((r, i) => `
        <div class="bar-row" title="${r.class}: precision ${(r.precision * 100).toFixed(1)}%, recall ${(r.recall * 100).toFixed(1)}%, ${r.support.toLocaleString()} wafers">
          <span class="lbl">${NAMES[r.class] || r.class}</span>
          <div class="bar-track"><div class="bar-fill" data-w="${(r.f1 * 100).toFixed(1)}" style="transition-delay:${i * 60}ms"></div></div>
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
      location.href = "/app/";
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

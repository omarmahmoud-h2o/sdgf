(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const readJSON = (id) => JSON.parse(document.getElementById(id).textContent);

  const GLOSSARY = readJSON("glossary-data");
  const LAYERS = readJSON("layer-data");
  const DIAGRAMS = readJSON("diagram-data");
  const panels = $$(".tab-panel");
  const tabButtons = $$(".tabs [role=tab]");

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
  };

  // ---------------------------------------------------------------- theme

  const THEMES = ["auto", "light", "dark"];
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  let theme = THEMES.includes(store.get("sdgf-guide-theme")) ? store.get("sdgf-guide-theme") : "auto";

  const resolvedTheme = () => (theme === "auto" ? (darkQuery.matches ? "dark" : "light") : theme);

  function applyTheme() {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    $("#theme-btn").textContent = `Theme: ${theme}`;
    $$(".diagram iframe").forEach(themeFrame);
  }

  // Archify pages keep their own theme; flip it with the page's own toggle so its
  // labels and stored preference stay consistent.
  function themeFrame(frame) {
    try {
      const doc = frame.contentDocument;
      if (!doc || !doc.documentElement) return;
      if (doc.documentElement.getAttribute("data-theme") === resolvedTheme()) return;
      const toggle = doc.getElementById("btn-theme");
      if (toggle) toggle.click();
      else doc.documentElement.setAttribute("data-theme", resolvedTheme());
    } catch { /* not loaded yet */ }
  }

  $("#theme-btn").addEventListener("click", () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    store.set("sdgf-guide-theme", theme);
    applyTheme();
  });
  darkQuery.addEventListener("change", () => theme === "auto" && applyTheme());

  // ---------------------------------------------------------------- diagrams

  function loadDiagrams(panel) {
    $$(".diagram", panel).forEach((fig) => {
      if ($("iframe", fig)) return;
      const html = DIAGRAMS[fig.dataset.diagram];
      if (!html) return;
      const frame = document.createElement("iframe");
      frame.title = fig.querySelector("figcaption").firstChild.textContent.trim();
      frame.addEventListener("load", () => {
        themeFrame(frame);
        const loading = $(".diagram-loading", fig);
        if (loading) loading.remove();
      });
      frame.srcdoc = html;
      $(".diagram-frame", fig).appendChild(frame);
    });
  }

  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".diagram-open");
    if (!btn) return;
    const html = DIAGRAMS[btn.closest(".diagram").dataset.diagram];
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    window.open(url, "_blank", "noopener");
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  });

  // ---------------------------------------------------------------- tabs + toc

  let activePanel = null;

  function showPanel(panel) {
    if (activePanel === panel) return;
    activePanel = panel;
    panels.forEach((p) => { p.hidden = p !== panel; });
    tabButtons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === panel.id)));
    store.set("sdgf-guide-tab", panel.id);
    loadDiagrams(panel);
    buildToc(panel);
  }

  function buildToc(panel) {
    const toc = $("#toc");
    toc.innerHTML = "";
    $$("h2[id], h3[id]", panel).forEach((h) => {
      const a = document.createElement("a");
      a.href = `#${h.id}`;
      a.textContent = h.textContent;
      a.className = h.tagName === "H3" ? "lvl-3" : "lvl-2";
      a.dataset.target = h.id;
      toc.appendChild(a);
    });
    spy();
  }

  function spy() {
    if (!activePanel) return;
    const heads = $$("h2[id], h3[id]", activePanel).filter((h) => h.offsetParent !== null);
    let current = heads[0];
    for (const h of heads) if (h.getBoundingClientRect().top < 120) current = h;
    $$("#toc a").forEach((a) => a.classList.toggle("active", current && a.dataset.target === current.id));
  }
  window.addEventListener("scroll", () => requestAnimationFrame(spy), { passive: true });

  tabButtons.forEach((b) => b.addEventListener("click", () => {
    history.pushState(null, "", `#${b.dataset.tab}`);
    route(false);
    window.scrollTo(0, 0);
  }));

  // ---------------------------------------------------------------- layer explorer

  function selectLayer(code, focusCard = false) {
    $$(".layer-card").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.layer === code)));
    $$(".layer-pane").forEach((p) => { p.hidden = p.dataset.layer !== code; });
    if (focusCard) $(`.layer-card[data-layer="${code}"]`).focus();
    spy();
  }

  $$(".layer-card").forEach((card) => card.addEventListener("click", () => {
    selectLayer(card.dataset.layer);
    history.replaceState(null, "", `#${LAYERS[card.dataset.layer]}`);
  }));

  // prev / next inside each pane
  const codes = $$(".layer-card").map((c) => c.dataset.layer);
  $$(".layer-pane").forEach((pane, i) => {
    const nav = document.createElement("div");
    nav.className = "layer-nav";
    const mk = (j, label) => {
      if (j < 0 || j >= codes.length) return document.createElement("span");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tool-btn";
      b.textContent = label;
      b.addEventListener("click", () => {
        selectLayer(codes[j]);
        history.replaceState(null, "", `#${LAYERS[codes[j]]}`);
        $(".layer-explorer").scrollIntoView({ block: "start", behavior: "smooth" });
      });
      return b;
    };
    nav.append(mk(i - 1, `← ${codes[i - 1] || ""}`), mk(i + 1, `${codes[i + 1] || ""} →`));
    pane.appendChild(nav);
  });
  if (codes.length) selectLayer(codes[0]);

  // FAG | CFA toggle
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg button");
    if (!btn) return;
    const cmp = btn.closest(".compare");
    setCompare(cmp, btn.dataset.show);
    store.set("sdgf-guide-compare", btn.dataset.show);
  });

  function setCompare(cmp, show) {
    cmp.dataset.show = show;
    $$(".seg button", cmp).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.show === show)));
  }
  const savedCompare = store.get("sdgf-guide-compare");
  if (["fag", "cfa", "both"].includes(savedCompare)) $$(".compare").forEach((c) => setCompare(c, savedCompare));

  // overview's six-checks table rows open the matching card
  $$("#overview tr").forEach((row) => {
    const code = row.cells[0] && row.cells[0].textContent.trim();
    if (!LAYERS[code]) return;
    row.classList.add("layer-link");
    row.tabIndex = 0;
    row.title = `Open ${code} in How it works`;
    const go = () => { location.hash = LAYERS[code]; };
    row.addEventListener("click", go);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });

  // ---------------------------------------------------------------- routing

  function reveal(el) {
    const panel = el.closest(".tab-panel");
    if (panel) showPanel(panel);
    const pane = el.closest(".layer-pane");
    if (pane) selectLayer(pane.dataset.layer);
    const cmp = el.closest(".compare");
    if (cmp && cmp.dataset.show !== "both" && el.closest("[data-side]")) setCompare(cmp, "both");
  }

  function flash(el) {
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
  }

  function route(scroll = true) {
    const id = decodeURIComponent(location.hash.slice(1));
    if (id === "glossary") { openGlossary(); return; }
    const el = id && document.getElementById(id);
    if (!el) {
      const saved = document.getElementById(store.get("sdgf-guide-tab") || "");
      showPanel(saved && saved.classList.contains("tab-panel") ? saved : panels[0]);
      return;
    }
    reveal(el);
    if (scroll && !el.classList.contains("tab-panel")) {
      requestAnimationFrame(() => { el.scrollIntoView({ block: "start" }); flash(el); });
    }
  }
  window.addEventListener("hashchange", () => route());

  // ---------------------------------------------------------------- glossary

  const dialog = $("#glossary");
  function openGlossary() {
    hideTip();
    if (!dialog.open) dialog.showModal();
  }
  $("#glossary-btn").addEventListener("click", openGlossary);
  $("#glossary-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener("close", () => {
    if (location.hash === "#glossary") history.replaceState(null, "", location.pathname + location.search);
  });

  const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const byTerm = new Map(GLOSSARY.map((g) => [g.term.toLowerCase(), g]));
  const termRe = new RegExp(
    `\\b(${GLOSSARY.map((g) => g.term).sort((a, b) => b.length - a.length).map(escapeRe).join("|")})s?\\b`,
    "i",
  );
  const SKIP = "code, pre, a, button, h1, h2, h3, h4, th, .term, .seg, .lc-code, .lc-name, .pill, .cmp-label";

  // mark the first use of each term in every h2 section
  panels.forEach((panel) => {
    let used = new Set();
    const walker = document.createTreeWalker(panel, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    const todo = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (n.nodeType === Node.ELEMENT_NODE) {
        if (n.tagName === "H2") used = new Set();
        continue;
      }
      const parent = n.parentElement;
      if (!parent || parent.closest(SKIP) || !parent.closest("p, li, td, .cmp-cell, .lc-q")) continue;
      const m = termRe.exec(n.nodeValue);
      if (!m) continue;
      const key = m[1].toLowerCase();
      if (used.has(key)) continue;
      used.add(key);
      todo.push([n, m]);
    }
    todo.forEach(([node, m]) => {
      const after = node.splitText(m.index);
      after.splitText(m[0].length);
      const span = document.createElement("span");
      span.className = "term";
      span.tabIndex = 0;
      span.dataset.term = m[1].toLowerCase();
      span.textContent = after.nodeValue;
      after.replaceWith(span);
    });
  });

  const tip = $("#term-tip");
  let tipFor = null;
  function showTip(el) {
    const g = byTerm.get(el.dataset.term);
    if (!g) return;
    tipFor = el;
    tip.innerHTML =
      `<span class="tt-term">${g.term}</span>` +
      (g.code ? `<span class="tt-code">in code: ${g.code}</span>` : "") +
      `<span>${g.defn}</span><a class="tt-more" href="#glossary">All terms</a>`;
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
    let top = r.bottom + 6;
    if (top + h > window.innerHeight - 8) top = r.top - h - 6;
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  function hideTip() { tip.hidden = true; tipFor = null; }

  document.addEventListener("mouseover", (e) => {
    const t = e.target.closest(".term");
    if (t) showTip(t);
    else if (tipFor && !e.target.closest("#term-tip")) hideTip();
  });
  document.addEventListener("focusin", (e) => { if (e.target.classList.contains("term")) showTip(e.target); });
  document.addEventListener("focusout", (e) => { if (e.target.classList.contains("term")) setTimeout(() => { if (!tip.contains(document.activeElement)) hideTip(); }, 100); });
  document.addEventListener("click", (e) => {
    const t = e.target.closest(".term");
    if (t) { e.preventDefault(); showTip(t); }
    else if (!e.target.closest("#term-tip")) hideTip();
  });
  window.addEventListener("scroll", hideTip, { passive: true });

  // ---------------------------------------------------------------- search

  const input = $("#search");
  const results = $("#search-results");
  const index = [];
  panels.forEach((panel) => {
    const tabTitle = $("h1", panel).textContent;
    let section = tabTitle;
    let sub = "";
    $$("h2, h3, p, li, td, pre, .cmp-cell, .lc-q", panel).forEach((el) => {
      if (el.tagName === "H2") { section = el.textContent; sub = ""; }
      if (el.tagName === "H3") sub = el.textContent;
      if (el.closest(".layer-card") && !el.classList.contains("lc-q")) return;
      if (el.tagName === "LI" && el.querySelector("p")) return;
      if (el.tagName === "P" && el.closest("li, td")) return;
      const text = el.textContent.replace(/\s+/g, " ").trim();
      if (text) index.push({ el, text, lower: text.toLowerCase(), where: [tabTitle, section, sub].filter((x, i, a) => x && a.indexOf(x) === i).join(" › ") });
    });
  });

  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  function snippet(text, words) {
    const lower = text.toLowerCase();
    const at = Math.max(0, lower.indexOf(words[0]) - 50);
    let s = (at > 0 ? "…" : "") + text.slice(at, at + 160) + (at + 160 < text.length ? "…" : "");
    s = escapeHtml(s);
    words.forEach((w) => { s = s.replace(new RegExp(`(${escapeRe(escapeHtml(w))})`, "gi"), "<mark>$1</mark>"); });
    return s;
  }

  function runSearch() {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { results.hidden = true; return; }
    const words = q.split(/\s+/);
    const hits = index.filter((it) => words.every((w) => it.lower.includes(w))).slice(0, 40);
    results.innerHTML = hits.length
      ? hits.map((h, i) => `<button type="button" role="option" data-i="${index.indexOf(h)}"><span class="sr-where">${escapeHtml(h.where)}</span>${snippet(h.text, words)}</button>`).join("")
      : `<div class="sr-empty">No matches for “${escapeHtml(input.value.trim())}”</div>`;
    results.hidden = false;
  }

  function goTo(item) {
    results.hidden = true;
    reveal(item.el);
    requestAnimationFrame(() => {
      item.el.scrollIntoView({ block: "center" });
      flash(item.el);
    });
  }

  input.addEventListener("input", runSearch);
  input.addEventListener("focus", runSearch);
  input.addEventListener("keydown", (e) => {
    const first = $("button", results);
    if (e.key === "Enter" && first) { e.preventDefault(); goTo(index[first.dataset.i]); }
    if (e.key === "ArrowDown" && first) { e.preventDefault(); first.focus(); }
    if (e.key === "Escape") { results.hidden = true; input.blur(); }
  });
  results.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) goTo(index[b.dataset.i]);
  });
  results.addEventListener("keydown", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (e.key === "ArrowDown" && b.nextElementSibling) { e.preventDefault(); b.nextElementSibling.focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); (b.previousElementSibling || input).focus(); }
    if (e.key === "Escape") { results.hidden = true; input.focus(); }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) results.hidden = true; });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement === document.body) { e.preventDefault(); input.focus(); }
    if (e.key === "Escape") hideTip();
  });

  // ---------------------------------------------------------------- start

  if (window.matchMedia("(max-width: 960px)").matches) $(".toc-wrap").open = false;
  $("#toc").addEventListener("click", (e) => {
    if (e.target.closest("a") && window.matchMedia("(max-width: 960px)").matches) $(".toc-wrap").open = false;
  });
  applyTheme();
  route();
})();

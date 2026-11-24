// Header, markdown pages (README.md is the home page), code highlighting and copy buttons.
const site = (() => {
  const PAGES = [["quickstart.html", "Quick start"], ["models.html", "Models"], ["builder.html", "Builder"], ["reference.html", "Reference"]];
  const page = location.pathname.split("/").pop() || "index.html";
  // project pages live at https://<owner>.github.io/<repo>/
  const repo = location.hostname.endsWith(".github.io")
    ? `https://github.com/${location.hostname.split(".")[0]}/${location.pathname.split("/")[1]}` : null;

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  function header() {
    const links = PAGES.map(([href, name]) =>
      `<a href="${href}"${href === page ? ' aria-current="page"' : ""}>${name}</a>`).join("");
    const top = document.createElement("header");
    top.className = "top";
    top.innerHTML = `<a class="brand" href="./"><picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/logo-white.svg">
        <img src="assets/logo-black.svg" alt=""></picture></a>
      <nav>${links}${repo ? `<a href="${repo}">GitHub</a>` : ""}</nav>`;
    document.body.prepend(top);
  }

  const RULES = {
    python: /(?<c>#[^\n]*)|(?<s>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(?<k>\b(?:from|import|as|def|return|for|in|if|else|with|None|True|False)\b)/g,
    sh: /(?<c>(?<=^|\s)#[^\n]*)|(?<s>"(?:[^"\\\n]|\\.)*"|'[^'\n]*')/gm,
  };
  RULES.py = RULES.python;
  RULES.bash = RULES.shell = RULES.sh;

  function highlight(code, lang) {
    const re = RULES[lang];
    if (!re) return;
    const text = code.textContent;
    let html = "", last = 0;
    for (const m of text.matchAll(re)) {
      const kind = Object.keys(m.groups).find((k) => m.groups[k] !== undefined);
      html += esc(text.slice(last, m.index)) + `<span class="t-${kind}">${esc(m[0])}</span>`;
      last = m.index + m[0].length;
    }
    code.innerHTML = html + esc(text.slice(last));
  }

  function copyButton(pre) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "copy";
    b.textContent = "Copy";
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code").textContent);
        b.textContent = "Copied";
      } catch {
        b.textContent = "Select and copy";
      }
      setTimeout(() => (b.textContent = "Copy"), 1500);
    });
    pre.append(b);
  }

  // install lines for uv: same packages, comments kept in their column
  function uv(line) {
    const m = line.match(/^pip install(.*?)( {2,}#.*)?$/);
    if (!m) return line;
    const note = m[2] ? " ".repeat(Math.max(2, m[2].indexOf("#") - 3)) + m[2].trim() : "";
    return `uv pip install${m[1]}${note}`;
  }

  function installer(code) {
    const pip = code.textContent;
    if (!/^pip install/m.test(pip)) return;
    const alt = pip.split("\n").map(uv).join("\n");
    const bar = document.createElement("div");
    bar.className = "pm";
    bar.innerHTML = '<button type="button" data-pm="pip">pip</button><button type="button" data-pm="uv">uv</button>';
    code.parentElement.before(bar);
    bar.addEventListener("pm", (e) => {
      code.textContent = e.detail === "uv" ? alt : pip;
      highlight(code, "sh");
      bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.pm === e.detail)));
    });
    bar.addEventListener("click", (e) => {
      if (!e.target.dataset.pm) return;
      try { localStorage.setItem("pm", e.target.dataset.pm); } catch { /* private mode */ }
      document.querySelectorAll(".pm").forEach((b) => b.dispatchEvent(new CustomEvent("pm", { detail: e.target.dataset.pm })));
    });
    let saved = "pip";
    try { saved = localStorage.getItem("pm") || "pip"; } catch { /* private mode */ }
    bar.dispatchEvent(new CustomEvent("pm", { detail: saved }));
  }

  // the sections of a page, for jumping: a column beside the text on wide screens
  function toc(main, heads) {
    let html = "", open = false;
    for (const [h, text] of heads) {
      if (h.tagName === "H2") {
        html += `${open ? "</ul></li>" : ""}<li><a href="#${h.id}">${esc(text)}</a><ul>`;
        open = true;
      } else {
        html += `<li><a href="#${h.id}">${esc(text)}</a></li>`;
      }
    }
    const nav = document.createElement("nav");
    nav.className = "toc";
    nav.setAttribute("aria-label", "On this page");
    nav.innerHTML = `<ul>${html}${open ? "</ul></li>" : ""}</ul>`;
    const h1 = main.querySelector("h1"), doc = document.createElement("div");
    doc.className = "doc";
    while (h1.nextSibling) doc.append(h1.nextSibling);
    h1.after(nav, doc);
  }

  function slug(text) {
    return text.trim().toLowerCase().replace(/[^\w\s-]/g, "").replace(/\s+/g, "-");
  }

  // the README links to the repository's files from its root and to the site by its full URL
  function relink(root) {
    const fix = (url) => url.replace(/^docs\//, "").replace(/^(models|reference|quickstart)\.md\b/, "$1.html")
      .replace(/^https:\/\/[^/]+\.github\.io\/[^/]+\/(.*)$/, (_, p) => p || "./");
    root.querySelectorAll("[src]").forEach((e) => e.setAttribute("src", fix(e.getAttribute("src"))));
    root.querySelectorAll("[srcset]").forEach((e) => e.setAttribute("srcset", fix(e.getAttribute("srcset"))));
    root.querySelectorAll("a[href]").forEach((e) => e.setAttribute("href", fix(e.getAttribute("href"))));
  }

  // everything above the first section: logo, name, links as buttons, figures
  function hero(root) {
    const box = document.createElement("div");
    box.className = "hero";
    const first = root.querySelector("h2");
    root.prepend(box);
    while (box.nextSibling && box.nextSibling !== first) box.append(box.nextSibling);
    for (const p of box.querySelectorAll(":scope > p")) {
      const links = [...p.querySelectorAll("a")];
      const rest = links.reduce((t, a) => t.replace(a.textContent, ""), p.textContent);
      if (p.querySelectorAll("picture").length > 1) {
        p.className = "figures";
      } else if (links.length > 1 && !rest.replace(/[\s·|]/g, "")) {
        p.className = "links";
        p.replaceChildren(...links);
      }
    }
  }

  async function markdown(main) {
    let text = null;
    for (const url of [main.dataset.md, "../" + main.dataset.md]) {    // the site, or docs/ of a checkout
      try {
        const r = await fetch(url);
        if (r.ok) { text = await r.text(); break; }
      } catch { /* next */ }
    }
    if (text === null) {
      main.textContent = `${main.dataset.md} could not be loaded.`;
      return;
    }
    main.innerHTML = marked.parse(text);
    relink(main);
    if (page === "index.html") hero(main);
    const heads = [...main.querySelectorAll("h2, h3")].map((h) => [h, h.textContent]);
    for (const [h, text] of heads) {
      h.id = slug(text);
      h.insertAdjacentHTML("beforeend", `<a class="anchor" href="#${h.id}" aria-label="link to this section">#</a>`);
    }
    main.querySelectorAll("table").forEach((t) => {
      const first = t.querySelector("thead th")?.textContent.trim();
      if (OPTION_TABLES.includes(first)) optionList(t);
    });
    if ("toc" in main.dataset) {
      toc(main, heads);
      if (main.querySelector(".opt")) filter(main);
    }
    main.querySelectorAll("table").forEach((t) => {
      const box = document.createElement("div");
      box.className = "table";
      t.replaceWith(box);
      box.append(t);
      // phones show a row as a block: every cell after the first under its column's name
      const names = [...t.querySelectorAll("thead th")].map((th) => th.textContent.trim());
      t.querySelectorAll("tbody tr").forEach((tr) => [...tr.children].forEach((td, i) => {
        if (i && names[i] && names[i] !== "description") td.dataset.label = names[i];
      }));
    });
    window.addEventListener("hashchange", reveal);
    reveal();
  }

  // a section's target may sit inside folded options
  function reveal() {
    const el = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!el) return;
    for (let d = el.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
    el.scrollIntoView();
    if (el.classList.contains("opt")) {
      el.classList.remove("hit");
      void el.offsetWidth;
      el.classList.add("hit");
    }
  }

  // option tables (first column "option", "argument", ...) as a list: name, description, default.
  // A row with only a bold first cell starts a group; the group "more" is folded.
  const OPTION_TABLES = ["option", "argument", "function", "key", "variable"];

  function optionList(t) {
    const names = [...t.querySelectorAll("thead th")].map((th) => th.textContent.trim());
    const section = t.closest("main").querySelectorAll("h2, h3");
    let owner = "";
    for (const h of section) if (h.compareDocumentPosition(t) & Node.DOCUMENT_POSITION_FOLLOWING) owner = h.id;
    const list = document.createElement("div");
    list.className = "opts";
    let box = list;
    for (const tr of t.querySelectorAll("tbody tr")) {
      const cells = [...tr.children];
      const group = cells[0].querySelector("strong");
      if (group && cells.slice(1).every((c) => !c.textContent.trim())) {
        const label = group.textContent.trim();
        if (label === "more") {
          box = document.createElement("details");
          box.className = "opt-more";
          box.innerHTML = "<summary></summary>";
          list.append(box);
        } else {
          box = list;
          list.insertAdjacentHTML("beforeend", `<div class="opt-group">${esc(label)}</div>`);
        }
        continue;
      }
      const row = document.createElement("div");
      row.className = "opt";
      const key = cells[0].querySelector("code")?.textContent.replace(/^-+/, "").replace(/\(.*$/, "").replace(/[^\w-]/g, "") || slug(cells[0].textContent);
      row.id = owner ? `${owner}--${key}` : key;
      const head = document.createElement("div");
      head.className = "opt-name";
      head.innerHTML = `<a class="opt-link" href="#${row.id}" aria-label="link to this option">${cells[0].innerHTML}</a>`;
      const body = document.createElement("div");
      body.className = "opt-desc";
      cells.slice(1).forEach((c, j) => {
        const name = names[j + 1];
        if (!c.innerHTML.trim()) return;
        if (name === "default" && c.textContent.trim() === "required") head.insertAdjacentHTML("beforeend", '<span class="opt-req">required</span>');
        else if (name === "default") body.insertAdjacentHTML("beforeend", `<div class="opt-def"><span>default</span> ${c.innerHTML}</div>`);
        else if (["description", "meaning", "returns"].includes(name) && names.length === 2 + names.includes("default")) {
          body.insertAdjacentHTML("afterbegin", `<div>${c.innerHTML}</div>`);
        } else body.insertAdjacentHTML("beforeend", `<div><span class="opt-label">${esc(name)}</span> ${c.innerHTML}</div>`);
      });
      row.append(head, body);
      box.append(row);
    }
    list.querySelectorAll("details.opt-more").forEach((d) => {
      const n = d.querySelectorAll(".opt").length;
      d.querySelector("summary").textContent = `${n} more option${n === 1 ? "" : "s"}`;
    });
    t.replaceWith(list);
  }

  // narrow the reference to the options that match
  function filter(main) {
    const box = document.createElement("div");
    box.className = "filter";
    box.innerHTML = '<input type="search" placeholder="Filter options, e.g. lr, skin, stress" aria-label="Filter options"><span class="filter-n" aria-live="polite"></span>';
    const input = box.querySelector("input"), count = box.querySelector(".filter-n");
    main.querySelector(".toc").prepend(box);
    const opts = [...main.querySelectorAll(".opt")];
    const blocks = [...main.querySelectorAll(".doc > *")];
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      main.classList.toggle("filtering", !!q);
      let n = 0;
      for (const o of opts) {
        const hit = !q || o.textContent.toLowerCase().includes(q);
        o.hidden = !hit;
        n += hit;
      }
      main.querySelectorAll("details.opt-more").forEach((d) => {
        if (q) d.open = !!d.querySelector(".opt:not([hidden])");
      });
      // while filtering, show only the headings and lists that still hold a match
      let h2 = null, h3 = null;
      for (const el of blocks) {
        if (!q) { el.classList.remove("filtered-out"); continue; }
        if (el.tagName === "H2") { h2 = el; h3 = null; el.classList.add("filtered-out"); continue; }
        if (el.tagName === "H3") { h3 = el; el.classList.add("filtered-out"); continue; }
        const keep = el.classList.contains("opts") && el.querySelector(".opt:not([hidden])");
        el.classList.toggle("filtered-out", !keep);
        if (keep) { h2?.classList.remove("filtered-out"); h3?.classList.remove("filtered-out"); }
      }
      count.textContent = q ? `${n} match${n === 1 ? "" : "es"}` : "";
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && document.activeElement !== input && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
        e.preventDefault();
        input.focus();
      }
    });
  }

  function decorate(root) {
    // short inline code (flags, model names) is not broken at its hyphens
    root.querySelectorAll(":not(pre) > code").forEach((c) => c.textContent.length <= 28 && c.classList.add("nw"));
    root.querySelectorAll("pre > code").forEach((code) => {
      const lang = (code.className.match(/language-(\w+)/) || [])[1];
      highlight(code, lang);
      copyButton(code.parentElement);
      if (lang === "sh") installer(code);
    });
  }

  if (page !== "index.html") header();      // the home page has its links in the hero
  const main = document.querySelector("main");
  const ready = (main.dataset.md ? markdown(main) : Promise.resolve()).then(() => decorate(main));
  return { esc, highlight, copyButton, ready };
})();

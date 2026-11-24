// The run-length estimate of the models page: MD steps in a given wall time, for a system size.
(() => {
  const TIMES = [[10, "10 s"], [60, "1 min"], [600, "10 min"], [1800, "30 min"], [3600, "1 h"],
    [10800, "3 h"], [43200, "12 h"], [86400, "24 h"], [259200, "72 h"]];
  const MACE = '<a href="https://github.com/ACEsuit/mace" target="_blank" rel="noopener">MACE-MP-0 small</a>';
  const SERIES = [
    { id: "gpu", name: "ILMlip S", where: "NVIDIA B200 180GB" },
    { id: "cpu2", name: "ILMlip S", where: "Intel Core Ultra 5 125U (laptop), 2 cores", limit: 1.5e6, beyond: "out of memory (32 GB)" },
    { id: "mace", name: MACE, where: "NVIDIA B200 180GB", beyond: "out of GPU memory (180 GB)" },
  ];
  const N_MIN = 50, N_MAX = 1e7, N_START = 1000, T_MIN = 10, T_MAX = 259200, T_READ = 3600, H = 300;
  const PAD = { left: 52, right: 14, top: 24, bottom: 44 };
  const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";

  // atom-steps per second at n atoms: log-log interpolation, flat beyond the measured sizes
  function rate(points, n) {
    const x = Math.log10(n), xs = points.map((p) => Math.log10(p[0])), ys = points.map((p) => Math.log10(p[1]));
    if (x <= xs[0]) return 10 ** ys[0];
    if (x >= xs[xs.length - 1]) return 10 ** ys[ys.length - 1];
    let i = 1;
    while (xs[i] < x) i++;
    return 10 ** (ys[i - 1] + (x - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1]));
  }

  const round2 = (n) => { const p = 10 ** (Math.floor(Math.log10(n)) - 1); return Math.round(n / p) * p; };
  const sig = (v) => String(Number(v.toPrecision(3)));
  function steps(v) {
    if (v < 1) return "< 1";
    if (v < 1e3) return String(Math.round(v));
    if (v < 1e6) return `${sig(v / 1e3)}k`;
    if (v < 1e9) return `${sig(v / 1e6)}M`;
    return `${sig(v / 1e9)} × 10⁹`;
  }
  const decade = (k) => (k < 3 ? String(10 ** k) : k < 6 ? `${10 ** (k - 3)}k` : k < 9 ? `${10 ** (k - 6)}M`
    : `10${[...String(k)].map((d) => SUP[d]).join("")}`);
  // the MD time of so many steps at 1 fs each
  function mdTime(n) {
    for (const [div, unit] of [[1e12, "ms"], [1e9, "µs"], [1e6, "ns"], [1e3, "ps"]]) if (n >= div) return `${sig(n / div)} ${unit}`;
    return n < 1 ? "" : `${sig(n)} fs`;
  }

  function duration(t) {
    if (t < 59.5) return `${Math.round(t)} s`;
    if (t < 3570) return `${Math.round(t / 60)} min`;
    if (t < 172800) return `${sig(t / 3600)} h`;
    return `${sig(t / 86400)} days`;
  }

  function start(box, data) {
    box.innerHTML = `
      <div class="est-size"><label for="est-n">System size</label>
        <input type="range" id="est-n" min="0" max="1000" step="1">
        <output for="est-n"></output></div>
      <div class="est-read"></div>
      <ul class="est-legend"><li class="est-head"><span class="name"></span><span class="val">MD steps</span><span class="len">at 1 fs</span></li>
        ${SERIES.map((s) => `<li class="est-${s.id}"><span class="key"></span>
        <span class="name">${s.name} <span>· ${s.where}</span></span><span class="val"></span><span class="len"></span></li>`).join("")}</ul>
      <svg class="est-plot" height="${H}" role="img" aria-label="MD steps against wall time"></svg>`;
    const input = box.querySelector("input"), out = box.querySelector("output"), svg = box.querySelector("svg");
    const toN = (v) => round2(N_MIN * (N_MAX / N_MIN) ** (v / 1000));
    let n = N_START, read = T_READ;
    input.value = Math.round(1000 * Math.log(n / N_MIN) / Math.log(N_MAX / N_MIN));

    const live = () => SERIES.map((s) => {
      const limit = s.limit ?? data[s.id][data[s.id].length - 1][0];
      return { ...s, r: s.id === "gpu" || n <= limit ? rate(data[s.id], n) : null };
    });

    function legend(rows) {
      box.querySelector(".est-read").innerHTML = `In <b>${duration(read)}</b> with ${n.toLocaleString("en")} atoms`;
      rows.forEach((s) => {
        box.querySelector(`.est-${s.id} .val`).textContent = s.r ? steps(s.r * read / n) : s.beyond;
        box.querySelector(`.est-${s.id} .len`).textContent = s.r ? mdTime(s.r * read / n) : "";
        box.querySelector(`.est-${s.id}`).classList.toggle("off", !s.r);
      });
    }

    function draw() {
      const rows = live(), on = rows.filter((s) => s.r);
      const W = Math.max(300, svg.clientWidth || box.clientWidth);
      const X = (t) => PAD.left + Math.log10(t / T_MIN) / Math.log10(T_MAX / T_MIN) * (W - PAD.left - PAD.right);
      const top = Math.max(...on.map((s) => s.r * T_MAX / n)), low = Math.min(...on.map((s) => s.r * T_MIN / n));
      const hi = Math.ceil(Math.log10(top)), lo = Math.max(0, Math.min(hi - 2, Math.floor(Math.log10(low))));
      const Y = (v) => PAD.top + (hi - Math.log10(v)) / (hi - lo) * (H - PAD.top - PAD.bottom);
      let g = "";
      for (let k = lo; k <= hi; k++) {
        g += `<line class="grid" x1="${PAD.left}" x2="${W - PAD.right}" y1="${Y(10 ** k)}" y2="${Y(10 ** k)}"/>`
          + `<text class="tick" x="${PAD.left - 8}" y="${Y(10 ** k) + 4}" text-anchor="end">${decade(k)}</text>`;
      }
      let right = -Infinity;
      for (const [t, label] of TIMES) {
        const x = X(t), w = label.length * 6.2 + 4;
        g += `<line class="grid" x1="${x}" x2="${x}" y1="${PAD.top}" y2="${H - PAD.bottom}"/>`;
        if (x - w / 2 > right) {
          g += `<text class="tick" x="${x}" y="${H - PAD.bottom + 17}" text-anchor="middle">${label}</text>`;
          right = x + w / 2;
        }
      }
      g += `<text class="axis" x="${W - PAD.right}" y="${H - 6}" text-anchor="end">wall time</text>`
        + `<text class="axis" x="${PAD.left}" y="13">MD steps</text>`;
      for (const s of on) {
        const t0 = Math.max(T_MIN, 10 ** lo * n / s.r);
        if (t0 < T_MAX) g += `<line class="line est-${s.id}" x1="${X(t0)}" y1="${Y(s.r * t0 / n)}" x2="${X(T_MAX)}" y2="${Y(s.r * T_MAX / n)}"/>`;
      }
      const x = X(read);
      g += `<line class="cursor" x1="${x}" x2="${x}" y1="${PAD.top}" y2="${H - PAD.bottom}"/>`;
      for (const s of on) {
        const v = s.r * read / n;
        if (v >= 10 ** lo) g += `<circle class="dot est-${s.id}" cx="${x}" cy="${Y(v)}" r="4"/>`;
      }
      g += `<rect class="hit" x="${PAD.left}" y="${PAD.top}" width="${W - PAD.left - PAD.right}" height="${H - PAD.top - PAD.bottom}"/>`;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.innerHTML = g;
      svg.dataset.w = W;
      legend(rows);
    }

    input.addEventListener("input", () => {
      n = toN(input.value);
      out.textContent = `${n.toLocaleString("en")} atoms`;
      draw();
    });
    svg.addEventListener("pointermove", (e) => {
      const W = Number(svg.dataset.w), box2 = svg.getBoundingClientRect();
      const f = ((e.clientX - box2.left) * W / box2.width - PAD.left) / (W - PAD.left - PAD.right);
      read = T_MIN * (T_MAX / T_MIN) ** Math.min(1, Math.max(0, f));
      draw();
    });
    svg.addEventListener("pointerleave", () => { read = T_READ; draw(); });
    new ResizeObserver(() => draw()).observe(box);
    out.textContent = `${n.toLocaleString("en")} atoms`;
    draw();
  }

  site.ready.then(async () => {
    const box = document.querySelector(".estimate");
    if (!box) return;
    try {
      const r = await fetch("assets/speed.json");
      start(box, await r.json());
    } catch {
      box.textContent = "The estimate could not be loaded.";
    }
  });
})();

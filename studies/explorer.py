"""Build the results explorer: one self-contained HTML page (D3, land outlines embedded) from real-data
join-path LBF results - the evidence map with one arc per tested family pair, the null band of the
known-unrelated pairs, and for a selected pair the per-concept attribution of the evidence, the annotated
tree (event fractions per edge, t_root, borrowing arrows) and root-form logos.

Usage: PYTHONPATH=. python studies/explorer.py <out.html> <inputs...>
  inputs: results/realdata/lbf_*.json (studies/realdata_lbf.py outputs) and/or merged diag files
          (results/cloud/diag_<run>.json) whose cells include kind=realdata_lbf
Needs data/northeuralex_languages.csv and a world-atlas land-110m.json (path in LAND, env MASSCOMP_LAND)."""
import sys, os, json, datetime, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAND = os.environ.get("MASSCOMP_LAND", os.path.join(HERE, "..", "data", "land-110m.json"))


def load_inputs(paths):
    pairs = []
    for p in paths:
        d = json.load(open(p))
        if d.get("study") == "diag" and "cells" in d:          # a merged battery
            for c in d.get("cells", []):
                if c.get("kind") == "realdata_lbf":
                    r = dict(c["result"]); r["source"] = os.path.basename(p); pairs.append(r)
        elif d.get("study") == "diag" and d.get("kind") == "realdata_lbf":   # one cell
            r = dict(d["result"]); r["source"] = os.path.basename(p); pairs.append(r)
        elif "lbf_mean" in d:
            d["source"] = os.path.basename(p); pairs.append(d)
    return pairs


def languages():
    out = {}
    with open(os.path.join(HERE, "..", "data", "northeuralex_languages.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["ID"]] = dict(name=r["Name"], lat=float(r["Latitude"]), lon=float(r["Longitude"]), family=r["Family"], subfamily=r["Subfamily"])
    return out


def build(pairs, out_path, note=""):
    langs = languages()
    land = json.load(open(LAND))
    fams = {}
    for r in pairs:
        for name, sub in list(r["config"]["families"]) + list(r["config"].get("donors") or []):
            if name in fams:
                continue
            ls = []

            def walk(x):
                if isinstance(x, (list, tuple)):
                    for y in x:
                        walk(y)
                else:
                    ls.append(x)
            walk(sub)
            fams[name] = dict(langs=ls, tree=sub, lat=float(np.mean([langs[L]["lat"] for L in ls])), lon=float(np.mean([langs[L]["lon"] for L in ls])),
                              label=name.capitalize(), family=langs[ls[0]]["family"])
    # one record per pair (G=fit preferred when both exist; the flat run kept as a control note)
    recs = []
    for r in pairs:
        c = r["config"]
        ests = []
        for e in r["estimates"]:
            t = e.get("trace", {})
            ests.append(dict(lbf=e["lbf"], ti_total=t.get("ti_total"), ti_concept=t.get("ti_concept"), t_root=t.get("t_root_at1"),
                             t_root_sd=t.get("t_root_sd_at1"), omega=t.get("omega_at1"), lam=t.get("lam_at1"), events=t.get("events_at1"),
                             borrow=t.get("borrow_from_at1"), root_forms=t.get("root_forms_at1"), fam_forms=t.get("fam_forms_at1")))
        recs.append(dict(fam1=c["fam1"], fam2=c["fam2"], known=r["known"], deep=r["deep"], g=c["g"], n=c["n"], K=c["K"], sweeps=c["sweeps"],
                         cset=c.get("concepts", "all"), pool=c.get("pool"),
                         lbf=r["lbf_mean"], lbf_sd=r["lbf_sd"], estimates=ests, concepts=r.get("concepts"), leaf_forms=r.get("leaf_forms"),
                         tree=r.get("tree"), secs=r.get("secs"), source=r.get("source")))
    data = dict(generated=datetime.date.today().isoformat(), note=note, languages={k: v for k, v in langs.items()}, families=fams, pairs=recs)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__LAND__", json.dumps(land, separators=(",", ":")))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(html)
    return len(recs)


TEMPLATE = r'''<meta charset="utf-8">
<title>Masscomp Evidence Atlas</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gentium+Plus:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: an atlas plate on top (the map), a number line under it, then a two-column dossier for the selected pair. */
:root {
  --bg: #eef1f4; --panel: #ffffff; --ink: #1b2430; --muted: #5b6672; --line: #cfd6dd; --land: #dfe5ea; --sea: #eef1f4;
  --for: #177fa3; --against: #c75b2a; --borrow: #8d5fc2; --null: #9aa6b2; --band: rgba(154,166,178,0.18); --focus: #177fa3;
  --display: "Gentium Plus", "Charis SIL", "Times New Roman", serif; --ui: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #10151b; --panel: #182028; --ink: #e6ebf0; --muted: #9aa6b2; --line: #2c3640; --land: #232d37; --sea: #10151b;
  --for: #3aa3c9; --against: #e0773f; --borrow: #ab86d8; --null: #7d8995; --band: rgba(125,137,149,0.22); --focus: #3aa3c9; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #10151b; --panel: #182028; --ink: #e6ebf0; --muted: #9aa6b2; --line: #2c3640; --land: #232d37; --sea: #10151b;
  --for: #3aa3c9; --against: #e0773f; --borrow: #ab86d8; --null: #7d8995; --band: rgba(125,137,149,0.22); --focus: #3aa3c9; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--ui); font-size: 15px; line-height: 1.45;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 20px 48px; }
* { box-sizing: border-box; }
h1, h2, h3 { font-family: var(--display); font-weight: 400; margin: 0; text-wrap: balance; }
h1 { font-size: 2rem; line-height: 1.1; }
h2 { font-size: 1.35rem; margin-bottom: 4px; }
h3 { font-size: 1.1rem; }
.wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 22px; }
header.top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px; }
header.top p { margin: 0; color: var(--muted); max-width: 62ch; }
.eyebrow { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; min-width: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.85rem; color: var(--muted); align-items: center; }
.legend i { display: inline-block; width: 22px; height: 0; border-top: 3px solid var(--for); vertical-align: middle; margin-right: 6px; }
.legend i.ag { border-color: var(--against); } .legend i.nl { border-color: var(--null); border-top-style: dashed; }
.legend i.dp { border-color: var(--for); border-top-style: dotted; }
.legend b { display: inline-block; width: 11px; height: 11px; border-radius: 2px; vertical-align: -1px; margin-right: 6px; }
svg { display: block; max-width: 100%; height: auto; }
.scroll { overflow-x: auto; min-width: 0; }
.scroll > svg { min-width: 520px; }
#line { min-width: 760px; }
svg text { font-family: var(--ui); fill: var(--ink); }
.map-wrap { position: relative; }
#map .land { fill: var(--land); stroke: none; }
#map .graticule { fill: none; stroke: var(--line); stroke-width: 0.5; stroke-opacity: 0.7; }
#map .arc { fill: none; stroke-linecap: round; cursor: pointer; transition: opacity .15s; }
#map .arc.hit { stroke: transparent; stroke-width: 16; }
#map .arc.sel { filter: drop-shadow(0 0 3px rgba(0,0,0,0.35)); }
#map .fam-dot { fill: var(--panel); stroke: var(--ink); stroke-width: 1.4; }
#map .lang-dot { fill: var(--ink); fill-opacity: 0.55; }
#map .fam-label { font-family: var(--display); font-size: 14px; paint-order: stroke; stroke: var(--panel); stroke-width: 3px; stroke-linejoin: round; }
#map .lbf-label { font-size: 11px; font-weight: 600; paint-order: stroke; stroke: var(--panel); stroke-width: 3px; stroke-linejoin: round; }
#map .dim { opacity: 0.25; }
.tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 6px 9px;
  font-size: 0.82rem; box-shadow: 0 4px 14px rgba(0,0,0,0.12); max-width: 300px; z-index: 5; }
.tip b { font-family: var(--display); font-size: 1rem; font-weight: 400; }
.row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
@media (max-width: 760px) { .row { grid-template-columns: minmax(0, 1fr); } h1 { font-size: 1.6rem; } }
.stat { display: flex; flex-wrap: wrap; gap: 10px 28px; align-items: baseline; }
.stat .big { font-family: var(--display); font-size: 2.2rem; line-height: 1; font-variant-numeric: tabular-nums; }
.stat .big.for { color: var(--for); } .stat .big.against { color: var(--against); } .stat .big.null { color: var(--null); }
.stat small { color: var(--muted); display: block; font-size: 0.78rem; }
.ipa { font-family: var(--display); font-size: 1.05rem; }
.muted { color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; font-size: 0.88rem; }
select, button { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 5px 8px; }
button { cursor: pointer; } button:hover { border-color: var(--ink); }
select:focus-visible, button:focus-visible, .arc:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.seg { margin-left: auto; display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); font-size: 0.82rem; max-width: 100%; }
.seg button { border: 0; border-radius: 0; padding: 6px 10px; background: transparent; color: var(--muted); }
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-selected="true"] { background: var(--ink); color: var(--panel); }
.seg small { display: block; font-size: 0.7rem; opacity: 0.8; }
@media (max-width: 640px) { .seg { margin-left: 0; } }
.line .ghost { fill: none; stroke: var(--muted); stroke-width: 1.2; }
.line .ghost-link { stroke: var(--muted); stroke-width: 1; stroke-opacity: 0.45; }
.bars .lab { font-size: 12px; fill: var(--ink); }
.bars .val { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.bars .leaf { font-family: var(--display); font-size: 12.5px; fill: var(--muted); }
.axis line, .axis path { stroke: var(--line); } .axis text { fill: var(--muted); font-size: 11px; }
.tree .edge { stroke: var(--line); stroke-width: 1.2; fill: none; }
.tree .strip { stroke: none; }
.tree .nlab { font-family: var(--display); font-size: 13.5px; }
.tree .elab { font-size: 10.5px; fill: var(--muted); }
.tree .borrow { fill: none; stroke: var(--borrow); stroke-width: 1.6; }
.logos { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.logo { border: 1px solid var(--line); border-radius: 4px; padding: 8px 10px; min-width: 0; }
.logo h4 { margin: 0 0 4px; font-weight: 500; font-size: 0.85rem; display: flex; justify-content: space-between; gap: 8px; }
.logo .seg { font-family: var(--display); font-size: 15px; font-weight: 700; }
.logo .leaves { font-family: var(--display); font-size: 0.9rem; color: var(--muted); margin-top: 4px; line-height: 1.35; }
.line .pair { cursor: pointer; }
.line .pair:hover circle { stroke: var(--ink); }
.line .plab { font-size: 10.5px; fill: var(--muted); }
.foot { color: var(--muted); font-size: 0.82rem; max-width: 80ch; }
.foot p { margin: 4px 0; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header class="top">
    <div>
      <div class="eyebrow">masscomp &middot; joint-forest model &middot; NorthEuraLex wordlists</div>
      <h1>Evidence Atlas</h1>
    </div>
    <p>Each arc is a tested pair of language families, weighted by the log Bayes factor for a shared ancestor over separate ones. Faint dashed arcs sit inside the noise band of pairs known to be unrelated. Select an arc to see which words carry the evidence, what happened on each branch, and the reconstructed roots.</p>
  </header>
  <div id="note" class="panel" style="border-color:var(--against);font-size:0.9rem" hidden></div>

  <section class="panel map-wrap">
    <div class="legend" id="legend">
      <span><i></i>evidence for a shared ancestor</span><span><i class="ag"></i>evidence against</span><span><i class="nl"></i>inside the unrelated band</span><span><i class="dp"></i>related only at Indo-European depth</span>
      <div class="seg" id="setsel" role="tablist" aria-label="Concept set" hidden></div>
    </div>
    <svg id="map" viewBox="0 0 1140 520" role="img" aria-label="Map of tested family pairs"></svg>
    <div class="tip" id="tip" hidden></div>
  </section>

  <section class="panel">
    <div class="controls">
      <span class="eyebrow">All pairs on one scale</span>
      <label for="pairsel">Pair <select id="pairsel"></select></label>
      <span class="muted" id="bandnote"></span>
    </div>
    <div class="scroll"><svg id="line" class="line" viewBox="0 0 1140 140" role="img" aria-label="Log Bayes factors of all pairs"></svg></div>
  </section>

  <section id="dossier" class="wrap" style="gap:18px">
    <div class="panel">
      <div class="stat" id="stat"></div>
    </div>
    <div class="row">
      <div class="panel">
        <h2 id="treeh">What happened on each branch</h2>
        <div class="muted" style="font-size:0.85rem">Fitted joined tree. Each branch carries the posterior share of words inherited <b style="color:var(--for)">&#9632;</b>, replaced <b style="color:var(--against)">&#9632;</b> or borrowed <b style="color:var(--borrow)">&#9632;</b>; violet arrows show borrowing routes the model used.</div>
        <div class="scroll"><svg id="tree" class="tree" viewBox="0 0 560 360" role="img" aria-label="Annotated family tree"></svg></div>
      </div>
      <div class="panel">
        <h2>The words doing the work</h2>
        <div class="muted" style="font-size:0.85rem" id="barsnote"></div>
        <div class="scroll"><svg id="bars" class="bars" role="img" aria-label="Per-concept evidence"></svg></div>
      </div>
    </div>
    <div class="panel">
      <h2>Reconstructed roots</h2>
      <div class="muted" style="font-size:0.85rem">Posterior over the joined root form for the concepts that carry most evidence, as a logo: the height of a segment is its posterior share at that position, aligned to the modal form (&empty; marks a gap). Daughter forms below.</div>
      <div class="logos" id="logos"></div>
    </div>
  </section>

  <section class="foot">
    <p id="footnote"></p>
    <p>Trees are fixed by the study (three languages per family, binary); only the join and its length t<sub>root</sub> are inferred. The Bayes factor is estimated along the join path and the per-concept attribution by thermodynamic integration over the same stones; the two totals differ by the ladder's discretisation. Data: NorthEuraLex 0.9 (Dellert et al. 2020, CC-BY-4.0), mapped onto a 40-segment inventory.</p>
  </section>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js"></script>
<script>
const DATA = __DATA__;
const LAND = __LAND__;
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (x, d = 1) => (x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(d));
const famLabel = (f) => DATA.families[f] ? DATA.families[f].label : f;
const pairName = (p) => `${famLabel(p.fam1)} – ${famLabel(p.fam2)}`;

// one table per concept set (the full list, basic vocabulary); within a set prefer the fitted-G run for a pair and keep flat runs as controls
const pairKey = (p) => [p.fam1, p.fam2].sort().join("|");
const SET_LABEL = { all: ["Full list", "1,016 concepts"], basic: ["Basic vocabulary", "Swadesh + Leipzig-Jakarta"] };
const setLabel = (s) => (SET_LABEL[s] || [s, ""])[0];
const byKey = {};
for (const p of DATA.pairs) {
  const s = p.cset || "all", k = pairKey(p);
  byKey[s] = byKey[s] || {};
  if (!byKey[s][k] || (p.g === "fit" && byKey[s][k].g !== "fit")) byKey[s][k] = p;
}
const SETS = Object.keys(byKey).sort((a, b) => (a === "all" ? -1 : b === "all" ? 1 : a.localeCompare(b)));
const pairsOf = (s) => Object.values(byKey[s] || {}).sort((a, b) => b.lbf - a.lbf);
const bandOf = (s) => {
  const un = pairsOf(s).filter(p => p.known === "unrelated").map(p => p.lbf);
  return { unrelated: un, tau: un.length ? d3.quantile(un.slice().sort(d3.ascending), 0.99) : null, lo: un.length ? d3.min(un) : null, hi: un.length ? d3.max(un) : null };
};
const state = { set: SETS[0], sel: null };
let PAIRS = pairsOf(state.set), BAND = bandOf(state.set), unrelated = BAND.unrelated, tau = BAND.tau, lo = BAND.lo, hi = BAND.hi;
state.sel = PAIRS.length ? PAIRS[0] : null;
const counterpart = (p) => { for (const s of SETS) if (s !== state.set && byKey[s][pairKey(p)]) return byKey[s][pairKey(p)]; return null; };

function setSet(s) {
  if (!byKey[s]) return;
  const k = state.sel ? pairKey(state.sel) : null;
  state.set = s; PAIRS = pairsOf(s); BAND = bandOf(s); unrelated = BAND.unrelated; tau = BAND.tau; lo = BAND.lo; hi = BAND.hi;
  state.sel = (k && byKey[s][k]) || (PAIRS.length ? PAIRS[0] : null);
  drawSetSwitch(); drawMap(); pairSelect(); drawLine(); drawDossier();
}
function drawSetSwitch() {
  const el = document.getElementById("setsel"); el.innerHTML = "";
  if (SETS.length < 2) { el.hidden = true; return; }
  el.hidden = false;
  for (const s of SETS) {
    const b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "tab"); b.setAttribute("aria-selected", String(s === state.set));
    const [name, sub] = SET_LABEL[s] || [s, ""];
    b.innerHTML = `${name}<small>${sub}</small>`; b.onclick = () => setSet(s); el.appendChild(b);
  }
}

function arcClass(p) { if (hi != null && p.lbf <= hi) return "null"; return p.lbf > 0 ? "for" : "against"; }
function arcColor(p) { const c = arcClass(p); return c === "null" ? css("--null") : (c === "for" ? css("--for") : css("--against")); }

// ---------- map ----------
function drawMap() {
  const svg = d3.select("#map"); svg.selectAll("*").remove();
  const W = 1140, H = 520;
  const land = topojson.feature(LAND, LAND.objects.land);
  const pts = Object.values(DATA.languages).map(l => [l.lon, l.lat]);
  const famPts = Object.values(DATA.families).map(f => [f.lon, f.lat]);
  const proj = d3.geoConicEqualArea().parallels([40, 62]).rotate([-40, 0]).clipExtent([[0, 0], [W, H]]);
  const box = { type: "MultiPoint", coordinates: pts.concat(famPts) };
  proj.fitExtent([[60, 40], [W - 60, H - 40]], box);
  const path = d3.geoPath(proj);
  const g = svg.append("g");
  g.append("path").datum(land).attr("class", "land").attr("d", path);
  g.append("path").datum(d3.geoGraticule().step([10, 10])).attr("class", "graticule").attr("d", path);
  // arcs (great circles between family centroids)
  const arcs = g.append("g");
  const widths = d3.scaleSqrt().domain([0, d3.max(PAIRS, p => Math.abs(p.lbf)) || 1]).range([1.2, 9]);
  for (const p of PAIRS.slice().reverse()) {
    const a = DATA.families[p.fam1], b = DATA.families[p.fam2];
    const line = { type: "LineString", coordinates: [[a.lon, a.lat], [b.lon, b.lat]] };
    const cls = arcClass(p);
    const el = arcs.append("path").datum(p).attr("class", "arc").attr("d", path(line))
      .attr("stroke", arcColor(p)).attr("stroke-width", widths(Math.abs(p.lbf)))
      .attr("stroke-dasharray", cls === "null" ? "4 6" : (p.deep ? "1 7" : null)).attr("stroke-opacity", cls === "null" ? 0.7 : 0.9)
      .attr("tabindex", 0).attr("role", "button").attr("aria-label", `${pairName(p)}: log Bayes factor ${fmt(p.lbf)}`);
    arcs.append("path").datum(p).attr("class", "arc hit").attr("d", path(line))
      .on("mousemove", (ev, d) => showTip(ev, d)).on("mouseleave", hideTip).on("click", (ev, d) => select(d));
    el.on("mousemove", (ev, d) => showTip(ev, d)).on("mouseleave", hideTip).on("click", (ev, d) => select(d))
      .on("keydown", (ev, d) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(d); } });
    // midpoint label
    const mid = d3.geoInterpolate([a.lon, a.lat], [b.lon, b.lat])(0.5);
    const [mx, my] = proj(mid);
    arcs.append("text").datum(p).attr("class", "lbf-label").attr("x", mx).attr("y", my - 6).attr("text-anchor", "middle").attr("fill", arcColor(p)).text(fmt(p.lbf, 0));
  }
  // families and languages
  const fg = g.append("g");
  for (const [name, f] of Object.entries(DATA.families)) {
    const [cx, cy] = proj([f.lon, f.lat]);
    for (const L of f.langs) {
      const l = DATA.languages[L]; const [x, y] = proj([l.lon, l.lat]);
      fg.append("line").attr("x1", cx).attr("y1", cy).attr("x2", x).attr("y2", y).attr("stroke", css("--muted")).attr("stroke-width", 0.6).attr("stroke-opacity", 0.6);
      fg.append("circle").attr("class", "lang-dot").attr("cx", x).attr("cy", y).attr("r", 2.2).append("title").text(l.name);
    }
    fg.append("circle").attr("class", "fam-dot").attr("cx", cx).attr("cy", cy).attr("r", 6);
    fg.append("text").attr("class", "fam-label").attr("x", cx + 9).attr("y", cy + 4).text(f.label + " (" + f.langs.join(", ") + ")");
  }
  highlightMap();
}
function highlightMap() {
  d3.selectAll("#map .arc:not(.hit)").classed("sel", d => state.sel && d === state.sel).classed("dim", d => state.sel && d !== state.sel && false);
}
function showTip(ev, p) {
  const t = document.getElementById("tip");
  const r = document.querySelector("#map").getBoundingClientRect();
  t.innerHTML = `<b>${pairName(p)}</b><br>log BF ${fmt(p.lbf)}${p.lbf_sd != null ? " ± " + p.lbf_sd.toFixed(1) + " (replicates)" : ""}<br><span class="muted">${p.known === "related" ? (p.deep ? "known related, Indo-European depth" : "known related") : "known unrelated"} · ${p.n} ${setLabel(p.cset || "all").toLowerCase()} concepts · G ${p.g}</span>`;
  t.hidden = false;
  const x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
  t.style.left = Math.min(x, r.width - 310) + "px"; t.style.top = y + "px";
}
function hideTip() { document.getElementById("tip").hidden = true; }

// ---------- number line ----------
function drawLine() {
  const svg = d3.select("#line"); svg.selectAll("*").remove();
  const W = 1140, H = 140, m = { l: 30, r: 30 };
  const ext = d3.extent(PAIRS, p => p.lbf); const pad = Math.max(2, (ext[1] - ext[0]) * 0.08);
  const x = d3.scaleLinear().domain([Math.min(ext[0] - pad, -2), Math.max(ext[1] + pad, 2)]).range([m.l, W - m.r]);
  if (tau != null) {
    svg.append("rect").attr("x", x(lo)).attr("y", 20).attr("width", Math.max(2, x(hi) - x(lo))).attr("height", 78).attr("fill", css("--band"));
    svg.append("text").attr("x", (x(lo) + x(hi)) / 2).attr("y", 14).attr("text-anchor", "middle").attr("fill", css("--muted")).style("font-size", "11px").text("unrelated pairs");
  }
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,100)`).call(d3.axisBottom(x).ticks(10));
  svg.append("text").attr("x", W - m.r).attr("y", 132).attr("text-anchor", "end").attr("fill", css("--muted")).style("font-size", "11px").text("log Bayes factor, shared ancestor vs separate (nats)");
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 20).attr("y2", 100).attr("stroke", css("--line"));
  // the same pair on the other concept set: a hollow marker joined to the active one, so the shift reads directly
  let ghosts = 0;
  const gg = svg.append("g");
  for (const p of PAIRS) {
    const q = counterpart(p); if (!q) continue; ghosts++;
    gg.append("line").attr("class", "ghost-link").attr("x1", x(q.lbf)).attr("x2", x(p.lbf)).attr("y1", 50).attr("y2", 50);
    gg.append("circle").attr("class", "ghost").attr("cx", x(q.lbf)).attr("cy", 50).attr("r", 3.5).append("title").text(`${pairName(p)} on ${setLabel(q.cset || "all").toLowerCase()}: ${fmt(q.lbf)}`);
  }
  // stagger labels
  const sorted = PAIRS.slice().sort((a, b) => a.lbf - b.lbf);
  const g = svg.append("g");
  const rowY = [-14, 22, -28, 36], rowLast = [-1e9, -1e9, -1e9, -1e9], minGap = 64;
  sorted.forEach((p, i) => {
    const gx = g.append("g").datum(p).attr("class", "pair").attr("transform", `translate(${x(p.lbf)},50)`).on("click", (ev, d) => select(d)).on("mousemove", (ev, d) => showTipLine(ev, d)).on("mouseleave", hideTip);
    gx.append("circle").attr("r", state.sel === p ? 7 : 5).attr("fill", arcColor(p)).attr("stroke", state.sel === p ? css("--ink") : css("--panel")).attr("stroke-width", 1.5);
    if (p.deep) gx.append("circle").attr("r", 9).attr("fill", "none").attr("stroke", arcColor(p)).attr("stroke-dasharray", "1 3");
    let row = rowY.findIndex((_, r) => x(p.lbf) - rowLast[r] >= minGap); if (row < 0) row = i % rowY.length; rowLast[row] = x(p.lbf);
    gx.append("text").attr("class", "plab").attr("y", rowY[row]).attr("text-anchor", "middle").text(`${famLabel(p.fam1).slice(0, 5)}–${famLabel(p.fam2).slice(0, 5)}`);
  });
  const other = SETS.find(s => s !== state.set);
  document.getElementById("bandnote").textContent = (tau != null ? `Noise band: ${unrelated.length} known-unrelated pairs span ${fmt(lo)} to ${fmt(hi)} nats.` : "")
    + (ghosts ? ` Hollow markers: the same pair on ${setLabel(other).toLowerCase()}.` : "");
}
function showTipLine(ev, p) { const t = document.getElementById("tip"); const r = document.querySelector("#map").getBoundingClientRect(); showTip(ev, p); t.style.top = (ev.clientY - r.top + 14) + "px"; }

// ---------- dossier ----------
function pairSelect() {
  const sel = document.getElementById("pairsel"); sel.innerHTML = "";
  for (const p of PAIRS) { const o = document.createElement("option"); o.value = [p.fam1, p.fam2].join("|"); o.textContent = `${pairName(p)}  (${fmt(p.lbf)})`; sel.appendChild(o); }
  sel.value = state.sel ? [state.sel.fam1, state.sel.fam2].join("|") : "";
  sel.onchange = () => { const p = PAIRS.find(q => [q.fam1, q.fam2].join("|") === sel.value); if (p) select(p); };
}
function select(p) {
  state.sel = p; highlightMap(); drawLine(); document.getElementById("pairsel").value = [p.fam1, p.fam2].join("|"); drawDossier();
}
function meanOf(arr) { return arr.length ? d3.mean(arr) : null; }
function drawDossier() {
  const p = state.sel; if (!p) return;
  const cls = arcClass(p);
  const verdict = cls === "null" ? "inside the unrelated band" : (p.lbf > 0 ? "evidence for a shared ancestor" : "evidence against a shared ancestor");
  const ests = p.estimates;
  const tr = meanOf(ests.map(e => e.t_root).filter(v => v != null));
  const om = meanOf(ests.map(e => e.omega).filter(v => v != null));
  const q = counterpart(p);
  const qcls = q ? (() => { const b = bandOf(q.cset || "all"); return b.hi != null && q.lbf <= b.hi ? "null" : (q.lbf > 0 ? "for" : "against"); })() : "";
  document.getElementById("stat").innerHTML = `
    <div><div class="big ${cls}">${fmt(p.lbf)}</div><small>log Bayes factor, ${pairName(p)}${p.lbf_sd != null ? ` · replicates ± ${p.lbf_sd.toFixed(1)}` : ""}</small></div>
    <div><div style="font-family:var(--display);font-size:1.3rem">${verdict}</div><small>${p.known === "related" ? (p.deep ? "known to be related, at Indo-European depth" : "known to be related") : "known to be unrelated"} · ${p.n} concepts · G ${p.g === "fit" ? "fitted to the daughters" : "flat"}</small></div>
    ${q ? `<div><div class="big ${qcls}" style="font-size:1.6rem">${fmt(q.lbf)}</div><small>the same pair on ${setLabel(q.cset || "all").toLowerCase()}${q.lbf_sd != null ? ` · ± ${q.lbf_sd.toFixed(1)}` : ""}</small></div>` : ""}
    ${tr != null ? `<div><div class="big" style="font-size:1.6rem">${tr.toFixed(2)}</div><small>t<sub>root</sub>, length of the join edges (posterior mean)</small></div>` : ""}
    ${om != null ? `<div><div class="big" style="font-size:1.6rem">${om.toFixed(3)}</div><small>&omega;, borrowing rate</small></div>` : ""}`;
  drawTree(p); drawBars(p); drawLogos(p);
  document.getElementById("footnote").textContent = `${pairName(p)}: ${p.n} concepts drawn from ${p.pool ? p.pool + " available on " : ""}${setLabel(p.cset || "all").toLowerCase()}; ${p.K} stones × ${p.sweeps} sweeps per estimate, ${ests.length} replicate${ests.length > 1 ? "s" : ""}; ${p.source || ""}`;
}

// tree: nodes from the stored forest (parent array), horizontal, root at left
function drawTree(p) {
  const svg = d3.select("#tree"); svg.selectAll("*").remove();
  if (!p.tree) return;
  const T = p.tree, n = T.parent.length;
  const ests = p.estimates.filter(e => e.events);
  const ev = {}; const bor = {};
  for (let e = 0; e < n; e++) if (T.parent[e] >= 0) {
    const vals = ests.map(x => x.events[String(e)]).filter(Boolean);
    ev[e] = vals.length ? { inherit: d3.mean(vals, v => v.inherit), replace: d3.mean(vals, v => v.replace), borrow: d3.mean(vals, v => v.borrow) } : null;
  }
  for (const x of ests) for (const [k, v] of Object.entries(x.borrow || {})) bor[k] = (bor[k] || 0) + v / ests.length;
  const tr = meanOf(ests.map(e => e.t_root).filter(v => v != null)) ?? 1.0;
  const roots = []; for (let v = 0; v < n; v++) if (T.parent[v] < 0) roots.push(v);
  // the joined root is the one outside every family; other roots are donor families, drawn below it
  const R = roots.find(v => (T.family[v] ?? -1) < 0) ?? roots[0];
  const donorRoots = roots.filter(v => v !== R);
  const children = {}; for (let v = 0; v < n; v++) children[v] = [];
  for (let v = 0; v < n; v++) if (T.parent[v] >= 0) children[T.parent[v]].push(v);
  const len = (v) => (T.parent[v] === R ? tr : T.length[v]);
  const depth = {}; const order = [];
  const visit = (v) => { order.push(v); for (const c of children[v]) { depth[c] = depth[v] + len(c); visit(c); } };   // pre-order: each family's leaves stay together
  for (const r of [R, ...donorRoots]) { depth[r] = 0; visit(r); }
  const leaves = order.filter(v => children[v].length === 0);
  const nDonorLeaves = leaves.filter(v => { let u = v; while (T.parent[u] >= 0) u = T.parent[u]; return u !== R; }).length;
  const W = 560, H = 360 + (donorRoots.length ? 30 + 22 * nDonorLeaves : 0), ml = 24, mr = 118, mt = 26, mb = 20;
  svg.attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear().domain([0, d3.max(leaves, v => depth[v])]).range([ml, W - mr]);
  const y = {}; let slot = 0;
  leaves.forEach((v) => { let u = v; while (T.parent[u] >= 0) u = T.parent[u]; if (u !== R && slot === leaves.length - nDonorLeaves) slot += 1; y[v] = mt + (slot + 0.5) * (H - mt - mb) / (leaves.length + (donorRoots.length ? 1 : 0)); slot += 1; });
  const post = order.slice().reverse(); for (const v of post) if (children[v].length) y[v] = d3.mean(children[v], c => y[c]);
  const g = svg.append("g");
  const col = { inherit: css("--for"), replace: css("--against"), borrow: css("--borrow") };
  for (const v of order) if (T.parent[v] >= 0) {
    const u = T.parent[v];
    g.append("path").attr("class", "edge").attr("d", `M${x(depth[u])},${y[u]} V${y[v]} H${x(depth[v])}`);
    const e = ev[v]; if (e) {
      const x0 = x(depth[u]) + 3, x1 = x(depth[v]) - 2, w = Math.max(0, x1 - x0); let acc = x0;
      for (const k of ["inherit", "replace", "borrow"]) {
        const ww = w * e[k]; if (ww > 0.5) g.append("rect").attr("class", "strip").attr("x", acc).attr("y", y[v] - 4).attr("width", ww).attr("height", 8).attr("fill", col[k]).append("title").text(`${k}: ${(100 * e[k]).toFixed(0)}%`);
        acc += ww;
      }
    }
    if (T.parent[v] === R) g.append("text").attr("class", "elab").attr("x", (x(depth[u]) + x(depth[v])) / 2).attr("y", y[v] - 8).attr("text-anchor", "middle").text(`t_root ${tr.toFixed(2)}`);
  }
  for (const v of order) {
    const name = T.names[String(v)]; const isLeaf = children[v].length === 0;
    let label = isLeaf ? (DATA.languages[name] ? DATA.languages[name].name : name) : (v === R ? "joined root" : (name && name.includes("/") ? "" : famLabel(name) + (donorRoots.includes(v) ? " (donor, not joined)" : "")));
    if (isLeaf) label = label.replace(/\s*\(.*\)$/, "");
    g.append("circle").attr("cx", x(depth[v])).attr("cy", y[v]).attr("r", isLeaf ? 3 : 3.5).attr("fill", isLeaf ? css("--ink") : css("--panel")).attr("stroke", css("--ink")).attr("stroke-width", 1.2);
    const atRoot = v === R || donorRoots.includes(v);
    if (label) g.append("text").attr("class", "nlab").attr("x", x(depth[v]) + (isLeaf ? 8 : (atRoot ? 0 : -8))).attr("y", y[v] + (isLeaf ? 4.5 : (atRoot ? 16 : -10)))
      .attr("text-anchor", isLeaf ? "start" : (atRoot ? "start" : "end")).text(label);
  }
  // borrowing arrows: key "e<-d"
  const defs = svg.append("defs");
  defs.append("marker").attr("id", "arrow").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", css("--borrow"));
  for (const [k, v] of Object.entries(bor)) {
    if (v < 0.02) continue;
    const [e, d] = k.split("<-").map(Number);
    const x1 = x(depth[d]), y1 = y[d], x2 = x(depth[e]), y2 = y[e];
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2 + (y2 > y1 ? -40 : 40);
    g.append("path").attr("class", "borrow").attr("d", `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`).attr("stroke-opacity", Math.min(1, 0.3 + 4 * v)).attr("marker-end", "url(#arrow)")
      .append("title").text(`${T.names[String(e)] || e} borrows from ${T.names[String(d)] || d}: ${(100 * v).toFixed(1)}% of words`);
  }
  document.getElementById("treeh").textContent = `What happened on each branch`;
}

function conceptLabel(c) { return String(c).replace(/^\d+_/, ""); }
function contributions(p) {
  const ests = p.estimates.filter(e => e.ti_concept);
  if (!ests.length || !p.concepts) return [];
  const M = p.concepts.length;
  return p.concepts.map((c, i) => ({ c, i, v: d3.mean(ests, e => e.ti_concept[i]) })).sort((a, b) => b.v - a.v);
}
function drawBars(p) {
  const svg = d3.select("#bars"); svg.selectAll("*").remove();
  const rows = contributions(p);
  const ests = p.estimates.filter(e => e.ti_total != null);
  document.getElementById("barsnote").textContent = rows.length ? `Attribution of the evidence to concepts (thermodynamic integration, summing to ${fmt(d3.mean(ests, e => e.ti_total))} nats). Hover a bar for the daughter forms.` : "No per-concept trace in this run.";
  if (!rows.length) return;
  const rh = 17, W = 560, ml = 128, mr = 56, mt = 8, H = mt + rows.length * rh + 24;
  svg.attr("viewBox", `0 0 ${W} ${H}`);
  const ext = d3.max(rows, r => Math.abs(r.v)) || 1;
  const x = d3.scaleLinear().domain([-ext, ext]).range([ml, W - mr]);
  const g = svg.append("g");
  g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", mt).attr("y2", H - 24).attr("stroke", css("--line"));
  rows.forEach((r, i) => {
    const yy = mt + i * rh;
    const gr = g.append("g").attr("transform", `translate(0,${yy})`);
    gr.append("rect").attr("x", Math.min(x(0), x(r.v))).attr("y", 2).attr("width", Math.abs(x(r.v) - x(0))).attr("height", rh - 4).attr("rx", 2)
      .attr("fill", r.v >= 0 ? css("--for") : css("--against")).attr("fill-opacity", 0.85)
      .append("title").text(`${conceptLabel(r.c)}: ${fmt(r.v, 2)} nats\n` + Object.entries(p.leaf_forms || {}).map(([L, f]) => `${L} ${f[r.c]}`).join("  "));
    gr.append("text").attr("class", "lab").attr("x", ml - 6).attr("y", rh - 5).attr("text-anchor", "end").text(conceptLabel(r.c));
    gr.append("text").attr("class", "val").attr("x", r.v >= 0 ? x(r.v) + 4 : x(r.v) - 4).attr("y", rh - 5).attr("text-anchor", r.v >= 0 ? "start" : "end").text(fmt(r.v, 1));
  });
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 22})`).call(d3.axisBottom(x).ticks(6));
}

// ---------- logos ----------
function align(ref, s) {
  // Levenshtein alignment of s to ref; returns per-ref-position symbol (segment or ∅)
  const n = ref.length, m = s.length;
  const D = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) D[i][0] = i; for (let j = 0; j <= m; j++) D[0][j] = j;
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) D[i][j] = Math.min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] === s[j - 1] ? 0 : 1));
  const out = new Array(n).fill("∅"); let i = n, j = m;
  while (i > 0 && j > 0) {
    if (D[i][j] === D[i - 1][j - 1] + (ref[i - 1] === s[j - 1] ? 0 : 1)) { out[i - 1] = s[j - 1]; i--; j--; }
    else if (D[i][j] === D[i - 1][j] + 1) { out[i - 1] = "∅"; i--; }
    else { j--; }
  }
  return out;
}
function drawLogos(p) {
  const box = document.getElementById("logos"); box.innerHTML = "";
  const rows = contributions(p);
  const est = p.estimates.find(e => e.root_forms && Object.keys(e.root_forms).length);
  if (!est || !rows.length) { box.innerHTML = `<div class="muted">No root-form samples in this run.</div>`; return; }
  const top = rows.slice(0, 8);
  for (const r of top) {
    const forms = est.root_forms[String(r.i)]; if (!forms || !forms.length) continue;
    const ref = Array.from(forms[0][0]);
    const cols = ref.map(() => ({}));
    let tot = 0;
    for (const [f, pr] of forms) { const al = align(ref, Array.from(f)); al.forEach((s, k) => cols[k][s] = (cols[k][s] || 0) + pr); tot += pr; }
    const cw = 26, H = 70, W = Math.max(60, cw * ref.length + 8);
    const div = document.createElement("div"); div.className = "logo";
    div.innerHTML = `<h4><span>${conceptLabel(r.c)}</span><span class="muted">${fmt(r.v, 1)} nats</span></h4>`;
    const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    cols.forEach((col, k) => {
      const entries = Object.entries(col).sort((a, b) => b[1] - a[1]); let y0 = H - 4;
      for (const [s, pr] of entries) {
        const h = (H - 8) * pr / tot; if (h < 1) continue;
        const gg = svg.append("g");
        gg.append("rect").attr("x", 4 + k * cw).attr("y", y0 - h).attr("width", cw - 3).attr("height", h).attr("rx", 2)
          .attr("fill", s === "∅" ? css("--null") : css("--for")).attr("fill-opacity", s === "∅" ? 0.35 : 0.12 + 0.7 * pr / tot);
        if (h >= 12) gg.append("text").attr("class", "seg").attr("x", 4 + k * cw + (cw - 3) / 2).attr("y", y0 - h / 2 + 5).attr("text-anchor", "middle").text(s);
        gg.append("title").text(`${s === "∅" ? "gap" : s}: ${(100 * pr / tot).toFixed(0)}%`);
        y0 -= h;
      }
    });
    div.appendChild(svg.node());
    const lf = p.leaf_forms || {};
    div.insertAdjacentHTML("beforeend", `<div class="leaves">${Object.entries(lf).map(([L, f]) => `<span title="${DATA.languages[L] ? DATA.languages[L].name : L}">${L} <b>${f[r.c]}</b></span>`).join(" · ")}</div>`);
    box.appendChild(div);
  }
}

function boot() {
  if (!PAIRS.length) { document.getElementById("stat").textContent = "No results yet."; return; }
  if (DATA.note) { const n = document.getElementById("note"); n.textContent = DATA.note; n.hidden = false; }
  drawSetSwitch(); drawMap(); pairSelect(); drawLine(); drawDossier();
  window.addEventListener("resize", () => {});
}
boot();
</script>
'''


if __name__ == "__main__":
    out = sys.argv[1]; ins = [a for a in sys.argv[2:] if not a.startswith("note=")]
    note = next((a[5:] for a in sys.argv[2:] if a.startswith("note=")), "")
    n = build(load_inputs(ins), out, note)
    print(f"wrote {out} with {n} pair records")

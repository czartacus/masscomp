"""Merge per-job result files into one and compute the study summary.
Usage: python studies/merge.py <recovery|confound|lbf> <out.json> <in.json ...>"""
import sys, json
import numpy as np

study, out, ins = sys.argv[1], sys.argv[2], sys.argv[3:]
if study == "diag":
    # diagnostic cells (studies/diag.py): no study-level summary, just the cells in spec order with a one-line digest each
    cells = [json.load(open(p)) for p in ins]
    cells = [c for c in cells if c.get("study") == "diag"]
    cells.sort(key=lambda c: c.get("spec", ""))
    digest = []
    for c in cells:
        if c["kind"] == "basin":
            for m, r in c["result"].items():
                digest.append(dict(spec=c["spec"], mode=m, root_cov_R=r["root_cov_R"], root_cov_fam=r["root_cov_fam"], mean_ll=r["mean_ll"],
                                   omega_ratio=r["omega_ratio"], t_root=r["t_root"], log_prior=r["log_prior"], secs=r["secs"]))
        elif c["kind"] == "essrate":
            r = c["result"]
            digest.append(dict(spec=c["spec"], ms_per_sweep=1000 * r["secs_per_sweep"], root_change_rate=r["root_change_rate"],
                               **{"ess_per_sec_" + k: v for k, v in r["ess_per_sec"].items()}))
        else:
            digest.append(dict(spec=c["spec"], max_abs_z=c["result"].get("max_abs_z"), moves=c["result"].get("moves"), secs=c["secs"]))
    json.dump({"study": "diag", "digest": digest, "cells": cells}, open(out, "w"), indent=1)
    print("DIGEST", json.dumps(digest, indent=1))
    raise SystemExit(0)
FIELD = {"recovery": "root_set_cov", "confound": "frac_both", "lbf": "lbf", "pt": "lbf", "lbfj": "lbf"}[study]
rows, partial, meta, pt_cells = [], {}, {}, {}
for p in ins:
    d = json.load(open(p))
    if study == "pt":
        if d.get("study") != "pt":
            continue
        # One file holds one or both marginals of (seed, kind, depth, rep); split h1/h0 jobs are joined here.
        for r, v in d["reps"].items():
            cell = pt_cells.setdefault((d["seed"], d["kind"], d["depth"], int(r)), {})
            for h in ("h1", "h0"):
                if h in v:
                    cell[h] = v[h]
                    partial[f"{d['seed']}:{d['kind']}:{d['depth']}:{h}:r{r}"] = {"total": v[h]["logp"], "terms": v[h]["terms"],
                                                                                "betas": v[h].get("betas")}
        d["rows"] = []
    rows.extend(r for r in d.get("rows", []) if FIELD in r)   # ignore files from other studies
    if isinstance(d.get("partial"), dict):
        partial.update(d["partial"])
    meta.update({k: v for k, v in d.items() if k not in ("rows", "partial", "summary")})
if study == "pt":
    for (seed, kind, depth, rep), v in sorted(pt_cells.items()):
        if "h1" in v and "h0" in v:
            rows.append(dict(seed=seed, kind=kind, depth=depth, rep=rep, lbf=v["h1"]["logp"] - v["h0"]["logp"],
                             logp_h1=v["h1"]["logp"], logp_h0=v["h0"]["logp"], swap_acc_min=min(v["h1"]["swap_acc"]),
                             swap_acc_mean=float(np.mean(v["h1"]["swap_acc"])), secs=v["h1"]["secs"] + v["h0"]["secs"]))
seen = set(); uniq = []
for r in rows:
    key = (r.get("seed"), r.get("kind"), r.get("depth"), r.get("rep", 0))
    if key not in seen:
        seen.add(key); uniq.append(r)
rows = uniq

if study == "recovery":
    summary = {"n": len(rows), "coverage": {}, "root_set_cov_mean": float(np.mean([r["root_set_cov"] for r in rows])),
               "root_set_cov_converged": float(np.mean([r["root_set_cov"] for r in rows if max(r["rhat"].values()) <= 1.25]) if any(max(r["rhat"].values()) <= 1.25 for r in rows) else float("nan")),
               "frac_converged": float(np.mean([max(r["rhat"].values()) <= 1.25 for r in rows]))}
    for name in rows[0]["cover"]:
        vals = [r["cover"][name] for r in rows if r["cover"][name] is not None]
        summary["coverage"][name] = float(np.mean(vals)) if vals else None
    det = [r["cover"].get("u_detectable") for r in rows if r.get("cover", {}).get("u_detectable") is not None]
    summary["u_detectable_coverage"] = float(np.mean(det)) if det else None
    summary["n_u_detectable"] = int(sum(r.get("n_u_detectable", 0) for r in rows))
    summary["omega_ratio_median"] = float(np.median([r["post_mean"]["omega"] / r["truth"]["omega"] for r in rows]))
    for key in ("root_hpd_rb", "root_hpd_rb_R", "root_hpd_rb_fam"):          # Rao-Blackwellised HPD membership of the true root forms
        vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        summary[key + "_mean"] = float(np.mean(vals)) if vals else None
    summary["root_set_cov_R_mean"] = float(np.mean([r["root_set_cov_R"] for r in rows]))
    ma = [v for r in rows for v in r.get("root_rb_mass_above", {}).values()]
    summary["root_rb_mass_above_quantiles"] = [float(np.quantile(ma, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if ma else None   # ~[0.1, 0.25, 0.5, 0.75, 0.9] if calibrated
    summary["pass"] = all(0.85 <= v <= 0.95 for k, v in summary["coverage"].items() if k not in ("u_big_entries", "u_detectable") and v is not None) and summary["root_set_cov_mean"] >= 0.8
elif study == "confound":
    summary = dict(n=len(rows), frac_reps_with_warning=float(np.mean([(r["frac_both"] >= 0.5) or (r["frac_flagged"] >= 0.5) for r in rows])),
                   p_mass_mean=float(np.mean([r["p_mass_mean"] for r in rows if "p_mass_mean" in r])),
                   f_mass_mean=float(np.mean([r["f_mass_mean"] for r in rows if "f_mass_mean" in r])),
                   t_root_mean=float(np.mean([r["t_root_mean"] for r in rows])))
    summary["pass"] = summary["frac_reps_with_warning"] >= 0.8
elif study in ("lbf", "pt", "lbfj"):
    summary = {}
    # Pool replicate chains stone by stone: each chain's stone term is log mean exp(d*ll) over its samples, so with
    # equal sample counts the pooled term is logsumexp over chains minus log R. Pooled LBF rows are added as kind
    # "<kind>" with rep="pooled" and used in the summary when present.
    def logsumexp(a, axis=0):
        a = np.asarray(a); mx = a.max(axis=axis, keepdims=True)
        return (mx + np.log(np.exp(a - mx).sum(axis=axis, keepdims=True))).squeeze(axis)
    terms = {}   # (seed, kind, depth, hyp) -> list of per-chain term arrays
    for key, v in partial.items():
        if not isinstance(v, dict) or "terms" not in v:
            continue
        parts = key.split(":")
        if len(parts) >= 5:
            seed, kind, depth, hyp = int(parts[0]), parts[1], float(parts[2]), parts[3]
        else:                                    # legacy key without depth
            seed, kind, hyp = int(parts[0]), parts[1], parts[2]
            depth = next((r["depth"] for r in rows if r["seed"] == seed and r["kind"] == kind), None)
        terms.setdefault((seed, kind, depth, hyp), []).append((np.array(v["terms"]), v.get("betas"), v["total"]))
    cells = sorted({(r["seed"], r["kind"], r["depth"]) for r in rows})
    pooled = []
    def pool(chains):
        """Stone-by-stone pooling is only valid when every chain used the same ladder (fixed betas);
        with adaptive ladders the stone terms are not comparable, so fall back to the mean of the totals."""
        ladders = [tuple(np.round(b, 6)) if b is not None else None for _, b, _ in chains]
        if all(l is not None and l == ladders[0] for l in ladders):
            return float(sum(logsumexp(np.stack([t for t, _, _ in chains]), axis=0) - np.log(len(chains)))), "stones"
        return float(np.mean([tot for _, _, tot in chains])), "mean"
    for seed, kind, depth in cells:
        c1, c0 = terms.get((seed, kind, depth, "h1"), []), terms.get((seed, kind, depth, "h0"), [])
        if len(c1) >= 2 and len(c0) >= 2:
            (l1, m1), (l0, m0) = pool(c1), pool(c0)
            pooled.append(dict(depth=depth, seed=seed, kind=kind, rep="pooled", lbf=l1 - l0, logp_h1=l1, logp_h0=l0,
                               n_chains=min(len(c1), len(c0)), pooling=m1 if m1 == m0 else "mixed"))
    if pooled:
        per_rep = [r for r in rows if r.get("rep") != "pooled"]
        summary["replicate_noise_sd"] = float(np.mean([np.std([r["lbf"] for r in per_rep if (r["seed"], r["kind"], r["depth"]) == k])
                                                       for k in cells if sum((r["seed"], r["kind"], r["depth"]) == k for r in per_rep) >= 2] or [np.nan]))
        # noise of a single log-marginal estimate, from replicate pairs of the same (cell, hypothesis)
        pairs = [np.std([pt_cells[c][h]["logp"] for c in pt_cells if c[:3] == k and h in pt_cells[c]], ddof=1)
                 for k in cells for h in ("h1", "h0") if sum(c[:3] == k and h in pt_cells[c] for c in pt_cells) >= 2]
        summary["logp_noise_sd"] = float(np.mean(pairs)) if pairs else None
        rows = [r for r in rows if r.get("rep") != "pooled"] + pooled
        summary["n_pooled_cells"] = len(pooled)
    use = [r for r in rows if r.get("rep") == "pooled"] or rows
    rows_all, rows = rows, use
    for depth in sorted({r["depth"] for r in rows if r["kind"] == "related"} or {r["depth"] for r in rows}):
        rel = np.array([r["lbf"] for r in rows if r["depth"] == depth and r["kind"] == "related"])
        un = np.array([r["lbf"] for r in rows if r["kind"] == "unrelated" and (r["depth"] == depth or r["depth"] in (0, 0.0, "-"))])
        tau = float(np.percentile(un, 99)) if len(un) else float("nan")
        summary[str(depth)] = dict(n_related=int(len(rel)), n_unrelated=int(len(un)), pooled=bool(rows and rows[0].get("rep") == "pooled"),
                                   related_mean=float(rel.mean()) if len(rel) else None, related_sd=float(rel.std()) if len(rel) else None,
                                   unrelated_mean=float(un.mean()) if len(un) else None, unrelated_sd=float(un.std()) if len(un) else None,
                                   unrelated_max=float(un.max()) if len(un) else None, tau=tau,
                                   power=float((rel > tau).mean()) if len(rel) and len(un) else None)
    rows = rows_all if "rows_all" in dir() else rows
else:
    raise SystemExit("unknown study")
json.dump({"summary": summary, "rows": rows, "partial": partial, **meta}, open(out, "w"), indent=1)
print("SUMMARY", json.dumps(summary, indent=1))

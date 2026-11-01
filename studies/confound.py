"""Plan §8.2 confounded case: both root edges carry the same change p->f. The root could be
p-initial or f-initial; the diagnostics must show it. Usage: python studies/confound.py <n> <seed0> [sweeps] [burn]"""
import sys, time
import numpy as np
from masscomp.segments import INDEX
from masscomp.tree import benchmark_a
from masscomp.simulate import sample_theta, simulate
from masscomp import events as ev
from studies.common import fit, rhat, save

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SWEEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
BURN = int(sys.argv[4]) if len(sys.argv) > 4 else 500
M = 25
NAME = "confound"
P, F = INDEX["p"], INDEX["f"]

import os, json as _json
_prev = f"results/{NAME}_{SEED0}.json"
rows = _json.load(open(_prev))["rows"] if os.path.exists(_prev) else []
_done = {r["seed"] for r in rows}
for i in range(N):
    seed = SEED0 + i
    if seed in _done:
        continue
    rng = np.random.default_rng(seed)
    forest = benchmark_a(related=True, t_root=1.5)
    th = sample_theta(forest, M, rng); th.t_root = 1.5
    th.lam = np.array([5.0, 3.0, 2.5, 6.5])          # near-identity so the confound is clean
    th.omega = 0.02
    for e in forest.children[forest.roots[0]]:
        th.u[e][(P, F)] = 5.0                          # the same change on both root edges, on top of the prior draw
    # six root forms forced p-initial so the confound has material to act on; everything else
    # (replacement, borrowing, random u_e on all edges incl. root edges) is the normal process
    R = forest.roots[0]
    rf = {}
    for c in range(6):
        rf[(R, c)] = (P,) + ev.sample_G(rng)[1:]
    ds = simulate(forest, th, M, np.random.default_rng(seed + 1), root_forms=rf)
    t0 = time.time()
    chains = fit(ds, seed, n_sweeps=SWEEPS, burn=BURN)
    both = 0; flagged = 0; n = 0; pm = []; fm = []
    for c in range(6):
        masses = {"p": 0.0, "f": 0.0, "other": 0.0}; tot = 0
        pc = []
        for ch in chains:
            v = []
            for o in ch:
                r = o["roots"][(R, c)]; tot += 1
                k = "p" if r and r[0] == P else ("f" if r and r[0] == F else "other")
                masses[k] += 1; v.append(1.0 if k == "p" else 0.0)
            pc.append(np.array(v))
        for k in masses: masses[k] /= tot
        pm.append(masses["p"]); fm.append(masses["f"])
        n += 1
        both += masses["p"] >= 0.1 and masses["f"] >= 0.1
        flagged += rhat(pc) > 1.1
    rows.append(dict(seed=seed, secs=time.time() - t0, frac_both=both / n, frac_flagged=flagged / n,
                     p_mass_mean=float(np.mean(pm)), f_mass_mean=float(np.mean(fm)),
                     t_root_mean=float(np.mean([o["t_root"] for ch in chains for o in ch])),
                     t_root_sd=float(np.std([o["t_root"] for ch in chains for o in ch]))))
    save(f"confound_{SEED0}", {"rows": rows, "partial": True})
    print(f"rep {i}: both-variants {both}/{n}, rhat-flagged {flagged}/{n}, P(root p-initial)={np.mean(pm):.2f} P(f)={np.mean(fm):.2f}, t_root {rows[-1]['t_root_mean']:.2f}±{rows[-1]['t_root_sd']:.2f} {rows[-1]['secs']:.0f}s", flush=True)

summary = dict(n=N, frac_reps_with_warning=float(np.mean([(r["frac_both"] >= 0.5) or (r["frac_flagged"] >= 0.5) for r in rows])))
summary["pass"] = summary["frac_reps_with_warning"] >= 0.8
print("SUMMARY", summary)
save(f"confound_{SEED0}", {"summary": summary, "rows": rows})

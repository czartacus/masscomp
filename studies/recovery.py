"""Plan §8.1 parameter recovery. Usage: python studies/recovery.py <n_replicates> <seed0> [sweeps] [burn] [M concepts]"""
import sys, time
import numpy as np
from masscomp.simulate import simulate_pair
from masscomp import events as ev
from masscomp.segments import EPS
from studies.common import fit, rhat, ess, covered, pooled, per_chain, root_set_coverage, recoverable_roots, root_hpd_rb, save

N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SWEEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
BURN = int(sys.argv[4]) if len(sys.argv) > 4 else 500
M = int(sys.argv[5]) if len(sys.argv) > 5 else 25
DEPTH = 1.5
NAME = "recovery"


def fired(ds, e, slot):
    """Number of inherited concepts on edge e where the rule's output is visible in the child."""
    tr, f, p, n = ds.truth, ds.forest, ds.forest.parent[e], 0
    for c in range(ds.M):
        if tr["event"].get((e, c)) != ev.INHERIT:
            continue
        yp, yc = tr["y"][(p, c)], tr["y"][(e, c)]
        if slot[0] == "ins":
            n += yc.count(slot[1]) > yp.count(slot[1])
        else:
            s, o = slot
            if s in yp and ((o == EPS and yc.count(s) < yp.count(s)) or (o != EPS and yc.count(o) > yp.count(o))):
                n += 1
    return n

import os, json as _json
_prev = f"results/{NAME}_{SEED0}.json"
rows = _json.load(open(_prev))["rows"] if os.path.exists(_prev) else []
_done = {r["seed"] for r in rows}
for r in range(N_REPS):
    seed = SEED0 + r
    if seed in _done:
        continue
    ds = simulate_pair(related=True, depth=DEPTH, M=M, seed=seed)
    th = ds.truth["theta"]
    t0 = time.time()
    chains = fit(ds, seed, n_sweeps=SWEEPS, burn=BURN, record_latents=True)
    row = dict(seed=seed, M=M, secs=time.time() - t0, cover={}, rhat={}, ess={}, truth={}, post_mean={})
    scalars = {"omega": (th.omega, "omega", None), "t_root": (th.t_root, "t_root", None)}
    for i in range(4):
        scalars[f"lam{i}"] = (th.lam[i], "lam", i)
    scalars["rho_mean"] = (th.rho.mean(), "rho", "mean")
    for name, (truth, key, idx) in scalars.items():
        if idx == "mean":
            vals = np.array([o["rho"].mean() for ch in chains for o in ch]); pc = [np.array([o["rho"].mean() for o in ch]) for ch in chains]
        else:
            vals = pooled(chains, key, idx); pc = per_chain(chains, key, idx)
        row["cover"][name] = covered(truth, vals)
        row["rhat"][name] = rhat(pc); row["ess"][name] = sum(ess(x) for x in pc)
        row["truth"][name] = float(truth); row["post_mean"][name] = float(vals.mean())
    # nonzero u_e entries: spike-and-slab, so report posterior inclusion probability and, for
    # entries with |w| > 1 (the only ones the data can see), interval coverage of the value
    # "detectable" entries: |w| > 2.5 that visibly fired on >= 3 inherited concepts (child shows the
    # rule's output where the parent did not). An entry that never fired has no observable effect,
    # whatever its w, and no method can find it; studies/u_diag.py showed the old "opportunity"
    # criterion counted such entries.
    incl, uc_big, uc_det, inc_det = [], [], [], []
    tr = ds.truth; f = ds.forest
    for e, ud in th.u.items():
        for slot, w in ud.items():
            vals = np.array([o["u"].get(e, {}).get(slot, 0.0) for ch in chains for o in ch])
            incl.append(float((vals != 0).mean()))
            nz = vals[vals != 0]
            if abs(w) > 1.0:
                uc_big.append(covered(w, nz) if len(nz) > 10 else False)
            if abs(w) > 2.5 and fired(ds, e, slot) >= 3:
                uc_det.append(covered(w, nz) if len(nz) > 10 else False); inc_det.append(float((vals != 0).mean()))
    row["u_inclusion_mean"] = float(np.mean(incl)) if incl else None
    row["cover"]["u_big_entries"] = float(np.mean(uc_big)) if uc_big else None
    row["cover"]["u_detectable"] = float(np.mean(uc_det)) if uc_det else None
    row["u_detectable_inclusion"] = float(np.mean(inc_det)) if inc_det else None
    row["n_u_entries"] = len(incl); row["n_u_big"] = len(uc_big); row["n_u_detectable"] = len(uc_det)
    rec = recoverable_roots(ds)
    R = ds.forest.roots[0]
    row["root_set_cov"] = root_set_coverage(chains, rec)                                   # joined root + family roots, recoverable only
    row["root_set_cov_R"] = root_set_coverage(chains, {k: v for k, v in rec.items() if k[0] == R})
    row["root_set_cov_fam"] = root_set_coverage(chains, {k: v for k, v in rec.items() if k[0] != R})
    row["n_recoverable"] = dict(R=sum(1 for k in rec if k[0] == R), fam=sum(1 for k in rec if k[0] != R))
    t1 = time.time()
    row["root_hpd_rb"], per = root_hpd_rb(ds, chains, rec)                                 # Rao-Blackwellised HPD membership, all recoverable roots
    hR = [v[2] for k, v in per.items() if k[0] == R]; hF = [v[2] for k, v in per.items() if k[0] != R]
    row["root_hpd_rb_R"] = float(np.mean(hR)) if hR else None
    row["root_hpd_rb_fam"] = float(np.mean(hF)) if hF else None
    row["root_rb_mass_above"] = {f"{k[0]}:{k[1]}": v[1] for k, v in per.items()}            # ~Uniform(0,1) if calibrated
    row["secs_rb"] = time.time() - t1
    for ch in chains:                                                                       # the full states are not written out
        for o in ch:
            o.pop("latents", None)
    rows.append(row)
    save(f"recovery_{SEED0}", {"rows": rows, "partial": True, "M": M})   # incremental, in case the run is cut short
    print(f"rep {r} seed {seed} {row['secs']:.0f}s cover={ {k: (round(v,2) if isinstance(v,float) else v) for k,v in row['cover'].items()} } roots={row['root_set_cov']:.2f} (R {row['root_set_cov_R']:.2f}, fam {row['root_set_cov_fam']:.2f}, rb {row['root_hpd_rb']:.2f} in {row['secs_rb']:.0f}s) rhat_max={max(row['rhat'].values()):.2f}", flush=True)

summary = {"n": N_REPS, "sweeps": SWEEPS, "burn": BURN, "coverage": {}, "root_set_cov_mean": float(np.mean([r["root_set_cov"] for r in rows]))}
for name in rows[0]["cover"]:
    vals = [r["cover"][name] for r in rows if r["cover"][name] is not None]
    summary["coverage"][name] = float(np.mean(vals)) if vals else None
summary["pass"] = all(0.85 <= v <= 0.95 for k, v in summary["coverage"].items() if k not in ("u_big_entries", "u_detectable") and v is not None) and summary["root_set_cov_mean"] >= 0.8
print("SUMMARY", summary)
save(f"recovery_{SEED0}", {"summary": summary, "rows": rows, "M": M})

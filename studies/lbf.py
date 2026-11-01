"""Plan §8.3 LBF separation. Usage: python studies/lbf.py <n_per_cell> <seed0> <depths csv> [K] [sweeps] [burn] [warmup] [power]
Stepping stone is run downward from a beta=1 fit (see marglik.stepping_stone)."""
import sys, time
import numpy as np
from masscomp.simulate import simulate_matched_pair, sample_theta, Dataset
from masscomp.tree import benchmark_a
from masscomp.marglik import stepping_stone
from studies.common import save

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DEPTHS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1.0,2.0,3.0").split(",")]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 32
SWEEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
BURN = int(sys.argv[6]) if len(sys.argv) > 6 else 500
WARMUP = int(sys.argv[7]) if len(sys.argv) > 7 else 1500
POWER = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0
M = 25


def lbf_for(ds, seed, label):
    """Same leaf data under H1 (joined forest) and H0 (two roots). Each log-marginal is cached in
    `partial` as soon as it finishes so an interrupted run resumes at the estimate, not the dataset."""
    f1 = benchmark_a(related=True, t_root=1.5); f0 = benchmark_a(related=False)
    ds1 = Dataset(f1, ds.M, ds.leaf_forms); ds0 = Dataset(f0, ds.M, ds.leaf_forms)
    rng = np.random.default_rng(seed)
    th1 = sample_theta(f1, ds.M, rng); th0 = sample_theta(f0, ds.M, rng)
    out = {}
    for hyp, dsh, th, sd in (("h1", ds1, th1, seed), ("h0", ds0, th0, seed + 7)):
        key = f"{seed}:{label}:{hyp}"
        if key not in partial:
            partial[key], _, _ = stepping_stone(dsh, th, sd, K=K, n_sweeps=SWEEPS, burn=BURN, warmup=WARMUP, power=POWER)
            _save()
        out[hyp] = partial[key]
    return out["h1"] - out["h0"], out["h1"], out["h0"]


import os, json as _json
_prev = f"results/lbf_{SEED0}.json"
_old = _json.load(open(_prev)) if os.path.exists(_prev) else {}
rows = _old.get("rows", []); partial = _old.get("partial", {})
_done = {(r["seed"], r["kind"]) for r in rows}


def _save(summary=None):
    save(f"lbf_{SEED0}", {"summary": summary, "rows": rows, "partial": partial, "K": K, "sweeps": SWEEPS, "burn": BURN,
                          "warmup": WARMUP, "power": POWER, "direction": "down"})
for depth in DEPTHS:
    for i in range(N):
        seed = SEED0 + 1000 * int(depth * 10) + i
        rel, unrel = simulate_matched_pair(depth, M, seed)
        for label, ds in (("related", rel), ("unrelated", unrel)):
            if (seed, label) in _done:
                continue
            t0 = time.time()
            lbf, l1, l0 = lbf_for(ds, seed, label)
            rows.append(dict(depth=depth, seed=seed, kind=label, lbf=lbf, logp_h1=l1, logp_h0=l0, secs=time.time() - t0))
            print(f"depth {depth} {label} seed {seed}: LBF={lbf:+.2f} (H1 {l1:.1f}, H0 {l0:.1f}) {rows[-1]['secs']:.0f}s", flush=True)
            _save()

summary = {}
for depth in DEPTHS:
    rel = np.array([r["lbf"] for r in rows if r["depth"] == depth and r["kind"] == "related"])
    un = np.array([r["lbf"] for r in rows if r["depth"] == depth and r["kind"] == "unrelated"])
    tau = float(np.percentile(un, 99)) if len(un) else float("nan")
    summary[str(depth)] = dict(related_mean=float(rel.mean()), related_sd=float(rel.std()), unrelated_mean=float(un.mean()),
                               unrelated_sd=float(un.std()), unrelated_max=float(un.max()), tau=tau,
                               power=float((rel > tau).mean()), n=int(len(rel)))
print("SUMMARY", summary)
_save(summary)

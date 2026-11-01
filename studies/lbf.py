"""Plan §8.3 LBF separation. Usage: python studies/lbf.py <n_per_cell> <seed0> <depths csv> [K] [sweeps] [burn] [warmup] [power]
Stepping stone is run downward from a beta=1 fit (see marglik.stepping_stone)."""
import sys, time
import numpy as np
from masscomp.simulate import simulate_matched_pair, sample_theta, Dataset
from masscomp.tree import benchmark_a
from masscomp.marglik import stepping_stone
from studies.common import save, N_PROCS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DEPTHS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1.0,2.0,3.0").split(",")]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 32
SWEEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
BURN = int(sys.argv[6]) if len(sys.argv) > 6 else 500
WARMUP = int(sys.argv[7]) if len(sys.argv) > 7 else 1500
POWER = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0
M = 25


def _estimate(args):
    """One log-marginal: (seed, label, hyp, dataset). Runs in a worker process."""
    seed, label, hyp, ds = args
    f = benchmark_a(related=(hyp == "h1"), t_root=1.5)
    dsh = Dataset(f, ds.M, ds.leaf_forms)
    rng = np.random.default_rng(seed)
    th = sample_theta(f, ds.M, rng)            # same draw order as before for h1; h0 uses the h0 forest's draw
    sd = seed if hyp == "h1" else seed + 7
    val, _, _ = stepping_stone(dsh, th, sd, K=K, n_sweeps=SWEEPS, burn=BURN, warmup=WARMUP, power=POWER)
    return f"{seed}:{label}:{hyp}", val


def run_cell(seed, datasets):
    """All four log-marginals of a (related, unrelated) pair, in parallel when MASSCOMP_PROCS > 1.
    Each is cached in `partial` as soon as it finishes so an interrupted run resumes at the estimate."""
    todo = [(seed, label, hyp, ds) for label, ds in datasets for hyp in ("h1", "h0") if f"{seed}:{label}:{hyp}" not in partial]
    if N_PROCS > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(min(N_PROCS, len(todo))) as ex:
            for fut in as_completed([ex.submit(_estimate, t) for t in todo]):
                key, val = fut.result(); partial[key] = val; _save()
    else:
        for t in todo:
            key, val = _estimate(t); partial[key] = val; _save()


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
        pending = [(label, ds) for label, ds in (("related", rel), ("unrelated", unrel)) if (seed, label) not in _done]
        if not pending:
            continue
        t0 = time.time()
        run_cell(seed, pending)
        for label, ds in pending:
            l1, l0 = partial[f"{seed}:{label}:h1"], partial[f"{seed}:{label}:h0"]; lbf = l1 - l0
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

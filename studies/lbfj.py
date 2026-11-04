"""Plan §8.3 LBF separation with the JOIN-PATH estimator (marglik.join_path): one stepping stone per
dataset from alpha = 0 (H0) to alpha = 1 (H1) along the root join, instead of two likelihood-path
marginals. Usage:
  python studies/lbfj.py <n_per_cell> <seed0> <depths csv> [K] [sweeps] [burn] [warmup] [power] [reps csv]
Datasets and seeds are the same as studies/lbf.py (seed = seed0 + 1000 * int(10 * depth) + i), so the
two estimators can be compared cell by cell. `reps` runs independent replicates of every estimate (the
estimator-noise check); with MASSCOMP_PROCS > 1 the replicates and the two kinds run in parallel.
Rows: depth, seed, kind, rep, lbf, secs, terms. Written to results/lbfj_<seed0>.json (study "lbfj";
merge.py treats it as an lbf study)."""
import sys, time, os, json as _json
import numpy as np
from masscomp.simulate import simulate_matched_pair, sample_theta, Dataset
from masscomp.tree import benchmark_a
from masscomp.marglik import join_path
from studies.common import save, N_PROCS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DEPTHS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.25,0.5,1.0").split(",")]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SWEEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 600
BURN = int(sys.argv[6]) if len(sys.argv) > 6 else 200
WARMUP = int(sys.argv[7]) if len(sys.argv) > 7 else 800
POWER = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0
REPS = [int(r) for r in (sys.argv[9] if len(sys.argv) > 9 else "1").split(",")]
TAG = f"lbfj_{SEED0}"
M = 25


def _estimate(args):
    seed, label, rep, ds = args
    f1 = benchmark_a(related=True, t_root=0.25)
    dsh = Dataset(f1, ds.M, ds.leaf_forms)                       # both kinds are fitted on the joined forest
    th = sample_theta(f1, ds.M, np.random.default_rng(seed + 7919 * rep))
    t0 = time.time()
    val, terms, s = join_path(dsh, th, seed * 10 + rep, K=K, n_sweeps=SWEEPS, burn=BURN, thin=4, warmup=WARMUP, power=POWER)
    return f"{seed}:{label}:r{rep}", dict(lbf=float(val), terms=[float(t) for t in terms], secs=time.time() - t0,
                                        t_root_end=float(s.st.theta.t_root))


_prev = f"results/{TAG}.json"
_old = _json.load(open(_prev)) if os.path.exists(_prev) else {}
rows = _old.get("rows", []); partial = _old.get("partial", {})


def _save():
    save(TAG, {"study": "lbfj", "rows": rows, "partial": partial, "K": K, "sweeps": SWEEPS, "burn": BURN, "warmup": WARMUP,
               "power": POWER, "estimator": "join_path"})


for depth in DEPTHS:
    for i in range(N):
        seed = SEED0 + 1000 * int(depth * 10) + i
        rel, unrel = simulate_matched_pair(depth, M, seed)
        todo = [(seed, label, rep, ds) for label, ds in (("related", rel), ("unrelated", unrel)) for rep in REPS
                if f"{seed}:{label}:r{rep}" not in partial]
        if N_PROCS > 1 and len(todo) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(min(N_PROCS, len(todo))) as ex:
                for fut in as_completed([ex.submit(_estimate, t) for t in todo]):
                    key, val = fut.result(); partial[key] = val; _save()
        else:
            for t in todo:
                key, val = _estimate(t); partial[key] = val; _save()
        for label in ("related", "unrelated"):
            for rep in REPS:
                v = partial[f"{seed}:{label}:r{rep}"]
                if not any(r["seed"] == seed and r["kind"] == label and r["rep"] == rep for r in rows):
                    rows.append(dict(depth=depth, seed=seed, kind=label, rep=rep, lbf=v["lbf"], secs=v["secs"], t_root_end=v["t_root_end"]))
                    print(f"depth {depth} {label} seed {seed} rep {rep}: LBF={v['lbf']:+.2f} ({v['secs']:.0f}s)", flush=True)
        _save()

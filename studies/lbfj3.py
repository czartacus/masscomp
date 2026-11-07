"""(lbfj3: the simulated NEWCOMER test - see the note at the end of this docstring.)
Plan §8.3 LBF separation with the JOIN-PATH estimator (marglik.join_path): one stepping stone per
dataset from alpha = 0 (H0) to alpha = 1 (H1) along the root join, instead of two likelihood-path
marginals. Usage:
  python studies/lbfj.py <n_per_cell> <seed0> <depths csv> [K] [sweeps] [burn] [warmup] [power] [reps csv]
Datasets and seeds are the same as studies/lbf.py (seed = seed0 + 1000 * int(10 * depth) + i), so the
two estimators can be compared cell by cell. `reps` runs independent replicates of every estimate (the
estimator-noise check); with MASSCOMP_PROCS > 1 the replicates and the two kinds run in parallel.
Rows: depth, seed, kind, rep, lbf, secs, terms. Written to results/lbfj_<seed0>.json (study "lbfj";
merge.py treats it as an lbf study).
Environment: MASSCOMP_SPACING = top (the plan's one-sided ladder) | bottom | both (the protocol's two-sided
ladder); MASSCOMP_ESTIMATOR = ss (stepping stone, marglik.join_path) | re (replica exchange,
marglik.join_path_re); MASSCOMP_U_RULES / MASSCOMP_SIM_U_RULES the sampler's and the generator's rule
budgets (see masscomp/simulate.py).
lbfj3: three families (masscomp.simulate.simulate_newcomer); "related" rows = the insider dataset (family 2
descends from the root of 0 and 1), "unrelated" = the outsider (family 2 independent); both fitted on the
three-family joined forest with only family 2's root edge mixed. Written to results/lbfj3_<seed0>.json."""
import sys, time, os, json as _json
import numpy as np
from masscomp.simulate import simulate_newcomer, sample_theta, Dataset
from masscomp.tree import benchmark_n
from masscomp.marglik import join_path, join_path_re
from masscomp import events as ev
from masscomp.simulate import SIM_U_RULES
from studies.common import save, N_PROCS
from masscomp.likelihood import NEWCOMER_FIRST

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DEPTHS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.25,0.5,1.0").split(",")]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SWEEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 600
BURN = int(sys.argv[6]) if len(sys.argv) > 6 else 200
WARMUP = int(sys.argv[7]) if len(sys.argv) > 7 else 800
POWER = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0
REPS = [int(r) for r in (sys.argv[9] if len(sys.argv) > 9 else "1").split(",")]
SPACING = os.environ.get("MASSCOMP_SPACING", "top")
ESTIMATOR = os.environ.get("MASSCOMP_ESTIMATOR", "ss")
TAG = f"lbfj3_{SEED0}"
MIX = [18]                                   # family 2's root: the newcomer
M = 25


def _estimate(args):
    seed, label, rep, ds = args
    f1 = benchmark_n(3, (0, 1, 2), t_root=0.25)
    if NEWCOMER_FIRST:
        f1.put_first(MIX)
    dsh = Dataset(f1, ds.M, ds.leaf_forms)                       # both kinds are fitted on the joined forest
    th = sample_theta(f1, ds.M, np.random.default_rng(seed + 7919 * rep))
    t0 = time.time()
    if ESTIMATOR == "re":
        val, terms, s = join_path_re(dsh, th, seed * 10 + rep, K=K, n_sweeps=SWEEPS, burn=BURN, thin=4, warmup=WARMUP,
                                     power=POWER, spacing=SPACING, mix_edges=MIX)
    else:
        val, terms, s = join_path(dsh, th, seed * 10 + rep, K=K, n_sweeps=SWEEPS, burn=BURN, thin=4, warmup=WARMUP, power=POWER,
                                  spacing=SPACING, mix_edges=MIX)
    return f"{seed}:{label}:r{rep}", dict(total=float(val), terms=[float(t) for t in terms], secs=time.time() - t0,
                                        t_root_end=float(s.st.theta.t_root))


_prev = f"results/{TAG}.json"
_old = _json.load(open(_prev)) if os.path.exists(_prev) else {}
rows = _old.get("rows", []); partial = _old.get("estimates", {})


def _save():
    save(TAG, {"study": "lbfj3", "rows": rows, "estimates": partial, "K": K, "sweeps": SWEEPS, "burn": BURN, "warmup": WARMUP,
               "power": POWER, "estimator": "join_path_re" if ESTIMATOR == "re" else "join_path", "spacing": SPACING,
               "u_rules": ev.U_RULES, "sim_u_rules": SIM_U_RULES or ev.U_RULES})


for depth in DEPTHS:
    for i in range(N):
        seed = SEED0 + 1000 * int(depth * 10) + i
        rel, unrel = simulate_newcomer(depth, M, seed)
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
                    rows.append(dict(depth=depth, seed=seed, kind=label, rep=rep, lbf=v["total"], secs=v["secs"], t_root_end=v["t_root_end"]))
                    print(f"depth {depth} {label} seed {seed} rep {rep}: LBF={v['total']:+.2f} ({v['secs']:.0f}s)", flush=True)
        _save()

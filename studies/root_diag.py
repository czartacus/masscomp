"""Is the low root-set coverage (0.46 joined root, 0.64 family roots at M=25, v4) a sampler problem or a
posterior-calibration problem? Fit one recovery dataset three ways and compare the 90% root-set coverage:
  copy  : the study protocol (child-copy init, theta from the prior)
  truth : latents and theta started at the simulated truth
  prior : latents and theta from a prior draw
If 'truth' covers ~0.9 and 'copy' ~0.6 the sampler is stuck near its start; if all are ~0.6 the sampler
mixes but the posterior sets are too narrow for another reason (metric, or a simulate/infer mismatch).
Usage: python studies/root_diag.py <seed> [sweeps] [burn] [modes]"""
import sys, time
import numpy as np
from masscomp.simulate import simulate_pair, sample_theta
from masscomp.mcmc import run_chain
from masscomp import events as ev
from studies.common import root_set_coverage, recoverable_roots, N_PROCS

seed = int(sys.argv[1]); SW = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; BURN = int(sys.argv[3]) if len(sys.argv) > 3 else 500
modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["copy", "truth", "prior"]
M, DEPTH = 25, 1.5   # as studies/recovery.py
ds = simulate_pair(related=True, depth=DEPTH, M=M, seed=seed)
rec = recoverable_roots(ds)
R = ds.forest.roots[0]


def one(args):
    mode, k = args
    rng = np.random.default_rng(seed * 100 + k)
    if mode == "truth":
        t = ds.truth
        y = {kk: v for kk, v in t["y"].items() if kk[0] not in set(ds.forest.leaves)}
        th0 = t["theta"].copy()
        _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=SW, burn=BURN, thin=5, init=(y, dict(t["event"]), dict(t["donor"])))
    elif mode == "prior":
        th0 = sample_theta(ds.forest, ds.M, rng)
        _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=SW, burn=BURN, thin=5, init_from_prior=True)
    else:
        th0 = sample_theta(ds.forest, ds.M, rng)
        _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=SW, burn=BURN, thin=5)
    return mode, out


if __name__ == "__main__":
    jobs = [(m, k) for m in modes for k in range(3)]
    t0 = time.time()
    if N_PROCS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(N_PROCS) as ex:
            res = list(ex.map(one, jobs))
    else:
        res = [one(j) for j in jobs]
    for m in modes:
        chains = [out for mm, out in res if mm == m]
        covR = root_set_coverage(chains, {k: v for k, v in rec.items() if k[0] == R})
        covF = root_set_coverage(chains, {k: v for k, v in rec.items() if k[0] != R})
        ll = np.mean([o["log_lik"] for ch in chains for o in ch])
        om = np.mean([o["omega"] for ch in chains for o in ch])
        # how often does the chain's root form equal the truth (per recorded sample)
        exact = np.mean([o["roots"].get(k) == v for ch in chains for o in ch for k, v in rec.items() if k[0] == R])
        print(f"{m:6s} root cov R={covR:.2f} fam={covF:.2f} exact(R)={exact:.2f} mean ll={ll:.1f} omega x{om / ds.truth['theta'].omega:.2f} ({time.time() - t0:.0f}s)", flush=True)

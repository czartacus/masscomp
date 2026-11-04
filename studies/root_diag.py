"""Is the low root-set coverage (0.46 joined root, 0.64 family roots at M=25, v4) a sampler problem or a
posterior-calibration problem? Fit one recovery dataset three ways and compare the 90% root-set coverage:
  copy  : the study protocol (child-copy init, theta from the prior)
  lattice: leaf-anchored lattice init (Sampler.INIT = "lattice"), theta from the prior
  truth : latents and theta started at the simulated truth
  prior : latents and theta from a prior draw
  ptK   : parallel-tempered chain with K rungs (study start), recording the beta = 1 replica
  ptKf  : same with the prior-side terms tempered too (full tempering, mixing only)
If 'truth' covers ~0.9 and 'copy' ~0.6 the sampler is stuck near its start; if all are ~0.6 the sampler
mixes but the posterior sets are too narrow for another reason (metric, or a simulate/infer mismatch).
Usage: python studies/root_diag.py <seed> [sweeps] [burn] [modes]        (env MASSCOMP_DEPTH, MASSCOMP_PROCS)
The same computation is available as `basin(...)` for studies/diag.py (cloud diag cells)."""
import sys, time, os
import numpy as np
from masscomp.simulate import simulate_pair, sample_theta
from masscomp.mcmc import run_chain
from masscomp import events as ev
from studies.common import root_set_coverage, recoverable_roots, N_PROCS

M_DEFAULT = 25


def _one(args):
    ds, mode, seed, k, SW, BURN, n_procs = args
    rng = np.random.default_rng(seed * 100 + k)
    if mode == "truth":
        t = ds.truth
        y = {kk: v for kk, v in t["y"].items() if kk[0] not in set(ds.forest.leaves)}
        th0 = t["theta"].copy()
        _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=SW, burn=BURN, thin=5, init=(y, dict(t["event"]), dict(t["donor"])))
    elif mode == "prior":
        th0 = sample_theta(ds.forest, ds.M, rng)
        _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=SW, burn=BURN, thin=5, init_from_prior=True)
    elif mode.startswith("pt"):
        # parallel-tempered chain, K rungs; the beta = 1 replica is the sample. Study-protocol start.
        from masscomp.marglik import stepping_stone_pt
        full = mode.endswith("f"); mode_k = mode[2:].rstrip("f")
        Kpt = int(mode_k) if mode_k else 4
        th0 = sample_theta(ds.forest, ds.M, rng)
        _, _, _, _, info = stepping_stone_pt(ds, th0, seed * 100 + 50 + k, K=Kpt, n_sweeps=SW, burn=BURN, thin=5,
                                             n_procs=n_procs, record_cold=True, init_from_prior=False, full=full)
        out = info["cold"]
    else:
        from masscomp.mcmc import Sampler
        old_init = Sampler.INIT
        Sampler.INIT = "lattice" if mode == "lattice" else "copy"
        th0 = sample_theta(ds.forest, ds.M, rng)
        try:
            _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=SW, burn=BURN, thin=5)
        finally:
            Sampler.INIT = old_init
    return mode, out


def basin(seed, depth, modes, SW=1500, BURN=500, M=M_DEFAULT, n_chains=3, n_procs=N_PROCS, verbose=True):
    """Run `n_chains` chains per mode on the recovery dataset (seed, depth, M) and summarise each mode.
    Returns {mode: summary dict} with the numbers root_diag prints (root coverage, mean ll, omega ratio, ...)."""
    ds = simulate_pair(related=True, depth=depth, M=M, seed=seed)
    rec = recoverable_roots(ds)
    R = ds.forest.roots[0]
    jobs = [(ds, m, seed, k, SW, BURN, n_procs) for m in modes for k in range(n_chains)]
    t0 = time.time()
    if any(m.startswith("pt") for m in modes):
        res = [_one(j) for j in jobs]          # the PT runs use the processes internally
    elif n_procs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_procs) as ex:
            res = list(ex.map(_one, jobs))
    else:
        res = [_one(j) for j in jobs]
    th = ds.truth["theta"]
    summary = {}
    for m in modes:
        chains = [out for mm, out in res if mm == m]
        covR = root_set_coverage(chains, {k: v for k, v in rec.items() if k[0] == R})
        covF = root_set_coverage(chains, {k: v for k, v in rec.items() if k[0] != R})
        ll = np.mean([o["log_lik"] for ch in chains for o in ch])
        om = np.mean([o["omega"] for ch in chains for o in ch])
        # how often does the chain's root form equal the truth (per recorded sample)
        exact = np.mean([o["roots"].get(k) == v for ch in chains for o in ch for k, v in rec.items() if k[0] == R])
        lam = np.mean([o["lam"] for ch in chains for o in ch], axis=0); tr = np.mean([o["t_root"] for ch in chains for o in ch])
        rho = np.mean([o["rho"].mean() for ch in chains for o in ch]); nu = np.mean([sum(len(d) for d in o["u"].values()) for ch in chains for o in ch])
        lp = np.mean([o["log_prior"] for ch in chains for o in ch])
        # per-chain means, so chain disagreement is visible
        ll_chain = [float(np.mean([o["log_lik"] for o in ch])) for ch in chains]
        om_chain = [float(np.mean([o["omega"] for o in ch])) for ch in chains]
        summary[m] = dict(mode=m, seed=seed, depth=depth, M=M, sweeps=SW, burn=BURN, n_chains=n_chains,
                          root_cov_R=float(covR), root_cov_fam=float(covF), exact_R=float(exact), mean_ll=float(ll),
                          omega=float(om), omega_ratio=float(om / th.omega), lam=[float(v) for v in lam], t_root=float(tr),
                          rho_mean=float(rho), n_u=float(nu), log_prior=float(lp), ll_chain=ll_chain, omega_chain=om_chain,
                          truth=dict(lam=[float(v) for v in th.lam], t_root=float(th.t_root), omega=float(th.omega),
                                     rho_mean=float(th.rho.mean()), n_u=int(sum(len(d) for d in th.u.values()))),
                          secs=float(time.time() - t0))
        if verbose:
            print(f"{m:6s} root cov R={covR:.2f} fam={covF:.2f} exact(R)={exact:.2f} mean ll={ll:.1f} omega x{om / th.omega:.2f} ({time.time() - t0:.0f}s)", flush=True)
            print(f"       lam={np.round(lam, 2)} (truth {np.round(th.lam, 2)}) t_root={tr:.2f} (truth {th.t_root}) rho_mean={rho:.3f} (truth {th.rho.mean():.3f}) |u|={nu:.1f} (truth {sum(len(d) for d in th.u.values())}) log_prior={lp:.1f}", flush=True)
    return summary


if __name__ == "__main__":
    seed = int(sys.argv[1]); SW = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; BURN = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["copy", "truth", "prior"]
    basin(seed, float(os.environ.get("MASSCOMP_DEPTH", "1.5")), modes, SW, BURN)

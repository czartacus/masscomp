"""Noise check for the parallel-tempered stepping stone: re-estimate one dataset's log p(X|H1) R times with
fresh seeds. Usage: python studies/pt_check.py <seed> <depth> <kind related|unrelated> <R> [K] [sweeps] [burn] [rep0] [hyps h1,h0] [M] [warmup]
hyps selects which marginals this job computes (cloud: one per job, merged by seed/kind/depth/rep).
Compare the spread with the sequential replicates (results/cloud/lbf_41355534462.json: sd ~20-30 nats)."""
import sys, time, json
import numpy as np
from masscomp.simulate import simulate_matched_pair, sample_theta, Dataset
from masscomp.tree import benchmark_a
from masscomp.marglik import stepping_stone_pt
from studies.common import N_PROCS

seed, depth, kind, R = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
K = int(sys.argv[5]) if len(sys.argv) > 5 else 24
SWEEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
BURN = int(sys.argv[7]) if len(sys.argv) > 7 else 500
REP0 = int(sys.argv[8]) if len(sys.argv) > 8 else 0   # first replicate id (cloud: one replicate per job)
HYPS = sys.argv[9].split(",") if len(sys.argv) > 9 else ["h1", "h0"]
M = int(sys.argv[10]) if len(sys.argv) > 10 else 25
WARMUP = int(sys.argv[11]) if len(sys.argv) > 11 else 0


def one(r):
    rel, unrel = simulate_matched_pair(depth, M, seed)
    ds = rel if kind == "related" else unrel
    out = {}
    for hyp in HYPS:
        f = benchmark_a(related=(hyp == "h1"), t_root=1.5)
        dsh = Dataset(f, ds.M, ds.leaf_forms)
        th = sample_theta(f, ds.M, np.random.default_rng(seed))
        t0 = time.time()
        val, terms, swaps, betas = stepping_stone_pt(dsh, th, seed + 100000 * (r + 1) + (0 if hyp == "h1" else 7),
                                                     K=K, n_sweeps=SWEEPS, burn=BURN, warmup=WARMUP)
        out[hyp] = dict(logp=val, terms=[float(t) for t in terms], swap_acc=[float(s) for s in swaps],
                        betas=[float(b) for b in betas], secs=time.time() - t0)
        print(f"rep {r} {hyp}: logp={val:.1f} swap_acc min/mean={min(swaps):.2f}/{np.mean(swaps):.2f} {out[hyp]['secs']:.0f}s", flush=True)
    if "h1" in out and "h0" in out:
        out["lbf"] = out["h1"]["logp"] - out["h0"]["logp"]
        print(f"rep {r}: LBF={out['lbf']:+.2f}", flush=True)
    return r, out


if __name__ == "__main__":
    tag = f"results/pt_{seed}_{kind}" + (f"_r{REP0}" if REP0 else "") + ("" if len(HYPS) == 2 else "_" + HYPS[0]) + ".json"
    res = {}
    if N_PROCS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(N_PROCS, R)) as ex:
            for r, out in ex.map(one, range(REP0, REP0 + R)):
                res[r] = out; json.dump(dict(study="pt", seed=seed, depth=depth, kind=kind, K=K, sweeps=SWEEPS, burn=BURN, M=M, warmup=WARMUP, reps=res), open(tag, "w"), indent=1)
    else:
        for r in range(REP0, REP0 + R):
            _, out = one(r); res[r] = out; json.dump(dict(study="pt", seed=seed, depth=depth, kind=kind, K=K, sweeps=SWEEPS, burn=BURN, M=M, warmup=WARMUP, reps=res), open(tag, "w"), indent=1)
    lbfs = [v["lbf"] for v in res.values() if "lbf" in v]
    if lbfs:
        print("LBFs", [round(x, 1) for x in lbfs], "sd", round(float(np.std(lbfs)), 1))

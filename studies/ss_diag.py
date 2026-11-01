"""Stepping-stone diagnostic: per-stone terms, within-stone ll spread and the contribution of the
single largest sample, for one dataset/hypothesis. Usage:
  python studies/ss_diag.py <seed> <related|unrelated> <h1|h0> <K> <sweeps> <burn> <ss_seed>"""
import sys, math, time
import numpy as np
from masscomp.simulate import simulate_matched_pair, sample_theta, Dataset
from masscomp.tree import benchmark_a
from masscomp.mcmc import Sampler

seed = int(sys.argv[1]); kind = sys.argv[2]; hyp = sys.argv[3]
K = int(sys.argv[4]); SW = int(sys.argv[5]); BURN = int(sys.argv[6]); SS = int(sys.argv[7])
rel, unrel = simulate_matched_pair(1.0, 25, seed)
ds = rel if kind == "related" else unrel
f = benchmark_a(related=(hyp == "h1"), t_root=1.5)
ds = Dataset(f, ds.M, ds.leaf_forms)
rng = np.random.default_rng(SS)
th = sample_theta(f, ds.M, rng)
betas = [(k / K) ** 3 for k in range(K + 1)]
s = Sampler(ds, th.copy(), rng, beta=betas[0], init_from_prior=True)
tot = 0.0; t0 = time.time()
for k in range(K):
    s.beta = betas[k]; d = betas[k + 1] - betas[k]
    lls = []
    for i in range(SW):
        s.sweep()
        if i >= BURN and (i - BURN) % 5 == 0:
            lls.append(s.st.log_lik())
    lls = np.array(lls); m = lls.max()
    w = np.exp(d * (lls - m)); term = d * m + math.log(w.mean())
    tot += term
    print(f"stone {k:2d} beta={betas[k]:.4f} d={d:.4f} ll mean={lls.mean():.1f} sd={lls.std():.1f} "
          f"max-mean={m - lls.mean():.1f} term={term:.2f} top-sample share={w.max() / w.sum():.2f} "
          f"ess={w.sum() ** 2 / (w ** 2).sum():.1f} gibbs-acc={s.acc['gibbs'][0] / max(1, s.acc['gibbs'][1]):.2f} "
          f"{time.time() - t0:.0f}s", flush=True)
print(f"TOTAL {kind} {hyp} K={K} ss_seed={SS}: {tot:.2f}")

"""Seed-agreement check for the downward stepping stone on one dataset.
Usage: python studies/ss_check.py <seed> <related|unrelated> <h1|h0> <K> <sweeps> <burn> <warmup> <ss_seed>"""
import sys, time
import numpy as np
from masscomp.simulate import simulate_matched_pair, sample_theta, Dataset
from masscomp.tree import benchmark_a
from masscomp.marglik import stepping_stone
seed, kind, hyp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
K, SW, BURN, WU, SS = map(int, sys.argv[4:9])
rel, unrel = simulate_matched_pair(1.0, 25, seed)
ds = rel if kind == "related" else unrel
f = benchmark_a(related=(hyp == "h1"), t_root=1.5)
ds = Dataset(f, ds.M, ds.leaf_forms)
th = sample_theta(f, ds.M, np.random.default_rng(SS))
t0 = time.time()
tot, terms, _ = stepping_stone(ds, th, SS, K=K, n_sweeps=SW, burn=BURN, warmup=WU, verbose=True)
print(f"TOTAL {kind} {hyp} K={K} sw={SW} ss_seed={SS}: {tot:.2f}  {time.time()-t0:.0f}s")

"""Plan §9: seconds per sweep at benchmark A and a 20-language / 50-concept configuration."""
import time, json, sys, cProfile, pstats, io
import numpy as np
from masscomp.simulate import simulate_pair, sample_theta, simulate
from masscomp.tree import benchmark_a, Forest
from masscomp.mcmc import Sampler


def big_forest():
    """Benchmark B topology: each family tree duplicated (20 leaves)."""
    a = benchmark_a(related=True)
    parent, length, family, leaves, names = list(a.parent), list(a.length), list(a.family), list(a.leaves), dict(a.names)
    R = a.roots[0]
    a = benchmark_a(related=False)
    for fam in range(2):
        off = len(parent)
        for v in range(9):
            p = a.parent[v]
            parent.append(off + p if p >= 0 else R); length.append(a.length[v] if p >= 0 else 1.5); family.append(2 + fam)
        for lv in [3, 4, 6, 7, 8]:
            leaves.append(off + lv); names[off + lv] = f"F{2+fam}L{lv}"
    return Forest(parent, length, family, leaves, names=names)


def time_config(name, ds, n=5, profile=False):
    rng = np.random.default_rng(0)
    s = Sampler(ds, sample_theta(ds.forest, ds.M, rng), rng)
    s.sweep()
    pr = cProfile.Profile() if profile else None
    t = time.time()
    if pr: pr.enable()
    for _ in range(n):
        s.sweep()
    if pr: pr.disable()
    dt = (time.time() - t) / n
    print(f"{name}: {dt:.2f} s/sweep; 3 chains x 5000 sweeps = {3*5000*dt/3600:.1f} h; acc={ {k: round(a/b,3) if b else None for k,(a,b) in s.acc.items()} }")
    if pr:
        st = pstats.Stats(pr); st.sort_stats("cumulative"); st.print_stats(8)
    return dt


if __name__ == "__main__":
    out = {}
    out["A_10x25"] = time_config("A (10 lang x 25 concepts)", simulate_pair(True, 1.5, 25, seed=1), profile="--profile" in sys.argv)
    f = big_forest(); rng = np.random.default_rng(2)
    from masscomp.simulate import simulate, sample_theta
    th = sample_theta(f, 50, rng); th.t_root = 1.5
    out["B_20x50"] = time_config("B (20 lang x 50 concepts)", simulate(f, th, 50, rng), n=3)
    json.dump(out, open("results/profile.json", "w"), indent=1)

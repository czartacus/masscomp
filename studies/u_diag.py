"""Why are detectable u_e entries not recovered in §8.1?  Fit one recovery dataset three ways:
  A  latents clamped at truth (forms + events): tests the u move and likelihood alone
  B  started at truth, nothing clamped:        tests whether the sampler stays near the rule
  C  started from the prior (the study setting)
and report the posterior inclusion of each detectable entry.
Usage: python studies/u_diag.py <seed> [M] [sweeps] [burn] [modes]   modes e.g. ABC
"""
import sys, time
import numpy as np
from masscomp.simulate import simulate_pair
from masscomp.mcmc import run_chain
from masscomp import events as ev

seed = int(sys.argv[1]); M = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SW = int(sys.argv[3]) if len(sys.argv) > 3 else 800; BURN = int(sys.argv[4]) if len(sys.argv) > 4 else 300
MODES = sys.argv[5] if len(sys.argv) > 5 else "ABC"
ds = simulate_pair(related=True, depth=1.5, M=M, seed=seed)
th = ds.truth["theta"]; tr = ds.truth; f = ds.forest
det = []
for e, ud in th.u.items():
    for slot, w in ud.items():
        seg = slot[1] if slot[0] == "ins" else slot[0]
        opp = sum(1 for c in range(M) if tr["event"].get((e, c)) == ev.INHERIT and (slot[0] == "ins" or seg in tr["y"][(f.parent[e], c)]))
        if abs(w) > 2.5 and opp >= 3:
            det.append((e, slot, w, opp))
print("detectable entries:", det, flush=True)
if not det:
    sys.exit(0)
leafset = set(f.leaves)
y_true = {k: v for k, v in tr["y"].items() if k[0] not in leafset}
init = (y_true, dict(tr["event"]), dict(tr["donor"]))
th0 = th.copy(); th0.u = {}          # parameters start at truth except u (the thing under test)

for mode in MODES:
    t0 = time.time()
    if mode == "A":
        _, samples = run_chain(ds, th0, seed, SW, BURN, thin=5, init=init, clamp_latents=True)
    elif mode == "B":
        _, samples = run_chain(ds, th0, seed, SW, BURN, thin=5, init=init)
    else:
        from masscomp.simulate import sample_theta
        rng = np.random.default_rng(seed * 100)
        _, samples = run_chain(ds, sample_theta(ds.forest, ds.M, rng), seed * 100 + 50, SW, BURN, thin=5)
    om = np.mean([s["omega"] for s in samples])
    print(f"mode {mode}  {time.time()-t0:.0f}s  omega post {om:.3f} (truth {th.omega:.3f})", flush=True)
    for e, slot, w, opp in det:
        vals = np.array([s["u"].get(e, {}).get(slot, 0.0) for s in samples])
        nz = vals[vals != 0]
        print(f"   edge {e} slot {slot} w={w:.2f} opp={opp}: inclusion {np.mean(vals != 0):.2f}  mean|nz {nz.mean() if len(nz) else float('nan'):.2f}", flush=True)
    # what does the chain put on that edge instead?
    for e, slot, w, opp in det:
        other = {}
        for s in samples:
            for sl, v in s["u"].get(e, {}).items():
                other[sl] = other.get(sl, 0) + 1
        top = sorted(other.items(), key=lambda t: -t[1])[:5]
        print(f"   edge {e} most-included slots: {[(sl, round(n/len(samples),2)) for sl, n in top]}", flush=True)

"""Planted branch-rule check. Simulate benchmark A (related, M concepts) with one strong rule
u_e[(p, f)] = +6 on the first non-root edge and otherwise prior-drawn theta; fit; report posterior
inclusion of that slot, the omega ratio, and the fraction of concepts on that edge labelled borrow.
Usage: python studies/u_planted.py [seed] [M] [sweeps]"""
import sys, numpy as np
from masscomp.simulate import simulate, sample_theta
from masscomp.tree import benchmark_a
from masscomp import events as ev
from masscomp.mcmc import run_chain

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
M = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SW = int(sys.argv[3]) if len(sys.argv) > 3 else 400
rng = np.random.default_rng(seed)
f = benchmark_a(related=True, t_root=1.5)
th = sample_theta(f, M, rng); th.t_root = 1.5
e = [x for x in f.edges if f.parent[x] not in f.roots][0] if any(f.parent[x] not in f.roots for x in f.edges) else f.edges[0]
P, F = 0, 1                       # segment ids: rule 0 -> 1 on edge e
th.u = {k: {} for k in f.edges}; th.u[e] = {(P, F): 6.0}
# make the rule matter: many roots start with p
th.omega = 0.05
roots = {(r, c): (P,) + ev.sample_G(rng)[1:] for r in f.roots for c in range(M)}   # every root form starts with segment 0
ds = simulate(f, th, M, rng, root_forms=roots)
# count opportunities
opp = sum(1 for c in range(M) if ds.truth["event"].get((e, c)) == ev.INHERIT and P in ds.truth["y"][(f.parent[e], c)])
print(f"edge {e}, opportunities (inherited, parent has segment 0): {opp}, true omega {th.omega:.3f}")
th0 = sample_theta(f, M, np.random.default_rng(seed + 1))
if len(sys.argv) > 4 and sys.argv[4] == "nou":
    th0.u = {k: {} for k in f.edges}        # start without rules (the study chains start from the prior)
if len(sys.argv) > 4 and sys.argv[4] == "clamp":        # latents at truth: tests the parameter moves alone
    leafset = set(f.leaves); tr = ds.truth
    init = ({k: v for k, v in tr["y"].items() if k[0] not in leafset}, dict(tr["event"]), dict(tr["donor"]))
    _, samples = run_chain(ds, th0, seed + 2, SW, SW // 3, thin=5, init=init, clamp_latents=True)
elif len(sys.argv) > 4 and sys.argv[4] == "truth":      # start at truth, nothing clamped
    leafset = set(f.leaves); tr = ds.truth
    init = ({k: v for k, v in tr["y"].items() if k[0] not in leafset}, dict(tr["event"]), dict(tr["donor"]))
    _, samples = run_chain(ds, th0, seed + 2, SW, SW // 3, thin=5, init=init)
elif len(sys.argv) > 4 and sys.argv[4] == "anneal":     # prior start with a tempering ramp over the burn-in
    _, samples = run_chain(ds, th0, seed + 2, SW, SW // 3, thin=5, init_from_prior=True, anneal=int(SW * 0.6))
elif len(sys.argv) > 4 and sys.argv[4] == "study":      # the study protocol: child-copy init, no anneal
    _, samples = run_chain(ds, th0, seed + 2, SW, SW // 3, thin=5)
else:
    _, samples = run_chain(ds, th0, seed + 2, SW, SW // 3, thin=5, init_from_prior=True)
incl = np.mean([1.0 if (P, F) in s["u"].get(e, {}) else 0.0 for s in samples])
w = [s["u"][e][(P, F)] for s in samples if (P, F) in s["u"].get(e, {})]
om = np.mean([s["omega"] for s in samples])
print(f"inclusion {incl:.2f}  w mean {np.mean(w) if w else float('nan'):.2f}  omega post/true {om / th.omega:.2f}")

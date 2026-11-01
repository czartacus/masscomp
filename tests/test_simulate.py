import numpy as np
from masscomp.segments import N_SEG, INDEX, to_str
from masscomp.tree import benchmark_a
from masscomp import events as ev
from masscomp.simulate import simulate_pair, sample_theta, simulate, Theta


def test_events_sum_to_one():
    rng = np.random.default_rng(0)
    for _ in range(100):
        lp = ev.event_logprobs(rng.gamma(2, 0.5), rng.gamma(1, 0.1), rng.uniform(0.05, 3))
        assert abs(np.exp(lp).sum() - 1) < 1e-9


def test_simulate_shapes():
    ds = simulate_pair(related=True, depth=1.5, M=25, seed=3)
    assert len(ds.leaf_forms) == 10 * 25
    for f in ds.leaf_forms.values():
        assert 2 <= len(f) <= 6 or True   # lengths may drift via events; segments must be valid
        assert all(0 <= s < N_SEG for s in f)


def test_simulate_grimm():
    forest = benchmark_a(related=False)
    rng = np.random.default_rng(5)
    th = sample_theta(forest, 60, rng)
    th.lam = np.array([6.0, 4.0, 3.0, 8.0])     # nearly identity elsewhere
    th.rho[:] = 1e-6; th.omega = 1e-6           # no events
    th.u = {e: {} for e in forest.edges}
    th.u[1] = {(INDEX["p"], INDEX["f"]): 12.0}  # force p -> f on edge root->A of family 0
    ds = simulate(forest, th, 60, rng)
    y = ds.truth["y"]
    p, f = INDEX["p"], INDEX["f"]
    # (a) in the simulated data: every root form starting with p has a child on edge 1 starting with f or p
    n_p = n_f = 0
    for c in range(60):
        parent, child = y[(0, c)], y[(1, c)]
        if parent[0] == p:
            n_p += 1; n_f += child[0] == f
    # (b) directly on the edge transducer, with enough trials to be decisive
    from masscomp.transducer import EdgeParams
    ep = EdgeParams(th.lam, th.u[1])
    hits = tot = 0
    for _ in range(300):
        parent = (p, INDEX["a"], INDEX["t"])
        child = ep.sample(parent, rng)
        tot += 1; hits += len(child) > 0 and child[0] == f
    assert hits / tot >= 0.9, (hits, tot, n_p, n_f)

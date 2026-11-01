import itertools, math, numpy as np
from masscomp.segments import N_SEG
from masscomp.tree import Forest
from masscomp.simulate import Theta, Dataset
from masscomp.transducer import EdgeParams
from masscomp import events as ev
from masscomp.marglik import stepping_stone


def test_stepping_stone_toy():
    """One root, two leaves, one concept, parameters clamped, no insertion/deletion:
    the marginal likelihood is a finite sum over root forms and event types."""
    forest = Forest([-1, 0, 0], [0.0, 0.5, 0.5], [0, 0, 0], [1, 2], names={1: "a", 2: "b"})
    lam = np.array([3.0, 3.0, 40.0, 40.0])           # no deletion, no insertion
    theta = Theta(lam, {1: {}, 2: {}}, np.array([0.8]), 0.2)
    x1, x2 = (0, 24), (6, 24)                          # "pa", "fa"
    ds = Dataset(forest, 1, {(1, 0): x1, (2, 0): x2})
    # --- exact enumeration
    ep = EdgeParams(lam, {}); adapt = ep
    evp1 = ev.event_logprobs(0.8, 0.0, 0.5)             # leaf 1 has no donor candidate
    evp2 = ev.event_logprobs(0.8, 0.2, 0.5)             # leaf 2 may borrow from leaf 1
    roots = list(itertools.product(range(N_SEG), repeat=2))
    logG2 = ev.log_G(x1)
    def leaf_term(x, other, root, k):
        if k == ev.INHERIT: return ep.logprob(root, x)
        if k == ev.REPLACE: return ev.log_G(x)
        return adapt.logprob(other, x)                   # one donor candidate
    total = -1e30
    for k1 in range(2):
        for k2 in range(3):
            if k1 == ev.INHERIT or k2 == ev.INHERIT:
                inner = np.logaddexp.reduce([logG2 + leaf_term(x1, x2, r, k1) + leaf_term(x2, x1, r, k2) for r in roots])
            else:
                inner = leaf_term(x1, x2, None, k1) + leaf_term(x2, x1, None, k2)   # root integrates out
            total = np.logaddexp(total, evp1[k1] + evp2[k2] + inner)
    exact = float(total)
    est, terms, _ = stepping_stone(ds, theta, seed=0, K=16, n_sweeps=600, burn=100, thin=2, clamp_params=True)
    assert abs(est - exact) < 0.6, (est, exact)   # MC sd ~0.25 at these settings

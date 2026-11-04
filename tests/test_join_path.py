"""The join-path LBF estimator (marglik.join_path) against exact enumeration on a two-family toy:
R -> A -> a, R -> B -> b, one concept, length-2 forms (no insertion / deletion, so every form is one
of N_SEG^2 strings and the sums are matrix products), parameters clamped."""
import itertools, math
import numpy as np
from masscomp.tree import Forest
from masscomp.simulate import Theta, Dataset
from masscomp.segments import N_SEG
from masscomp.transducer import EdgeParams
from masscomp import events as ev
from masscomp.marglik import join_path


def _tables(theta):
    forms = list(itertools.product(range(N_SEG), repeat=2))
    idx = {f: i for i, f in enumerate(forms)}
    ep = EdgeParams(theta.lam, {})
    T = np.array([[math.exp(ep.logprob(a, b)) for b in forms] for a in forms])     # T[y, y'] = P(y' | y), no indels
    G = np.array([math.exp(ev.log_G(f)) for f in forms])
    return forms, idx, T, G


def _log_z1(theta, T, G, ia, ib, t_root, t_leaf):
    """R -> A, R -> B (length t_root), A -> a, B -> b (length t_leaf). Donor candidates as the Forest
    gives them: B may borrow from A; a from B; b from A or a."""
    rho, om = theta.rho[0], theta.omega
    pA = np.exp(ev.event_logprobs(rho, 0.0, t_root)); pB = np.exp(ev.event_logprobs(rho, om, t_root))
    pa = np.exp(ev.event_logprobs(rho, om, t_leaf)); pb = np.exp(ev.event_logprobs(rho, om, t_leaf))
    n = len(G)
    Z = 0.0
    for kA in (0, 1):
        FA = T if kA == 0 else np.tile(G, (n, 1))                                     # FA[yR, yA]
        for kB in (0, 1, 2):
            for ka in (0, 1, 2):
                for kb in (0, 1, 2, 3):
                    # leaf factors as a function of (yA, yB): fa[yA, yB] * fb[yA, yB]
                    fa = T[:, ia][:, None] * np.ones((1, n)) if ka == 0 else (G[ia] * np.ones((n, n)) if ka == 1 else np.ones((n, 1)) * T[:, ia][None, :])
                    if kb == 0:
                        fb = T[:, ib][:, None] * np.ones((1, n))
                    elif kb == 1:
                        fb = G[ib] * np.ones((n, n))
                    elif kb == 2:
                        fb = 0.5 * T[:, ib][:, None] * np.ones((1, n))              # donor A
                    else:
                        fb = 0.5 * T[ia, ib] * np.ones((n, n))                        # donor a
                    Mab = fa * fb
                    w = pA[kA] * pB[kB] * pa[ka] * pb[min(kb, 2)]            # kinds 2 and 3 are the two donors
                    # a node whose form nothing downstream constrains integrates its G prior to 1 over all
                    # strings; the length-2 sum below gives G's mass on length 2 (1/5), so scale it back
                    if kA != 0 and kB != 0:
                        w *= 5.0                                                   # R unconstrained
                    if kA == 1 and ka != 0 and kb != 2 and kB != 2:
                        w *= 5.0                                                   # A ~ G, unconstrained
                    if kB == 1 and kb != 0 and ka != 2:
                        w *= 5.0                                                   # B ~ G, unconstrained
                    if kB == 2:                                                       # B borrows from A: fB[yA, yB] = T[yA, yB]
                        gA = (G[:, None] * FA).sum(axis=0)                            # sum over yR
                        Z += w * float((gA[:, None] * T * Mab).sum())
                    else:
                        FB = T if kB == 0 else np.tile(G, (n, 1))                     # FB[yR, yB]
                        S = (G[:, None] * FA).T @ FB                                  # S[yA, yB] = sum_yR G FA FB
                        Z += w * float((S * Mab).sum())
    return math.log(Z)


def _log_z0(theta, T, G, ia, ib, t_leaf):
    rho, om = theta.rho[0], theta.omega
    pa = np.exp(ev.event_logprobs(rho, om, t_leaf)); pb = np.exp(ev.event_logprobs(rho, om, t_leaf))
    n = len(G)
    Z = 0.0
    for ka in (0, 1, 2):
        for kb in (0, 1, 2, 3):
            fa = T[:, ia][:, None] * np.ones((1, n)) if ka == 0 else (G[ia] * np.ones((n, n)) if ka == 1 else np.ones((n, 1)) * T[:, ia][None, :])
            if kb == 0:
                fb = T[:, ib][:, None] * np.ones((1, n))
            elif kb == 1:
                fb = G[ib] * np.ones((n, n))
            elif kb == 2:
                fb = 0.5 * T[:, ib][:, None] * np.ones((1, n))
            else:
                fb = 0.5 * T[ia, ib] * np.ones((n, n))
            w = pa[ka] * pb[min(kb, 2)]
            if ka != 0 and kb != 2:
                w *= 5.0                                                           # A unconstrained
            if kb != 0 and ka != 2:
                w *= 5.0                                                           # B unconstrained
            Z += w * float((G[:, None] * G[None, :] * fa * fb).sum())
    return math.log(Z)


def test_join_path_toy():
    lam = np.array([3.0, 3.0, 40.0, 40.0])                     # no deletion, no insertion
    t_root, t_leaf = 0.4, 0.5
    h1 = Forest([-1, 0, 0, 1, 2], [0.0, t_root, t_root, t_leaf, t_leaf], [-1, 0, 1, 0, 1], [3, 4], names={3: "a", 4: "b"})
    h0 = Forest([-1, -1, 0, 1], [0.0, 0.0, t_leaf, t_leaf], [0, 1, 0, 1], [2, 3], names={2: "a", 3: "b"})
    assert h1.donor_candidates(2) == [1] and h1.donor_candidates(3) == [2] and h1.donor_candidates(4) == [1, 3]
    assert h0.donor_candidates(2) == [1] and h0.donor_candidates(3) == [0, 2]
    xa, xb = (0, 24), (6, 24)                                  # "pa", "fa"
    theta = Theta(lam, {e: {} for e in h1.edges}, np.array([0.8]), 0.3, t_root)
    forms, idx, T, G = _tables(theta)
    exact = _log_z1(theta, T, G, idx[xa], idx[xb], t_root, t_leaf) - _log_z0(theta, T, G, idx[xa], idx[xb], t_leaf)
    ds = Dataset(h1, 1, {(3, 0): xa, (4, 0): xb})
    ests = [join_path(ds, theta, seed=seed, K=12, n_sweeps=500, burn=100, thin=2, clamp_params=True, clamp_u=True)[0] for seed in (0, 1)]
    assert abs(np.mean(ests) - exact) < 0.5, (ests, exact)

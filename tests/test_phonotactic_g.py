"""The innovation distribution G is a length table times a segment unigram (events.set_G / fit_G).
With a non-flat G installed, the lattice FFBS (which pays the segment weight at emission and the
length weight at the terminal) must still match brute-force enumeration, and the MH / Gibbs form
moves must still target the right conditional; the flat default must reproduce the plan's G."""
import itertools, math
import numpy as np
from masscomp.transducer import EdgeParams
from masscomp.segments import N_SEG
from masscomp import events as ev
from masscomp.ffbs import draw, clear_memo
from tests.test_ffbs import _brute, _lse


def _random_G(rng, T):
    glen = np.full(T + 1, -np.inf); glen[1:] = np.log(rng.dirichlet(np.ones(T) * 2.0))
    gseg = np.log(rng.dirichlet(np.ones(N_SEG) * 0.7))
    ev.set_G(glen, gseg)


def test_flat_default():
    ev.reset_G()
    for L in range(0, 10):
        f = tuple(range(L))
        if 2 <= L <= 6:
            assert abs(ev.log_G(f) - (ev.LOG_G_LEN - L * math.log(N_SEG))) < 1e-12
        else:
            assert ev.log_G(f) < -39
    # the installed tables are normalised distributions
    assert abs(np.exp(ev.G_LEN_RAW[np.isfinite(ev.G_LEN_RAW)]).sum() - 1) < 1e-12
    assert abs(np.exp(ev.G_SEG).sum() - 1) < 1e-12


def test_set_fit_sample_G():
    rng = np.random.default_rng(3)
    try:
        _random_G(rng, 7)
        assert abs(np.exp(ev.G_LEN_RAW[1:8]).sum() - 1) < 1e-12 and abs(np.exp(ev.G_SEG).sum() - 1) < 1e-12
        f = (1, 5, 5, 2)
        assert abs(ev.log_G(f) - (ev.G_LEN[4] + ev.G_SEG[1] + 2 * ev.G_SEG[5] + ev.G_SEG[2])) < 1e-12
        assert ev.log_G(tuple(range(9))) < -39                      # beyond the support: soft floor
        # sample_G follows the installed tables
        draws = [ev.sample_G(rng) for _ in range(20000)]
        lens = np.bincount([len(d) for d in draws], minlength=8)[1:8] / len(draws)
        assert np.abs(lens - np.exp(ev.G_LEN_RAW[1:8])).max() < 0.015, lens
        segs = np.bincount([s for d in draws for s in d], minlength=N_SEG) / sum(len(d) for d in draws)
        assert np.abs(segs - np.exp(ev.G_SEG)).max() < 0.015
        # fit_G: histogram of the lengths and the segments, add-k smoothed
        forms = [(0, 1), (0, 1, 2), (3,), (0, 0, 0, 0)]
        ev.fit_G(forms, len_pseudo=0.0, seg_pseudo=0.0)
        assert abs(math.exp(ev.G_LEN_RAW[1]) - 0.25) < 1e-12 and abs(math.exp(ev.G_LEN_RAW[2]) - 0.25) < 1e-12
        assert abs(math.exp(ev.G_SEG[0]) - 6 / 10) < 1e-12 and abs(math.exp(ev.G_SEG[3]) - 1 / 10) < 1e-12
        assert ev.log_G((0, 1, 2, 3, 4)) < -39                      # no form of length 5 was seen
    finally:
        ev.reset_G()


def test_lattice_matches_brute_force_under_phonotactic_G():
    rng = np.random.default_rng(1)
    lam = np.array([4.5, 3.3, 1.7, 5.1])
    e_in = EdgeParams(lam, {}); e1 = EdgeParams(lam * 0.7, {(3, 5): 2.0}); e2 = EdgeParams(lam * 0.9, {})
    x = (3, 7, 1); z1 = (5, 7); z2 = (3, 7, 1, 4); T = 2
    try:
        _random_G(rng, T + 1)                                        # some G mass beyond the lattice's T too
        clear_memo()
        for in_mode, outs in ((0, [(e1, z1)]), (0, [(e1, z1), (e2, z2)]), (1, [(e1, z1), (e2, z2)])):
            scores = _brute(in_mode, e_in, x, outs, T)
            bf = _lse(list(scores.values()))
            forms = []
            for _ in range(3000):
                f_, logZ = draw(rng, in_mode, (e_in.log_sub_np, e_in.log_ins_np), x,
                                [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in outs], T=T)
                forms.append(f_)
            assert abs(logZ - bf) < 1e-6, (in_mode, len(outs), logZ, bf)
            top = max(scores, key=scores.get)
            p_exact = math.exp(scores[top] - bf)
            p_samp = sum(f_ == top for f_ in forms) / len(forms)
            assert abs(p_samp - p_exact) < 0.05, (in_mode, len(outs), top, p_samp, p_exact)
    finally:
        ev.reset_G(); clear_memo()


def test_gibbs_matches_enumeration_under_phonotactic_G():
    from masscomp.tree import Forest
    from masscomp.simulate import Theta, Dataset
    from masscomp.mcmc import Sampler
    rng = np.random.default_rng(2)
    forest = Forest([-1, 0, 0], [0.0, 0.5, 0.5], [0, 0, 0], [1, 2], names={1: "a", 2: "b"})
    lam = np.array([4.5, 3.0, 2.0, 6.0]); theta = Theta(lam, {1: {}, 2: {}}, np.array([0.8]), 0.0)
    x1, x2 = (0, 24, 2), (6, 24, 2)
    ds = Dataset(forest, 1, {(1, 0): x1, (2, 0): x2})
    ep = EdgeParams(lam, {})
    try:
        _random_G(rng, 6)
        clear_memo()
        forms = [f for L in (2, 3, 4) for f in itertools.product(range(N_SEG), repeat=L)]
        lw = np.array([ev.log_G(f) + ep.logprob(f, x1) + ep.logprob(f, x2) for f in forms])
        w = np.exp(lw - lw.max()); w /= w.sum()
        exact = dict(zip(forms, w))
        init = ({(0, 0): (0, 24, 2)}, {(1, 0): ev.INHERIT, (2, 0): ev.INHERIT}, {})
        s = Sampler(ds, theta, np.random.default_rng(0), init=init)
        S = s.candidates(0, 0)
        # the candidate set is built from the transducer's favourite parents, not from G, so under a
        # skewed G it covers less of the conditional (0.93 here vs 0.98 flat); the candidate move is
        # off by default and the lattice move is exact, so only the draw over S is checked tightly
        mass_in = sum(exact.get(f, 0.0) for f in S)
        assert mass_in > 0.85, mass_in
        counts = {}
        for _ in range(4000):
            s.move_forms_gibbs(); f = s.st.y[(0, 0)]; counts[f] = counts.get(f, 0) + 1
        for f, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
            p_exact = exact[f] / mass_in
            assert abs(c / 4000 - p_exact) < 0.03, (f, c / 4000, p_exact)
    finally:
        ev.reset_G(); clear_memo()

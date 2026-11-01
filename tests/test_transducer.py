import itertools, numpy as np, pytest
from masscomp.segments import N_SEG, EPS
from masscomp.transducer import EdgeParams, brute_force_logprob

LAM = (2.0, 3.0, 1.0, 2.5)


def _all_outputs(max_len):
    for L in range(max_len + 1):
        for tup in itertools.product(range(N_SEG), repeat=L):
            yield tup


def test_transducer_normalises_len1():
    """|y|=1: outputs up to length 3 (ins, sub, ins); exhaustive."""
    ep = EdgeParams(LAM, {(0, 6): 1.5, ("ins", 3): 0.7})
    for y in [(0,), (12,), (27,)]:
        total = np.logaddexp.reduce([ep.logprob(y, o) for o in _all_outputs(3)])
        assert abs(total) < 1e-6, total


def test_transducer_normalises_len2_no_insertion():
    """|y|=2 with insertions switched off (λ3 huge): outputs up to length 2; exhaustive.
    Insertion normalisation is covered by the |y|=1 test."""
    ep = EdgeParams((2.0, 3.0, 1.0, 60.0), {(4, 14): 2.0})
    for y in [(0, 24), (4, 25)]:
        total = np.logaddexp.reduce([ep.logprob(y, o) for o in _all_outputs(2)])
        assert abs(total) < 1e-6, total


def test_transducer_normalises_len2_montecarlo():
    """|y|=2 with insertions on: empirical frequency of sampled outputs matches logprob."""
    rng = np.random.default_rng(0)
    ep = EdgeParams(LAM)
    y = (0, 24)
    N = 40000
    counts = {}
    for _ in range(N):
        o = ep.sample(y, rng); counts[o] = counts.get(o, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    for o, c in top:
        p = np.exp(ep.logprob(y, o))
        se = np.sqrt(p * (1 - p) / N)
        assert abs(c / N - p) < 5 * se + 1e-3, (o, c / N, p)


def test_transducer_identity():
    ep = EdgeParams((10.0, 3.0, 1.0, 10.0))
    for y in [(0, 24), (2, 25, 16), (6, 27, 10, 24)]:
        assert np.exp(ep.logprob(y, y)) > 0.99


def test_dp_vs_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = {(int(rng.integers(N_SEG)), int(rng.integers(N_SEG + 1))): float(rng.normal())}
        ep = EdgeParams(tuple(rng.normal(size=4)), u)
        y1 = tuple(int(x) for x in rng.integers(N_SEG, size=rng.integers(1, 4)))
        y2 = tuple(int(x) for x in rng.integers(N_SEG, size=rng.integers(0, 5)))
        a, b = ep.logprob(y1, y2), brute_force_logprob(ep, y1, y2)
        if b < -1e20:
            assert a <= -40.0        # impossible output: soft floor, see transducer.FLOOR
            continue
        assert abs(a - b) < 1e-8, (y1, y2, a, b)

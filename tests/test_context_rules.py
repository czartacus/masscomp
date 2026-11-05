"""Left-context-conditioned rules u_e[(s, o, ctx)] / [("ins", o, ctx)] (transducer.py): the 3D DP, the
derived tables, sampling, and the lattice-vs-exact distinction the moves rely on."""
import itertools, math
import numpy as np
from masscomp.segments import N_SEG, EPS
from masscomp.transducer import EdgeParams, DUMMY
from masscomp import events as ev

LAM = (2.0, 3.0, 1.0, 2.5)


def _all_outputs(max_len):
    for L in range(max_len + 1):
        for tup in itertools.product(range(N_SEG), repeat=L):
            yield tup


def test_context_rule_normalises_and_conditions():
    ep0 = EdgeParams(LAM, {})
    ep = EdgeParams(LAM, {(0, 6, 24): 3.0, ("ins", 3, 0): 1.5})      # p -> f after segment 24; insert 3 after p
    assert ep.conditioned and not ep0.conditioned
    # normalisation: |y| = 1 exhaustively over outputs up to length 3; |y| = 2 with insertions off
    ep1 = EdgeParams(LAM, {(0, 6, DUMMY): 3.0, ("ins", 3, 0): 1.5})
    total = np.logaddexp.reduce([ep1.logprob((0,), o) for o in _all_outputs(3)])
    assert abs(total) < 1e-6, total
    ep2 = EdgeParams((2.0, 3.0, 1.0, 60.0), {(0, 6, 24): 3.0, (0, EPS, 12): 1.0})
    for y in [(24, 0), (12, 0)]:
        total = np.logaddexp.reduce([ep2.logprob(y, o) for o in _all_outputs(2)])
        assert abs(total) < 1e-6, (y, total)
    # the rule applies after 24 only (the insertion rule of `ep` touches row 0 of the insertion table too,
    # so the comparison uses the substitution rule alone)
    ep_sub = EdgeParams(LAM, {(0, 6, 24): 3.0})
    assert ep_sub.logprob((24, 0), (24, 6)) > ep0.logprob((24, 0), (24, 6)) + 2.0
    assert abs(ep_sub.logprob((12, 0), (12, 6)) - ep0.logprob((12, 0), (12, 6))) < 1e-9
    # the start-of-form context is DUMMY = N_SEG
    ep_s = EdgeParams(LAM, {(0, 6, DUMMY): 3.0})
    assert ep_s.logprob((0, 12), (6, 12)) > ep0.logprob((0, 12), (6, 12)) + 2.0
    assert abs(ep_s.logprob((12, 0), (12, 6)) - ep0.logprob((12, 0), (12, 6))) < 1e-9
    # the context-free density is what the lattice sees: unchanged by the conditioned rules
    assert abs(ep_sub.logprob_lattice((24, 0), (24, 6)) - ep0.logprob((24, 0), (24, 6))) < 1e-9
    assert abs(ep0.logprob_lattice((24, 0), (24, 6)) - ep0.logprob((24, 0), (24, 6))) < 1e-12


def test_derive_matches_rebuild():
    base = {(4, 14): 2.0, ("ins", 3): 0.7}
    for slot, w in [((0, 6, 24), 3.0), (("ins", 3, 0), 1.5), ((4, 14), -1.0), (("ins", 5), 0.4), ((7, EPS, 2), 2.2)]:
        for start in (base, {**base, (1, 2, 5): 1.0}):                 # from a context-free and from a conditioned table
            ep = EdgeParams(LAM, start)
            d = ep.derive(slot, w)
            full = EdgeParams(LAM, {**start, slot: start.get(slot, 0.0) + w})
            assert np.allclose(d.log_sub_np, full.log_sub_np) and np.allclose(d.log_ins_np, full.log_ins_np)
            if full.log_sub3_np is not None:
                assert d.log_sub3_np is not None and np.allclose(d.log_sub3_np, full.log_sub3_np)
            for y1, y2 in [((24, 0, 3), (24, 6, 3)), ((4, 7), (14,)), ((0, 5), (0, 3, 5))]:
                assert abs(d.logprob(y1, y2) - full.logprob(y1, y2)) < 1e-9


def test_sample_matches_logprob_under_context_rules():
    rng = np.random.default_rng(0)
    ep = EdgeParams((3.0, 3.0, 2.0, 4.0), {(0, 6, 24): 4.0, ("ins", 3, 0): 2.0})
    y1 = (24, 0)
    draws = [ep.sample(y1, rng) for _ in range(20000)]
    counts = {}
    for d in draws:
        counts[d] = counts.get(d, 0) + 1
    for form, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        p = math.exp(ep.logprob(y1, form))
        assert abs(n / len(draws) - p) < 0.012, (form, n / len(draws), p)


def test_slots_and_counts(monkeypatch):
    """With the context switch on, the slot list carries the conditioned slots and the prior budget
    is spread over them; with it off nothing changes (N_U_SLOTS as in the plan)."""
    assert ev.N_U_SLOTS == N_SEG * N_SEG + N_SEG or ev.U_CONTEXT
    if ev.U_CONTEXT:
        assert len(ev.U_SLOTS) == ev.N_U_SLOTS == (N_SEG + 2) * (N_SEG * N_SEG + N_SEG)

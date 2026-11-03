"""The lattice FFBS draw (ffbs.py) must match brute-force enumeration of P_in(y) * prod T_k(z_k|y)."""
import itertools, math
import numpy as np
from masscomp.transducer import EdgeParams
from masscomp.segments import N_SEG
from masscomp import events as ev
from masscomp.ffbs import draw


def _lse(a):
    m = max(a); return m + math.log(sum(math.exp(v - m) for v in a))


def _brute(in_mode, e_in, x, outs, T):
    scores = {}
    for L in range(1, T + 1):
        for y in itertools.product(range(N_SEG), repeat=L):
            sc = e_in.logprob(x, y) if in_mode == 1 else ev.log_G(y)
            if in_mode == 1 and sc < -39:
                continue
            ok = True
            for ep, z in outs:
                v = ep.logprob(y, z)
                if v < -39:
                    ok = False; break
                sc += v
            if ok:
                scores[y] = sc
    return scores


def test_ffbs_matches_brute_force():
    rng = np.random.default_rng(0)
    lam = np.array([4.5, 3.3, 1.7, 5.1])
    e_in = EdgeParams(lam, {}); e1 = EdgeParams(lam * 0.7, {(3, 5): 2.0}); e2 = EdgeParams(lam * 0.9, {})
    x = (3, 7, 1); z1 = (5, 7); z2 = (3, 7, 1, 4); T = 2
    for in_mode, outs in ((1, [(e1, z1), (e2, z2)]), (0, [(e1, z1)]), (1, []), (0, [(e1, z1), (e2, z2)])):
        scores = _brute(in_mode, e_in, x, outs, T)
        bf = _lse(list(scores.values()))
        forms = []
        for _ in range(3000):
            f_, logZ = draw(rng, in_mode, (e_in.log_sub_np, e_in.log_ins_np), x, [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in outs], T=T)
            forms.append(f_)
        assert abs(logZ - bf) < 1e-6, (in_mode, len(outs), logZ, bf)
        top = max(scores, key=scores.get)
        p_exact = math.exp(scores[top] - bf)
        p_samp = sum(f_ == top for f_ in forms) / len(forms)
        assert abs(p_samp - p_exact) < 0.05, (in_mode, len(outs), top, p_samp, p_exact)

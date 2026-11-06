"""Rate-time coupling and the directional cost (transducer.RATE_TIME / DIRECTIONAL): tables stay
normalised, agree with the plan's tables at t = T_REF, and change in the stated direction."""
import subprocess, sys, os, math
import numpy as np
from masscomp.transducer import EdgeParams, RATE_TIME, T_REF, DIRECTIONAL
from masscomp.segments import N_SEG, EPS, STRENGTH, FORTITION, INDEX

LAM = (4.5, 3.0, 2.0, 6.0)


def test_switch_off_ignores_t():
    if RATE_TIME:
        return
    a, b = EdgeParams(LAM, {}, t=0.05), EdgeParams(LAM, {}, t=5.0)
    assert np.allclose(a.log_sub_np, b.log_sub_np) and np.allclose(a.log_ins_np, b.log_ins_np)


def test_strength_scale():
    s = lambda x: STRENGTH[INDEX[x]]
    assert s("p") > s("b") > s("v") > s("m") > s("l") > s("j") > s("a")
    assert s("t") > s("s") > s("z") and s("k") == s("p")
    assert FORTITION[INDEX["b"], INDEX["p"]] == 1 and FORTITION[INDEX["p"], INDEX["b"]] == 0
    assert FORTITION[INDEX["a"], INDEX["t"]] == 7 and FORTITION[INDEX["s"], INDEX["a"]] == 0


def _run(env, code):
    e = dict(os.environ, **env)
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rate_time_direction_and_normalisation():
    """In a subprocess with the switch on: P(identity) falls with t, every row stays normalised, t = T_REF
    reproduces the plan's tables."""
    code = """
import numpy as np, math
from masscomp.transducer import EdgeParams, T_REF, RATE_TIME
from masscomp.segments import N_SEG, EPS
assert RATE_TIME
lam = (4.5, 3.0, 2.0, 6.0)
ref = EdgeParams(lam, {})                      # no t: the plan's table
same = EdgeParams(lam, {}, t=T_REF)
assert np.allclose(ref.log_sub_np, same.log_sub_np) and np.allclose(ref.log_ins_np, same.log_ins_np)
short, long = EdgeParams(lam, {}, t=T_REF / 4), EdgeParams(lam, {}, t=4 * T_REF)
for ep in (short, same, long):
    assert np.allclose(np.exp(ep.log_sub_np).sum(axis=1), 1.0) and np.allclose(np.exp(ep.log_ins_np).sum(axis=1), 1.0)
p_id = lambda ep: np.exp(np.diag(ep.log_sub_np[:, :N_SEG])).mean()
assert p_id(short) > p_id(same) > p_id(long), (p_id(short), p_id(same), p_id(long))
# small-t limit: P(change) proportional to t
pc_short, pc_same = 1 - p_id(short), 1 - p_id(same)
assert 0.2 < pc_short / pc_same < 0.3, pc_short / pc_same
# the sampled logprob agrees with brute force over all outputs of length <= 3 for |y| = 2 (normalisation of the DP)
import itertools
noins = EdgeParams((4.5, 3.0, 2.0, 60.0), {}, t=4 * T_REF)      # insertions off, so outputs of length <= 2 carry all the mass
tot = np.logaddexp.reduce([noins.logprob((0, 5), o) for L in range(3) for o in itertools.product(range(N_SEG), repeat=L)])
assert abs(tot) < 1e-6, tot
print("ok")
"""
    r = _run({"MASSCOMP_RATE_TIME": "1"}, code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_directional_cost():
    code = """
import numpy as np
from masscomp.transducer import EdgeParams, DIRECTIONAL
from masscomp.segments import INDEX
from masscomp import events as ev
assert DIRECTIONAL and ev.N_LAM == 5 and ev.LAM_MIN[4] == 0.0
lam = (4.5, 3.0, 2.0, 6.0, 1.5)
ep = EdgeParams(lam, {})
p, b, v, a = INDEX["p"], INDEX["b"], INDEX["v"], INDEX["a"]
assert ep.log_sub_np[b, p] < ep.log_sub_np[p, b] - 1.4          # fortition b -> p costs lam4 more than p -> b
assert abs((ep.log_sub_np[v, b] - ep.log_sub_np[b, v])) > 0.5  # and the same for v -> b against b -> v
assert np.allclose(np.exp(ep.log_sub_np).sum(axis=1), 1.0)
assert ev.log_prior_lam(np.array([4.5, 3, 2, 6, -0.1])) == float("-inf") and np.isfinite(ev.log_prior_lam(np.array([4.5, 3, 2, 6, 0.5])))
from masscomp.simulate import sample_theta
from masscomp.tree import benchmark_a
th = sample_theta(benchmark_a(related=True, t_root=0.25), 25, np.random.default_rng(3))
assert th.lam.shape == (5,) and th.lam[4] >= 0
print("ok")
"""
    r = _run({"MASSCOMP_DIRECTIONAL": "1"}, code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_directional_sampler_runs():
    """The sampler sweeps with five lambda components (the lambda move's step vector, prior, tables)."""
    code = """
import numpy as np
from masscomp.simulate import simulate, sample_theta
from masscomp.tree import benchmark_a
from masscomp.mcmc import Sampler
f = benchmark_a(related=True, t_root=0.25)
rng = np.random.default_rng(5)
th = sample_theta(f, 6, rng)
ds = simulate(f, th, 6, rng)
s = Sampler(ds, th.copy(), np.random.default_rng(1))
for _ in range(3):
    s.sweep()
assert s.st.theta.lam.shape == (5,) and s.acc["lam"][1] >= 3
print("ok")
"""
    r = _run({"MASSCOMP_DIRECTIONAL": "1"}, code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]

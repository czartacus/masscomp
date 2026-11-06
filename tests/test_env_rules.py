"""Environment-conditioned rules u_e[(s, o, env)] / [("ins", o, env)] (segments.ENV_NAMES, transducer.py):
the per-position tables and DP, the derived tables, sampling, the prior with its own budget, the lattice-
vs-exact distinction the moves rely on, and the sampler finding a planted intervocalic change."""
import itertools, math, os, subprocess, sys, json
import numpy as np
from masscomp.segments import N_SEG, EPS, VOWELS, CONSONANTS, ENV_NAMES, BOUNDARY, envs_of, INDEX
from masscomp.transducer import EdgeParams, DUMMY, brute_force_logprob
from masscomp import events as ev

LAM = (2.0, 3.0, 1.0, 2.5)
E = {n: i for i, n in enumerate(ENV_NAMES)}
p, t, k, b, d, f, s_ = (INDEX[x] for x in "ptkbdfs")
a, e, i = INDEX["a"], INDEX["e"], INDEX["i"]


def _all_outputs(max_len):
    for L in range(max_len + 1):
        for tup in itertools.product(range(N_SEG), repeat=L):
            yield tup


def test_environments():
    assert envs_of(BOUNDARY, a) == [E["#_"], E["_V"]]
    assert envs_of(a, a) == [E["V_"], E["_V"], E["V_V"]]
    assert set(envs_of(a, i)) == {E["V_"], E["_V"], E["V_V"], E["_Vfront"], E["_Vhigh"]}
    assert envs_of(INDEX["n"], BOUNDARY) == [E["_#"], E["C_"], E["N_"]]
    assert envs_of(p, t) == [E["C_"], E["_C"]]


def test_env_rule_conditions_and_normalises():
    ep0 = EdgeParams(LAM, {})
    ep = EdgeParams(LAM, {(t, d, E["V_V"]): 4.0})                    # t -> d between vowels
    assert ep.conditioned and not ep0.conditioned
    assert ep.logprob((a, t, a), (a, d, a)) > ep0.logprob((a, t, a), (a, d, a)) + 2.0
    assert abs(ep.logprob((t, a), (d, a)) - ep0.logprob((t, a), (d, a))) < 1e-9       # initial t: unchanged
    assert abs(ep.logprob((a, t), (a, d)) - ep0.logprob((a, t), (a, d))) < 1e-9       # final t: unchanged
    # normalisation: exhaustively over outputs up to length 3 for a one-segment input in each environment
    for u in ({(a, e, E["#_"]): 3.0, ("ins", k, E["_#"]): 1.5}, {(a, EPS, E["_#"]): 2.0, ("ins", s_, E["#_"]): 1.0}):
        ep1 = EdgeParams(LAM, u)
        total = np.logaddexp.reduce([ep1.logprob((a,), o) for o in _all_outputs(3)])
        assert abs(total) < 1e-6, (u, total)
    ep2 = EdgeParams((2.0, 3.0, 1.0, 60.0), {(t, d, E["V_V"]): 3.0, (t, EPS, E["_#"]): 1.0})
    for y in [(a, t), (t, a)]:
        total = np.logaddexp.reduce([ep2.logprob(y, o) for o in _all_outputs(2)])
        assert abs(total) < 1e-6, (y, total)
    # DP against brute force on forms with and without the environments
    for y1, y2 in [((a, t, a), (a, d, a)), ((a, t, a), (a, t)), ((t, a, t), (d, a, d)), ((k, a, t, e), (k, a, d, e, s_))]:
        assert abs(ep.logprob(y1, y2) - brute_force_logprob(ep, y1, y2)) < 1e-9
    # the context-free density is what the lattice sees: unchanged by the conditioned rules
    assert abs(ep.logprob_lattice((a, t, a), (a, d, a)) - ep0.logprob((a, t, a), (a, d, a))) < 1e-9
    assert abs(ep0.logprob_lattice((a, t, a), (a, d, a)) - ep0.logprob((a, t, a), (a, d, a))) < 1e-12


def test_derive_matches_rebuild():
    base = {(k, INDEX["x"]): 2.0, ("ins", s_): 0.7}
    slots = [((t, d, E["V_V"]), 3.0), (("ins", e, E["_#"]), 1.5), ((k, INDEX["x"]), -1.0), (("ins", f), 0.4), ((p, EPS, E["_C"]), 2.2)]
    for slot, w in slots:
        for start in (base, {**base, (p, b, E["V_"]): 1.0}, {**base, (t, d, E["V_V"]): 1.0}):
            ep = EdgeParams(LAM, start)
            dd = ep.derive(slot, w)
            full = EdgeParams(LAM, {**start, slot: start.get(slot, 0.0) + w})
            assert np.allclose(dd.log_sub_np, full.log_sub_np) and np.allclose(dd.log_ins_np, full.log_ins_np)
            for y1, y2 in [((a, t, a), (a, d, a)), ((k, a), (INDEX["x"],)), ((p, t), (b, t)), ((a, p, t), (a, p, e))]:
                assert abs(dd.logprob(y1, y2) - full.logprob(y1, y2)) < 1e-9, (slot, start, y1, y2)
    # a death: deriving by -w removes the rule
    ep = EdgeParams(LAM, {(t, d, E["V_V"]): 3.0})
    dead = ep.derive((t, d, E["V_V"]), -3.0)
    assert not dead.conditioned and abs(dead.logprob((a, t, a), (a, d, a)) - EdgeParams(LAM, {}).logprob((a, t, a), (a, d, a))) < 1e-9


def test_sample_matches_logprob_under_env_rules():
    rng = np.random.default_rng(0)
    ep = EdgeParams((3.0, 3.0, 2.0, 4.0), {(t, d, E["V_V"]): 4.0, ("ins", s_, E["_#"]): 2.0})
    y1 = (a, t, a)
    draws = [ep.sample(y1, rng) for _ in range(20000)]
    counts = {}
    for dr in draws:
        counts[dr] = counts.get(dr, 0) + 1
    for form, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        pr = math.exp(ep.logprob(y1, form))
        assert abs(n / len(draws) - pr) < 0.012, (form, n / len(draws), pr)


def test_prior_budgets_in_subprocess():
    """With MASSCOMP_U_ENV_RULES set, the conditioned slots have their own budget and a conditioned rule
    costs log(N_ENV) more than a context-free one; the context-free draws of sample_u are unchanged."""
    code = r'''
import json, math, numpy as np
from masscomp import events as ev
from masscomp.segments import N_SEG, N_ENV, INDEX
t = INDEX["t"]; d = INDEX["d"]
free = ev.log_prior_u({(t, d): 2.0}, 1.0) - ev.log_prior_u({}, 1.0)
cond = ev.log_prior_u({(t, d, 4): 2.0}, 1.0) - ev.log_prior_u({}, 1.0)
rng = np.random.default_rng(3); u = ev.sample_u(1.0, rng)
rng2 = np.random.default_rng(3); draws = [rng2.random() for _ in ev.U_SLOTS]
print(json.dumps(dict(n_slots=ev.N_U_SLOTS, n_env=ev.N_U_ENV_SLOTS, n_all=ev.N_U_ALL_SLOTS, free=free, cond=cond,
                      n_free=sum(1 for k in u if len(k) == 2), n_cond=sum(1 for k in u if len(k) == 3))))
'''
    env = dict(os.environ, PYTHONPATH=".", MASSCOMP_U_RULES="10", MASSCOMP_U_ENV_RULES="10")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    assert r["n_slots"] == N_SEG * N_SEG + N_SEG and r["n_env"] == len(ENV_NAMES) * r["n_slots"] and r["n_all"] == r["n_slots"] + r["n_env"]
    # same budget on both kinds: the inclusion cost differs by log N_ENV (the binomial complement terms are ~equal)
    assert abs((r["free"] - r["cond"]) - math.log(len(ENV_NAMES))) < 0.02, (r["free"], r["cond"])
    # off (this process): a conditioned rule is impossible and the slot list is the plan's
    assert ev.U_ENV_RULES == 0 and ev.N_U_ALL_SLOTS == ev.N_U_SLOTS and ev.log_prior_u({(t, d, 4): 2.0}, 1.0) == -math.inf


def test_sampler_finds_planted_intervocalic_rule():
    """A branch with t -> d between vowels in eight words and t kept elsewhere: with conditioned rules on,
    the chain should end up carrying the (t, d, V_V) rule with a clearly positive weight; run in a
    subprocess with the switch set."""
    code = r'''
import json, numpy as np
from masscomp.segments import INDEX, ENV_NAMES
from masscomp.tree import benchmark_a
from masscomp.simulate import sample_theta, simulate
from masscomp.mcmc import run_chain
E = {n: i for i, n in enumerate(ENV_NAMES)}
t, d, a, e, k, s_, p, m, n = [INDEX[x] for x in ["t", "d", "a", "e", "k", "s", "p", "m", "n"]]
f = benchmark_a(related=False)
rng = np.random.default_rng(1)
M = 24
th = sample_theta(f, M, rng)
th.lam = np.array([5.0, 3.0, 2.0, 6.0][:len(th.lam)], float); th.u = {}; th.omega = 0.01; th.rho[:] = 0.05
leaf = f.leaves[0]
th.u[leaf] = {(t, d, E["V_V"]): 7.0}
words = [(a, t, a), (e, t, e), (a, t, e, k), (p, a, t, a), (k, e, t, a), (m, a, t, e), (s_, a, t, a), (n, e, t, e),
         (t, a, k), (t, e, s_), (a, k, t, a), (e, s_, t, e), (t, a, m), (t, e, n, a), (k, a, m, a), (p, e, n, e), (s_, a, n, e), (m, e, k, a)]
roots = {(r, c): words[c % len(words)] for c in range(M) for r in f.roots}
ds = simulate(f, th, M, np.random.default_rng(2), root_forms=roots)
th0 = th.copy(); th0.u = {}
s, out = run_chain(ds, th0, 5, n_sweeps=160, burn=60, thin=2)
u = s.st.theta.u.get(leaf, {})
print(json.dumps({str(kk): v for kk, v in u.items()}))
'''
    env = dict(os.environ, PYTHONPATH=".", MASSCOMP_U_RULES="3", MASSCOMP_U_ENV_RULES="3")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    w = json.loads(res.stdout.strip().splitlines()[-1])
    key = str((t, d, E["V_V"]))
    assert key in w and w[key] > 2.0, w

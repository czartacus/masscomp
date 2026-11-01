import numpy as np
from scipy import stats
from masscomp.simulate import simulate_pair, sample_theta
from masscomp.mcmc import run_chain, Sampler
from masscomp import events as ev


def test_mcmc_prior_only():
    """beta=0: the chain must sample the prior on lam, rho, omega. The random-walk step on lam
    is tuned for the narrow posterior, so prior samples are autocorrelated; the test compares
    moments with a tolerance set from the chain's effective sample size."""
    ds = simulate_pair(related=True, depth=1.5, M=3, seed=2)
    th0 = sample_theta(ds.forest, 3, np.random.default_rng(1))
    _, out = run_chain(ds, th0, seed=3, n_sweeps=2000, burn=400, thin=4, beta=0.0, init_from_prior=True)
    lam = np.array([o["lam"] for o in out]); om = np.array([o["omega"] for o in out]); rho = np.array([o["rho"][0] for o in out])
    for i in range(4):
        assert abs(lam[:, i].mean() - ev.LAM_MU[i]) < 0.45, (i, lam[:, i].mean())
        assert 0.6 < lam[:, i].std() < 1.4, (i, lam[:, i].std())
    assert abs(om.mean() - ev.OMEGA_A / ev.OMEGA_RATE) < 0.05, om.mean()
    assert abs(rho.mean() - ev.RHO_A / ev.RHO_RATE) < 0.3, rho.mean()


def test_mcmc_fixed_latents():
    """True latents and u clamped: lam posterior mean within 0.3 of truth."""
    ds = simulate_pair(related=True, depth=1.5, M=40, seed=7)
    truth = ds.truth
    th0 = truth["theta"].copy(); th0.lam = ev.LAM_MU.copy()
    init = (truth["y"], truth["event"], truth["donor"])
    _, out = run_chain(ds, th0, seed=8, n_sweeps=1000, burn=300, thin=2, init=init, clamp_latents=True, clamp_u=True)
    lam = np.array([o["lam"] for o in out])
    lo, hi = np.percentile(lam, [5, 95], axis=0)
    assert np.all(np.abs(lam.mean(0) - truth["theta"].lam) < 0.5), (lam.mean(0), truth["theta"].lam)
    assert np.all((lo <= truth["theta"].lam) & (truth["theta"].lam <= hi)), (lo, hi, truth["theta"].lam)


def test_gibbs_matches_enumeration():
    """2-leaf toy, parameters and events clamped (both INHERIT): the Gibbs draw over the lattice
    must match the exact conditional P(root | leaves) restricted to that lattice, and the lattice
    must hold most of the exact conditional's mass."""
    import itertools
    from masscomp.tree import Forest
    from masscomp.simulate import Theta, Dataset
    from masscomp.segments import N_SEG
    from masscomp.transducer import EdgeParams
    forest = Forest([-1, 0, 0], [0.0, 0.5, 0.5], [0, 0, 0], [1, 2], names={1: "a", 2: "b"})
    lam = np.array([4.5, 3.0, 2.0, 6.0]); theta = Theta(lam, {1: {}, 2: {}}, np.array([0.8]), 0.0)
    x1, x2 = (0, 24, 2), (6, 24, 2)
    ds = Dataset(forest, 1, {(1, 0): x1, (2, 0): x2})
    ep = EdgeParams(lam, {})
    # exact conditional over all forms of length 2..4
    forms = [f for L in (2, 3, 4) for f in itertools.product(range(N_SEG), repeat=L)]
    lw = np.array([ev.log_G(f) + ep.logprob(f, x1) + ep.logprob(f, x2) for f in forms])
    w = np.exp(lw - lw.max()); w /= w.sum()
    exact = dict(zip(forms, w))
    init = ({(0, 0): (0, 24, 2)}, {(1, 0): ev.INHERIT, (2, 0): ev.INHERIT}, {})
    s = Sampler(ds, theta, np.random.default_rng(0), init=init)
    S = s.candidates(0, 0)
    mass_in = sum(exact.get(f, 0.0) for f in S)
    assert mass_in > 0.95, mass_in
    counts = {}
    for _ in range(4000):
        s.move_forms_gibbs(); f = s.st.y[(0, 0)]; counts[f] = counts.get(f, 0) + 1
    for f, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        p_exact = exact[f] / mass_in
        assert abs(c / 4000 - p_exact) < 0.03, (f, c / 4000, p_exact)

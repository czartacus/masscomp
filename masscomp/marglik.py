"""Stepping-stone marginal likelihood (plan §6).

log p(X|H) = sum_k log E_{beta_k}[ L^{beta_{k+1} - beta_k} ],  beta_k = (k/K)^3,
with the expectation over samples drawn at power posterior beta_k, chained.
`L` is the likelihood part of the target (leaf-form terms); the prior part is everything else.
"""
import math
import numpy as np
from .mcmc import Sampler


def stepping_stone(ds, theta0, seed, K=32, n_sweeps=2000, burn=500, thin=5, verbose=False,
                   direction="down", warmup=None, power=3.0, **sweep_kw):
    """direction="down": fit at beta=1 for `warmup` sweeps (default 3*n_sweeps) and then cool
    through the stones K-1..0, so every stone starts in equilibrium (Xie et al. 2011 run the same
    way). direction="up" is the original prior-to-posterior chaining. The estimator is identical;
    only the chain's initial state at each stone differs. `power` sets the stone spacing; the
    prototype study found the likelihood climbs mostly at beta 0.25-0.9 on benchmark A, so
    power=2 (denser there than the plan's cubic) is used for the grid."""
    rng = np.random.default_rng(seed)
    betas = [(k / K) ** power for k in range(K + 1)]   # beta_k = (k/K)^power
    log_terms = [None] * K
    if direction == "down":
        s = Sampler(ds, theta0.copy(), rng, beta=1.0, init_from_prior=True)
        for _ in range(warmup if warmup is not None else 3 * n_sweeps):
            s.sweep(**sweep_kw)
        order = range(K - 1, -1, -1)
    else:
        s = Sampler(ds, theta0.copy(), rng, beta=betas[0], init_from_prior=True)
        order = range(K)
    for k in order:
        s.beta = betas[k]
        d = betas[k + 1] - betas[k]
        lls = []
        for i in range(n_sweeps):
            s.sweep(**sweep_kw)
            if i >= burn and (i - burn) % thin == 0:
                lls.append(s.st.log_lik())
        lls = np.array(lls)
        m = lls.max()
        w = np.exp(d * (lls - m))
        term = d * m + math.log(w.mean())
        log_terms[k] = term
        if verbose:
            print(f"  stone {k:2d} beta={betas[k]:.4f} mean ll={lls.mean():.1f} sd={lls.std():.1f} "
                  f"term={term:.3f} ess={w.sum() ** 2 / (w ** 2).sum():.1f}", flush=True)
    return float(sum(log_terms)), log_terms, s


def log_bayes_factor(ds1, ds0, theta1, theta0, seed, **kw):
    """LBF = log p(X|H1) - log p(X|H0). ds1/ds0 are the same leaf data on joined / separate forests."""
    l1, _, _ = stepping_stone(ds1, theta1, seed, **kw)
    l0, _, _ = stepping_stone(ds0, theta0, seed + 1000, **kw)
    return l1 - l0, l1, l0

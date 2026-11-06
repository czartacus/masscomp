"""Forward simulator (plan §4)."""
import os
from dataclasses import dataclass, field
import numpy as np
from .segments import N_SEG
from .tree import benchmark_a
from .transducer import EdgeParams
from . import events as ev

# The simulator's rule budget: by default the sampler's (MASSCOMP_U_RULES), so simulated data and
# fitted prior agree; MASSCOMP_SIM_U_RULES pins the generator's budget separately, so that a study
# can fit the plan's seeded datasets (3 rules per edge) with the protocol's larger budget. The
# context-free draw consumes one uniform per slot whatever the budget, so the datasets of a seed
# are bit-identical across sampler budgets.
SIM_U_RULES = float(os.environ.get("MASSCOMP_SIM_U_RULES", "0"))


@dataclass
class Theta:
    lam: np.ndarray                 # (4,)
    u: dict                         # edge -> {slot: weight}
    rho: np.ndarray                 # (M,)
    omega: float
    t_root: float = 0.0             # only meaningful for a joined forest

    def copy(self):
        return Theta(self.lam.copy(), {e: dict(d) for e, d in self.u.items()}, self.rho.copy(), self.omega, self.t_root)


@dataclass
class Dataset:
    forest: object
    M: int
    leaf_forms: dict                # (leaf, c) -> form
    truth: dict = field(default_factory=dict)   # y, event, donor, theta (simulation only)


def sample_theta(forest, M, rng):
    lam = rng.normal(ev.LAM_MU[:4], ev.LAM_SD)
    for i in range(4):                                   # truncated components by rejection (no draw when unbounded,
        while lam[i] < ev.LAM_MIN[i]:                    # so the random stream of every seeded dataset is unchanged)
            lam[i] = rng.normal(ev.LAM_MU[i], ev.LAM_SD)
    if ev.N_LAM > 4:                                     # the directional weight, drawn after the plan's four
        extra = rng.normal(ev.LAM_MU[4], ev.LAM_SD)
        while extra < ev.LAM_MIN[4]:
            extra = rng.normal(ev.LAM_MU[4], ev.LAM_SD)
        lam = np.append(lam, extra)
    p_sim = SIM_U_RULES / ev.N_U_SLOTS if SIM_U_RULES > 0 else None
    u = {e: ev.sample_u(forest.length[e], rng, p=p_sim) for e in forest.edges}
    rho = rng.gamma(ev.RHO_A, 1 / ev.RHO_RATE, size=M)
    omega = float(rng.gamma(ev.OMEGA_A, 1 / ev.OMEGA_RATE))
    t_root = float(rng.gamma(ev.TROOT_A, 1 / ev.TROOT_RATE))
    return Theta(lam, u, rho, omega, t_root)


def edge_params(forest, theta):
    return {e: EdgeParams(theta.lam, theta.u.get(e, {}), t=forest.length[e]) for e in forest.edges}


def adapt_params(theta):
    return EdgeParams(theta.lam, {})


def simulate(forest, theta, M, rng, root_forms=None):
    """Events in topological order; donors are always earlier in that order, so their final
    forms are known when the borrowing happens (see Forest.donor_candidates).
    root_forms: optional {(root, c): form} to fix particular root forms (used by studies)."""
    eps = edge_params(forest, theta)
    adapt = adapt_params(theta)
    y = {}
    root_forms = root_forms or {}
    # pass 1: vertical only
    for c in range(M):
        for v in forest.topo:
            p = forest.parent[v]
            if p < 0:
                y[(v, c)] = root_forms.get((v, c), ev.sample_G(rng))
            else:
                y[(v, c)] = eps[v].sample(y[(p, c)], rng)
    event, donor = {}, {}
    # pass 2: events
    for c in range(M):
        for v in forest.topo:
            p = forest.parent[v]
            if p < 0:
                continue
            cands = forest.donor_candidates(v)
            lp = ev.event_logprobs(theta.rho[c], theta.omega if cands else 0.0, forest.length[v])
            kind = int(rng.choice(3, p=np.exp(lp)))
            event[(v, c)] = kind
            if kind == ev.INHERIT:
                y[(v, c)] = eps[v].sample(y[(p, c)], rng)
            elif kind == ev.REPLACE:
                y[(v, c)] = ev.sample_G(rng)
            else:
                d = int(rng.choice(cands))
                donor[(v, c)] = d
                y[(v, c)] = adapt.sample(y[(d, c)], rng)
    leaf_forms = {(l, c): y[(l, c)] for l in forest.leaves for c in range(M)}
    return Dataset(forest, M, leaf_forms, truth=dict(y=y, event=event, donor=donor, theta=theta))


def simulate_pair(related, depth, M, seed, theta=None):
    rng = np.random.default_rng(seed)
    forest = benchmark_a(related=related, t_root=depth)
    if theta is None:
        theta = sample_theta(forest, M, rng)
        theta.t_root = depth
    return simulate(forest, theta, M, rng)


def simulate_matched_pair(depth, M, seed):
    """A related and an unrelated dataset with the same within-family parameters (plan §8.3)."""
    rng = np.random.default_rng(seed)
    f1 = benchmark_a(related=True, t_root=depth)
    th = sample_theta(f1, M, rng); th.t_root = depth
    related = simulate(f1, th, M, np.random.default_rng(seed + 1))
    f0 = benchmark_a(related=False)
    th0 = th.copy(); th0.u = {e: th.u[e] for e in f0.edges}
    unrelated = simulate(f0, th0, M, np.random.default_rng(seed + 2))
    return related, unrelated

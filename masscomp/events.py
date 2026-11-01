"""Lexical events as competing hazards, single-event per edge (plan §3.3), and priors (§3.5)."""
import math
import numpy as np
from scipy import stats
from .segments import N_SEG, N_OUT

INHERIT, REPLACE, BORROW = 0, 1, 2
MIN_LEN, MAX_LEN = 2, 6
LOG_G_LEN = -math.log(MAX_LEN - MIN_LEN + 1)

# ---- priors -----------------------------------------------------------------------
LAM_MU = np.array([4.5, 3.0, 2.0, 6.0])   # identity bonus, feature distance, deletion, insertion
LAM_SD = 1.0
SIGMA2_U = 1.0                            # u_e[s,s'] ~ N(0, t_e * SIGMA2_U)
N_U_SLOTS = N_SEG * (N_OUT - 1) + N_SEG   # off-diagonal sub/del entries + insertion entries
PI_U = 3.0 / N_U_SLOTS                    # each slot present with this prob (expected 3 per edge)
RHO_A, RHO_RATE = 2.0, 2.0
OMEGA_A, OMEGA_RATE = 1.0, 10.0
TROOT_A, TROOT_RATE = 2.0, 1.0


def event_logprobs(rho, omega, t):
    """log P(inherit), log P(replace), log P(borrow) for one edge/concept.
    omega=0 (edge with no donor candidates) gives P(borrow)=0."""
    if omega <= 0:
        s = -rho * t
        return np.array([s, math.log1p(-math.exp(s)) if s < -1e-12 else math.log(max(-s, 1e-300)), -1e30])
    h = rho + omega
    s = -h * t
    one_minus = math.log1p(-math.exp(s)) if s < -1e-12 else math.log(max(-s, 1e-300))
    return np.array([s, math.log(rho / h) + one_minus, math.log(omega / h) + one_minus])


def log_G(form):
    """Innovation distribution: uniform length 2-6, uniform segments. Lengths outside that
    range get a soft floor rather than zero (a hard zero in the support breaks tempering)."""
    L = len(form)
    if L < MIN_LEN or L > MAX_LEN:
        return -40.0 - 8.0 * (max(0, L - MAX_LEN) + max(0, MIN_LEN - L))
    return LOG_G_LEN - L * math.log(N_SEG)


def sample_G(rng):
    L = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    return tuple(int(x) for x in rng.integers(N_SEG, size=L))


def u_slots():
    out = [(s, o) for s in range(N_SEG) for o in range(N_OUT) if o != s]
    out += [("ins", s) for s in range(N_SEG)]
    return out


U_SLOTS = u_slots()


def log_prior_u(u, t):
    """Sparse spike-and-slab: each slot present w.p. PI_U, value N(0, t*SIGMA2_U)."""
    k = len(u)
    lp = k * math.log(PI_U) + (N_U_SLOTS - k) * math.log1p(-PI_U)
    var = t * SIGMA2_U
    for w in u.values():
        lp += -0.5 * w * w / var - 0.5 * math.log(2 * math.pi * var)
    return lp


def sample_u(t, rng):
    u = {}
    for slot in U_SLOTS:
        if rng.random() < PI_U:
            u[slot] = float(rng.normal(0, math.sqrt(t * SIGMA2_U)))
    return u


def log_prior_lam(lam):
    return float(stats.norm.logpdf(lam, LAM_MU, LAM_SD).sum())


def log_prior_rho(rho):
    return float(stats.gamma.logpdf(rho, RHO_A, scale=1 / RHO_RATE).sum())


def log_prior_omega(omega):
    return float(stats.gamma.logpdf(omega, OMEGA_A, scale=1 / OMEGA_RATE))


def log_prior_troot(t):
    return float(stats.gamma.logpdf(t, TROOT_A, scale=1 / TROOT_RATE))

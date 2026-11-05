"""Lexical events as competing hazards, single-event per edge (plan §3.3), and priors (§3.5)."""
import math, os
import numpy as np
from scipy import stats
from .segments import N_SEG, N_OUT

INHERIT, REPLACE, BORROW = 0, 1, 2
MIN_LEN, MAX_LEN = 2, 6
LOG_G_LEN = -math.log(MAX_LEN - MIN_LEN + 1)

# ---- priors -----------------------------------------------------------------------
LAM_MU = np.array([4.5, 3.0, 2.0, 6.0])   # identity bonus, feature distance, deletion, insertion
LAM_SD = float(__import__("os").environ.get("MASSCOMP_LAM_SD", "1.0"))   # real-data probes: a tight prior pins lambda (see README, 4 Nov)
SIGMA2_U = 16.0                           # u_e[s,s'] ~ N(0, t_e * SIGMA2_U); slab sd 4*sqrt(t).
# With sd 1 (the plan's value) a prior-drawn rule almost never changes a substitution probability
# by more than a few percent (identity sits at logit ~6, a one-feature neighbour at ~ -0.3), so
# no prior-drawn rule is audible in the data and the u layer cannot be tested. Sound laws are
# near-categorical: a slab wide enough to reach logit parity with identity is the right prior.
N_U_SLOTS = N_SEG * (N_OUT - 1) + N_SEG   # off-diagonal sub/del entries + insertion entries
# Rule budget: each slot present w.p. pi_u(t). The plan's prior expects 3 rules per edge whatever its
# length (MASSCOMP_U_RULES, default 3). MASSCOMP_U_PER_T=<kappa> makes it kappa * t_e rules per edge
# instead (a real edge carries dozens; the budget should scale with time like the slab variance does).
U_RULES = float(os.environ.get("MASSCOMP_U_RULES", "3"))
U_PER_T = float(os.environ.get("MASSCOMP_U_PER_T", "0"))
PI_U = U_RULES / N_U_SLOTS                 # the length-free default, kept for callers that need a number


def pi_u(t):
    """Inclusion probability of one rule slot on an edge of length t."""
    if U_PER_T > 0:
        return min(0.5, U_PER_T * t / N_U_SLOTS)
    return PI_U
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


# ---- innovation distribution G -------------------------------------------------------------
# G(form) = G_LEN[len] * prod_i G_SEG[form_i]: a length distribution and a segment unigram. The plan's
# flat G (uniform length 2-6, uniform segments) is the default; set_G / fit_G install a phonotactic one
# (length histogram and segment frequencies of a wordlist), which is what real data need (README,
# Romance probe of 4 Nov: under flat G a root segment costs log N_SEG = 3.7 nats, and the model
# reconstructs consonant skeletons). Lengths with no probability get a soft floor, -40 - 8 per
# segment beyond the support, rather than zero (a hard zero in the support breaks tempering).
G_T_MAX = 32


def _flat_G():
    glen = np.full(G_T_MAX + 1, -np.inf)
    for L in range(MIN_LEN, MAX_LEN + 1):
        glen[L] = LOG_G_LEN
    return glen, np.full(N_SEG, -math.log(N_SEG))


def _with_floor(glen):
    """Lengths outside the distribution's support get the soft floor relative to its nearest edge."""
    out = glen.copy()
    sup = np.where(np.isfinite(glen))[0]
    lo, hi = int(sup.min()), int(sup.max())
    for L in range(len(out)):
        if not np.isfinite(out[L]):
            out[L] = -40.0 - 8.0 * (max(0, L - hi) + max(0, lo - L))
    return out


G_LEN_RAW, G_SEG = _flat_G()
G_LEN = _with_floor(G_LEN_RAW)
G_SEG_P = np.exp(G_SEG)
G_LEN_P = np.exp(G_LEN_RAW); G_LEN_P[~np.isfinite(G_LEN_RAW)] = 0.0; G_LEN_P /= G_LEN_P.sum()
G_IS_FLAT = True


def set_G(len_logp, seg_logp):
    """Install an innovation distribution: len_logp[L] (normalised over its finite entries, -inf
    elsewhere), seg_logp[s] over the inventory (normalised)."""
    global G_LEN_RAW, G_LEN, G_SEG, G_SEG_P, G_LEN_P, G_IS_FLAT
    glen = np.full(G_T_MAX + 1, -np.inf); n = min(len(len_logp), G_T_MAX + 1); glen[:n] = len_logp[:n]
    fin = np.isfinite(glen); glen[fin] -= np.log(np.exp(glen[fin]).sum())
    seg = np.asarray(seg_logp, float); seg = seg - np.log(np.exp(seg).sum())
    G_LEN_RAW, G_SEG = glen, seg
    G_LEN = _with_floor(glen); G_SEG_P = np.exp(seg)
    G_LEN_P = np.exp(glen); G_LEN_P[~fin] = 0.0; G_LEN_P /= G_LEN_P.sum()
    fl, fs = _flat_G()
    G_IS_FLAT = bool(np.allclose(np.where(np.isfinite(glen), glen, -1e9), np.where(np.isfinite(fl), fl, -1e9)) and np.allclose(seg, fs))


def reset_G():
    set_G(*_flat_G())


def fit_G(forms, len_pseudo=1.0, seg_pseudo=0.5, max_len=None):
    """A phonotactic G from a collection of forms (tuples of segment ids): add-k smoothed length
    histogram over 1..max_len (default: the longest form seen) and add-k smoothed segment unigram."""
    forms = list(forms)
    T = max_len or max(len(f) for f in forms)
    cnt = np.full(T + 1, len_pseudo); cnt[0] = 0.0
    for f in forms:
        if 1 <= len(f) <= T:
            cnt[len(f)] += 1
    seg = np.full(N_SEG, seg_pseudo)
    for f in forms:
        for x in f:
            seg[x] += 1
    with np.errstate(divide="ignore"):
        glen = np.full(T + 1, -np.inf); glen[1:] = np.log(cnt[1:] / cnt[1:].sum())
        set_G(glen, np.log(seg / seg.sum()))


def random_G(seed, max_len=8, len_conc=2.0, seg_conc=0.7):
    """A random non-flat G (Dirichlet length table over 1..max_len and Dirichlet unigram) for the
    consistency tests: the simulator and the sampler must agree under any G, not just the flat one."""
    rng = np.random.default_rng(seed)
    glen = np.full(max_len + 1, -np.inf); glen[1:] = np.log(rng.dirichlet(np.full(max_len, len_conc)))
    set_G(glen, np.log(rng.dirichlet(np.full(N_SEG, seg_conc))))


def save_G(path):
    import json
    json.dump({"len_logp": [float(v) for v in G_LEN_RAW], "seg_logp": [float(v) for v in G_SEG]}, open(path, "w"))


def load_G(path):
    import json
    d = json.load(open(path))
    set_G(np.array(d["len_logp"], float), np.array(d["seg_logp"], float))


def install_from_env():
    """MASSCOMP_G = flat (default) | random:<seed>[:<max_len>] | <path to a json written by save_G>."""
    spec = os.environ.get("MASSCOMP_G", "flat")
    if spec in ("flat", "fit"):              # "fit": a study fits G from its data (studies/realdata_probe.py); flat until then
        reset_G()
    elif spec.startswith("random:"):
        parts = spec.split(":")
        random_G(int(parts[1]), int(parts[2]) if len(parts) > 2 else 8)
    else:
        load_G(spec)


install_from_env()


def log_G(form):
    """log G(form) under the installed innovation distribution."""
    L = len(form)
    base = G_LEN[L] if L <= G_T_MAX else -40.0 - 8.0 * (L - G_T_MAX)
    return base + sum(G_SEG[s] for s in form)


def sample_G(rng):
    if G_IS_FLAT:                     # the plan's G, drawn exactly as before so every seeded dataset is unchanged
        L = int(rng.integers(MIN_LEN, MAX_LEN + 1))
        return tuple(int(x) for x in rng.integers(N_SEG, size=L))
    L = int(rng.choice(len(G_LEN_P), p=G_LEN_P))
    return tuple(int(x) for x in rng.choice(N_SEG, size=L, p=G_SEG_P))


def u_slots():
    out = [(s, o) for s in range(N_SEG) for o in range(N_OUT) if o != s]
    out += [("ins", s) for s in range(N_SEG)]
    return out


U_SLOTS = u_slots()


def log_prior_u(u, t):
    """Sparse spike-and-slab: each slot present w.p. pi_u(t), value N(0, t*SIGMA2_U)."""
    k = len(u); p = pi_u(t)
    lp = k * math.log(p) + (N_U_SLOTS - k) * math.log1p(-p)
    var = t * SIGMA2_U
    for w in u.values():
        lp += -0.5 * w * w / var - 0.5 * math.log(2 * math.pi * var)
    return lp


def sample_u(t, rng):
    u = {}
    p = pi_u(t)
    for slot in U_SLOTS:
        if rng.random() < p:
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

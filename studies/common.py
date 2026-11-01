"""Shared helpers for the studies: multi-chain fits, diagnostics, JSON output."""
import json, math, os, time
import numpy as np
from masscomp.simulate import sample_theta, Theta
from masscomp.mcmc import run_chain
from masscomp import events as ev


def fit(ds, seed, n_chains=3, n_sweeps=1500, burn=500, thin=5, **kw):
    """Run n_chains from independent prior draws; return per-chain sample lists."""
    chains = []
    for k in range(n_chains):
        rng = np.random.default_rng(seed * 100 + k)
        th0 = sample_theta(ds.forest, ds.M, rng)
        _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=n_sweeps, burn=burn, thin=thin, **kw)
        chains.append(out)
    return chains


def rhat(chains_values):
    """Gelman-Rubin R-hat for a scalar, chains_values: list of 1-D arrays."""
    x = np.array(chains_values)
    m, n = x.shape
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    if W <= 0:
        return 1.0
    return math.sqrt(((n - 1) / n * W + B / n) / W)


def ess(xs):
    """Effective sample size by initial positive sequence of autocorrelations."""
    x = np.asarray(xs, dtype=float); n = len(x); x = x - x.mean()
    if n < 4 or x.var() == 0:
        return float(n)
    acf = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    s = 0.0
    for k in range(1, n):
        if acf[k] < 0.05:
            break
        s += acf[k]
    return n / (1 + 2 * s)


def interval(xs, q=0.9):
    xs = np.sort(np.asarray(xs)); a = (1 - q) / 2
    return float(xs[int(a * len(xs))]), float(xs[min(len(xs) - 1, int((1 - a) * len(xs)))])


def covered(truth, xs, q=0.9):
    lo, hi = interval(xs, q)
    return bool(lo <= truth <= hi)


def pooled(chains, key, idx=None):
    vals = []
    for ch in chains:
        for o in ch:
            v = o[key]
            vals.append(v if idx is None else v[idx])
    return np.array(vals)


def per_chain(chains, key, idx=None):
    return [np.array([o[key] if idx is None else o[key][idx] for o in ch]) for ch in chains]


def recoverable_roots(ds):
    """(node, c) pairs for the joined root and the family roots whose true form is in principle
    recoverable: at least one child edge inherited in the simulated truth. Root forms replaced
    on every child edge are unobserved by construction and are excluded from the metric."""
    f = ds.forest; t = ds.truth
    nodes = list(f.roots) + [v for v in f.internal if f.family[v] >= 0 and f.parent[v] in f.roots or (f.parent[v] < 0)]
    nodes = sorted(set(nodes))
    out = {}
    for v in nodes:
        for c in range(ds.M):
            if any(t["event"].get((ch, c)) == ev.INHERIT for ch in f.children[v]):
                out[(v, c)] = t["y"][(v, c)]
    return out


def root_set_coverage(chains, truth_roots, q=0.9, key_field="roots"):
    """Fraction of (node, concept) whose true form lies in the smallest set of forms with
    posterior mass >= q."""
    hits = 0; n = 0
    for key, true_form in truth_roots.items():
        counts = {}
        for ch in chains:
            for o in ch:
                f = o[key_field].get(key)
                if f is None:
                    continue
                counts[f] = counts.get(f, 0) + 1
        tot = sum(counts.values()); acc = 0.0; inset = set()
        for f, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            inset.add(f); acc += c / tot
            if acc >= q:
                break
        hits += true_form in inset; n += 1
    return hits / n if n else float("nan")


def save(name, obj):
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}.json"
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)
    return path


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)

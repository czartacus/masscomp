"""Shared helpers for the studies: multi-chain fits, diagnostics, JSON output."""
import json, math, os, time
import numpy as np
from masscomp.simulate import sample_theta, Theta
from masscomp.mcmc import run_chain
from masscomp import events as ev


N_PROCS = int(os.environ.get("MASSCOMP_PROCS", "1"))   # parallel processes per study job


def _one_chain(args):
    ds, seed, k, n_sweeps, burn, thin, kw = args
    rng = np.random.default_rng(seed * 100 + k)
    th0 = sample_theta(ds.forest, ds.M, rng)
    _, out = run_chain(ds, th0, seed=seed * 100 + 50 + k, n_sweeps=n_sweeps, burn=burn, thin=thin, **kw)
    return out


def fit(ds, seed, n_chains=3, n_sweeps=1500, burn=500, thin=5, **kw):
    """Run n_chains from independent prior draws; return per-chain sample lists. Chains run in
    parallel processes when MASSCOMP_PROCS > 1 (same seeds, so results are identical either way)."""
    jobs = [(ds, seed, k, n_sweeps, burn, thin, kw) for k in range(n_chains)]
    if N_PROCS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(N_PROCS, n_chains)) as ex:
            return list(ex.map(_one_chain, jobs))
    return [_one_chain(j) for j in jobs]


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
    if isinstance(obj, dict) and "env" not in obj:        # record the sampler switches the job ran with
        obj["env"] = {k: v for k, v in os.environ.items() if k.startswith("MASSCOMP_")}
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


def node_conditional(s, v, c):
    """The exact conditional of the form at internal node v for concept c given everything else in
    the sampler's current state, with the event of v's incoming edge (if any) and the events of v's
    child edges (inherit / replace / borrow from each donor) marginalised. Returns logp(form), the
    log posterior probability of `form` at (v, c), or None when a cross-family edge borrows from v
    in this state (its term depends on y_v but the lattice holds at most two out-edges). Same
    arithmetic as move_root_regrow; used to Rao-Blackwellise root-form probabilities, because the
    sample-set estimate of the 90% set is biased against diffuse posteriors."""
    import itertools, math
    from masscomp.ffbs import log_Z
    st, f = s.st, s.st.f
    for e in f.edges:
        if st.event.get((e, c)) == ev.BORROW and st.donor.get((e, c)) == v:
            return None
    chs = list(f.children[v])
    opts = {ch: [(ev.INHERIT, None), (ev.REPLACE, None)] + [(ev.BORROW, d) for d in st.cands[ch] if d != v] for ch in chs}
    # incoming options for v itself (the parent form or a donor form as the lattice input)
    if f.parent[v] >= 0:
        ins = [(ev.INHERIT, None), (ev.REPLACE, None)] + [(ev.BORROW, d) for d in st.cands[v]]
    else:
        ins = [(ev.REPLACE, None)]
    cfgs = []
    for kin, din in ins:
        w_in = st.evp[v][c][kin] if f.parent[v] >= 0 else 0.0
        if kin == ev.INHERIT:
            in_mode, tab, x = 1, st.ep[v], st.y[(f.parent[v], c)]
        elif kin == ev.REPLACE:
            in_mode, tab, x = 0, None, None
        else:
            w_in += -math.log(len(st.cands[v])); in_mode, tab, x = 1, st.adapt, st.y[(din, c)]
        tabs = (tab.log_sub_np, tab.log_ins_np) if tab is not None else None
        for combo in itertools.product(*[opts[ch] for ch in chs]):
            w = w_in; lat = []
            for ch, (k, d) in zip(chs, combo):
                w += st.evp[ch][c][k]
                if k == ev.REPLACE:
                    w += ev.log_G(st.y[(ch, c)])
                elif k == ev.BORROW:
                    w += -math.log(len(st.cands[ch])) + st.adapt.logprob(st.y[(d, c)], st.y[(ch, c)])
                else:
                    lat.append((st.ep[ch], st.y[(ch, c)]))
            if len(lat) > 2:
                return None
            lz = log_Z(in_mode, tabs, x, [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat])
            cfgs.append((w, lz, in_mode, tab, x, lat))
    tot = _lse_list([w + lz for w, lz, *_ in cfgs])

    def logp(form):
        acc = []
        for w, lz, in_mode, tab, x, lat in cfgs:
            sc = (ev.log_G(form) if in_mode == 0 else tab.logprob(x, form)) + sum(ep.logprob(form, z) for ep, z in lat)
            acc.append(w + sc)
        return _lse_list(acc) - tot
    return logp


def root_conditional(s, R, c):
    return node_conditional(s, R, c)


def _lse_list(xs):
    m = max(xs)
    if m <= -1e29:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def root_hpd_rb(ds, chains, truth_roots, q=0.9, n_states=40, seed=0):
    """Is the true form of each root node in truth_roots inside the q-HPD set of its marginal posterior?
    Estimated with Rao-Blackwellised probabilities rather than the sample set: p(f) = mean over recorded
    states s of p_s(f), where p_s is the exact conditional of the form given the rest of state s with the
    adjacent events marginalised (node_conditional). The truth is in the HPD set iff the posterior mass
    of forms more probable than it is below q, estimated as the fraction of recorded samples t with
    p(y_t) > p(truth). The sample-set estimate (root_set_coverage) is biased against diffuse posteriors:
    with hundreds of forms at probability < 1%, the forms seen in a few hundred samples cover a small
    fraction of the mass and the truth is counted as missed although the true HPD set is large and
    contains it (seed 424 of the depth-1.5 recovery study: 0.00 by the sample set, 1.00 by this
    estimate, 4 Nov 2026). Returns (coverage, per-(node, concept) dict of (log p(truth), mass_above, hit))."""
    from masscomp.mcmc import Sampler
    keys = sorted(truth_roots)
    samples = [o for ch in chains for o in ch if "latents" in o]
    if not keys or not samples:
        return float("nan"), {}
    rng = np.random.default_rng(seed)
    sub = [samples[i] for i in sorted(rng.choice(len(samples), size=min(n_states, len(samples)), replace=False))]
    forms = {k: {o["roots"][k] for o in samples} | {truth_roots[k]} for k in keys}
    forms = {k: sorted(fs) for k, fs in forms.items()}
    logp_acc = {k: np.full(len(fs), -np.inf) for k, fs in forms.items()}
    n_used = {k: 0 for k in keys}
    for o in sub:
        th = Theta(np.array(o["lam"]), {e: dict(d) for e, d in o["u"].items()}, np.array(o["rho"]), float(o["omega"]), float(o["t_root"]))
        y, event, donor = o["latents"]
        s = Sampler(ds, th, np.random.default_rng(0), init=(dict(y), dict(event), dict(donor)))
        for (v, c) in keys:
            lp = node_conditional(s, v, c)
            if lp is None:
                continue
            vals = np.array([lp(f) for f in forms[(v, c)]])
            logp_acc[(v, c)] = np.logaddexp(logp_acc[(v, c)], vals); n_used[(v, c)] += 1
    per = {}
    hits = 0; n = 0
    for k in keys:
        if n_used[k] == 0:
            continue
        lp = logp_acc[k] - math.log(n_used[k])                 # log of the RB marginal p(f)
        idx = {f: i for i, f in enumerate(forms[k])}
        lpt = lp[idx[truth_roots[k]]]
        above = np.mean([lp[idx[o["roots"][k]]] > lpt for o in samples])
        hit = bool(above < q)
        hits += hit; n += 1
        per[k] = (float(lpt), float(above), hit)
    return (hits / n if n else float("nan")), per

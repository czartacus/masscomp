"""Stepping-stone marginal likelihood (plan §6).

log p(X|H) = sum_k log E_{beta_k}[ L^{beta_{k+1} - beta_k} ],  beta_k = (k/K)^3,
with the expectation over samples drawn at power posterior beta_k, chained.
`L` is the likelihood part of the target (leaf-form terms); the prior part is everything else.
"""
import math
import numpy as np
from . import events as ev
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


def _pt_worker(conn, ds, theta0, seeds, betas, sweep_kw, init=None, from_prior=True, full=False):
    """Owns a block of PT chains. Protocol: receive list of betas for its chains (or None to stop),
    sweep each chain once at that beta, reply with the untempered log-likelihoods."""
    chains = [Sampler(ds, theta0.copy(), np.random.default_rng(sd), beta=b, init=init, init_from_prior=init is None and from_prior)
              for sd, b in zip(seeds, betas)]
    while True:
        msg = conn.recv()
        if msg is None:
            break
        bl, snap = msg if isinstance(msg, tuple) else (msg, -1)
        for s, b in zip(chains, bl):
            s.beta = b
            if full:
                s.bp = b
            s.sweep(**sweep_kw)
        conn.send(([s.st.log_lik() for s in chains], chains[snap].snapshot() if snap >= 0 else None))
    conn.close()


def stepping_stone_pt(ds, theta0, seed, K=24, n_sweeps=2000, burn=500, thin=5, power=2.0, verbose=False,
                      swap_every=1, n_procs=None, adapt=True, adapt_every=25, warmup=0, record_cold=False,
                      init_from_prior=True, full=False, **sweep_kw):
    """Parallel-tempered stepping stone: all K+1 power posteriors run at once, and after every sweep
    neighbouring stones propose to exchange states (accepted with prob min(1, exp((b_i - b_j)(ll_j - ll_i)))).
    The estimator is the same as stepping_stone; the ladder only improves mixing, since a state stuck
    in one forest/root mode at a cold stone can be replaced by one that wandered in from a hot stone.
    Cost per sweep is (K+1) single-chain sweeps, so n_sweeps here corresponds to the per-stone chain
    length of the sequential version. Returns (log_ml, per-stone terms, swap acceptance per pair).

    n_procs > 1 (default: env MASSCOMP_PROCS) spreads the K+1 chains over persistent worker processes.
    A swap then exchanges the two chains' beta labels rather than their states, so only scalars cross
    process boundaries; the sampled distribution is identical (each chain keeps its own RNG stream).

    adapt=True re-spaces the interior betas every adapt_every sweeps during the first 80% of burn-in so
    that d_beta * sd(ll) is equal across stones (see maybe_adapt); beta_0 = 0 and beta_K = 1 stay fixed
    and the ladder is frozen before any likelihood is recorded, so the estimator is unchanged.

    record_cold=True also returns, in info["cold"], the snapshots (Sampler.snapshot) of whichever chain
    holds beta = 1 at each recorded sweep: a tempered posterior sample for studies whose single chains
    cannot leave the child-copy basin (root_diag, 3 Nov 2026).

    full=True also tempers the prior-side terms (Sampler.bp = beta): the whole target is flattened, so
    hot replicas can cross barriers that sit in the internal-edge / event terms rather than the leaf
    likelihood. The stepping-stone value is then NOT a marginal likelihood; use with record_cold for
    mixing only.

    warmup > 0 first runs one chain at beta = 1 for that many sweeps from a prior draw and starts every
    stone from its final state (parameters and latents); the stones then relax downhill to their own
    equilibria, which is fast, instead of each climbing from the prior, which the form sampler does slowly
    (a 3 Nov 2026 check showed stones still drifting by hundreds of nats after 200 sweeps from the prior). The grid of 3 Nov 2026 had acceptance 0.00 at
    the top ten stones with the fixed (k/K)^2 ladder and replicate sd ~19 nats."""
    import os
    if n_procs is None:
        n_procs = int(os.environ.get("MASSCOMP_PROCS", "1"))
    rng = np.random.default_rng(seed)
    betas = [(k / K) ** power for k in range(K + 1)]
    seeds = [int(rng.integers(2 ** 31)) for _ in range(K + 1)]
    init = None
    if warmup > 0:
        warm = Sampler(ds, theta0.copy(), np.random.default_rng(int(rng.integers(2 ** 31))), beta=1.0, init_from_prior=init_from_prior)
        for _ in range(warmup):
            warm.sweep(**sweep_kw)
        theta0 = warm.st.theta.copy()
        init = (dict(warm.st.y), dict(warm.st.event), dict(warm.st.donor))
        if verbose:
            print(f"  warmup {warmup} sweeps at beta=1: ll={warm.st.log_lik():.1f}", flush=True)
    lls = [[] for _ in range(K + 1)]
    cold = []            # snapshots of whichever chain holds beta = 1 (record_cold)
    acc = np.zeros(K); tries = np.zeros(K)
    # rank[c] = stone index currently held by chain c; chain_of[k] = chain holding stone k
    rank = list(range(K + 1)); chain_of = list(range(K + 1))

    def do_swaps(i, cur_by_chain):
        cur = [cur_by_chain[chain_of[k]] for k in range(K + 1)]   # ll by stone
        for k in range(i % 2, K, 2):
            a = (betas[k] - betas[k + 1]) * (cur[k + 1] - cur[k])
            tries[k] += 1
            if a >= 0 or rng.random() < math.exp(a):
                ca, cb = chain_of[k], chain_of[k + 1]
                chain_of[k], chain_of[k + 1] = cb, ca
                rank[ca], rank[cb] = k + 1, k
                cur[k], cur[k + 1] = cur[k + 1], cur[k]
                acc[k] += 1
        return cur

    win_ll = [[] for _ in range(K + 1)]
    adapt_until = int(0.8 * burn)

    def maybe_adapt(i):
        """Re-space the interior betas so that d_beta * sd(ll) is equal across the ladder. Swap acceptance
        between neighbours is governed by d_beta * |ll_j - ll_i|, and the variance of stone k's estimate by
        d_beta_k * sd_k(ll), so this equalises both. Works from the likelihood spread measured over the last
        window at each stone (unlike acceptance-gradient schemes, it moves even when all acceptances are 0)."""
        if not adapt or i >= adapt_until or i == 0 or i % adapt_every:
            return
        sd = np.array([np.std(w) if len(w) > 2 else np.nan for w in win_ll])
        for w in win_ll:
            w.clear()
        if np.isnan(sd).any():
            return
        sd = np.maximum(sd, 1e-3)
        b = np.array(betas)
        # cumulative integral of sd over beta (trapezoid on the current ladder), then equal increments
        seg = 0.5 * (sd[1:] + sd[:-1]) * np.diff(b)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(0.0, cum[-1], K + 1)[1:-1]
        new = np.interp(targets, cum, b)
        new = np.maximum.accumulate(np.maximum(new, 1e-5))
        for k in range(1, K):
            betas[k] = float(min(new[k - 1], 1.0 - 1e-6))

    _do_swaps = do_swaps

    def do_swaps(i, cur_by_chain):
        cur = _do_swaps(i, cur_by_chain)
        if i < adapt_until:
            for k in range(K + 1):
                win_ll[k].append(cur[k])
        maybe_adapt(i)
        return cur

    if n_procs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        n_procs = min(n_procs, K + 1)
        blocks = [list(range(w, K + 1, n_procs)) for w in range(n_procs)]
        procs, conns = [], []
        for blk in blocks:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_pt_worker, args=(child, ds, theta0, [seeds[c] for c in blk],
                                                      [betas[c] for c in blk], sweep_kw, init, init_from_prior, full), daemon=True)
            p.start(); procs.append(p); conns.append(parent)
        try:
            for i in range(n_sweeps):
                want = record_cold and i >= burn and (i - burn) % thin == 0
                for blk, conn in zip(blocks, conns):
                    local = blk.index(chain_of[K]) if (want and chain_of[K] in blk) else -1
                    conn.send(([betas[rank[c]] for c in blk], local))
                cur_by_chain = [0.0] * (K + 1)
                for blk, conn in zip(blocks, conns):
                    lls_blk, snap = conn.recv()
                    for c, ll in zip(blk, lls_blk):
                        cur_by_chain[c] = ll
                    if snap is not None:
                        cold.append(snap)
                if i == burn:
                    acc[:] = 0; tries[:] = 0            # report acceptance on the frozen ladder
                if i % swap_every == 0:
                    cur = do_swaps(i, cur_by_chain)
                else:
                    cur = [cur_by_chain[chain_of[k]] for k in range(K + 1)]
                if i >= burn and (i - burn) % thin == 0:
                    for k in range(K + 1):
                        lls[k].append(cur[k])
        finally:
            for conn in conns:
                conn.send(None)
            for p in procs:
                p.join()
    else:
        chains = [Sampler(ds, theta0.copy(), np.random.default_rng(sd), beta=b, init=init, init_from_prior=init is None and init_from_prior)
                  for sd, b in zip(seeds, betas)]
        for i in range(n_sweeps):
            for c, s in enumerate(chains):
                s.beta = betas[rank[c]]
                if full:
                    s.bp = s.beta
                s.sweep(**sweep_kw)
            cur_by_chain = [s.st.log_lik() for s in chains]
            if record_cold and i >= burn and (i - burn) % thin == 0:
                cold.append(chains[chain_of[K]].snapshot())
            if i == burn:
                acc[:] = 0; tries[:] = 0
            if i % swap_every == 0:
                cur = do_swaps(i, cur_by_chain)
            else:
                cur = [cur_by_chain[chain_of[k]] for k in range(K + 1)]
            if i >= burn and (i - burn) % thin == 0:
                for k in range(K + 1):
                    lls[k].append(cur[k])
    log_terms = []
    info = dict(mean_ll=[float(np.mean(lls[k])) for k in range(K + 1)], sd_ll=[float(np.std(lls[k])) for k in range(K + 1)],
                ess=[], warmup=warmup, cold=cold)
    for k in range(K):
        x = np.array(lls[k]); d = betas[k + 1] - betas[k]; m = x.max()
        w = np.exp(d * (x - m))
        log_terms.append(d * m + math.log(w.mean()))
        info["ess"].append(float(w.sum() ** 2 / (w ** 2).sum()))
        if verbose:
            print(f"  stone {k:2d} beta={betas[k]:.4f} mean ll={x.mean():.1f} sd={x.std():.1f} term={log_terms[-1]:.3f} "
                  f"ess={info['ess'][-1]:.1f} swap_acc={acc[k] / max(tries[k], 1):.2f}", flush=True)
    return float(sum(log_terms)), log_terms, acc / np.maximum(tries, 1), list(betas), info


def join_path(ds, theta0, seed, K=16, n_sweeps=1000, burn=300, thin=5, power=2.0, warmup=None, verbose=False,
              direction="down", init_from_prior=True, spacing="top", trace=None, ess_min=0.0, max_extend=2, **sweep_kw):
    """Log Bayes factor H1 : H0 by a stepping stone along the JOIN rather than along the likelihood.

    The target at join temperature alpha keeps the whole model except that the inherit / borrow terms of
    the two root edges are raised to alpha and the family roots' innovation prior G to 1 - alpha
    (State.alpha). alpha = 1 is H1; at alpha = 0 the family roots are independent G draws and the joined
    root, t_root, the root-edge rules and events are dummies with proper priors, so Z(0) = p(X | H0).
    Hence log p(X|H1) - log p(X|H0) = sum_k log E_{alpha_k}[exp((alpha_{k+1} - alpha_k) * Delta)] with
    Delta = State.join_delta(), a sum over 2M (edge, concept) terms. Compared with the likelihood path
    (stepping_stone / stepping_stone_pt), which carries the variance of the full leaf likelihood (hundreds
    of terms, replicate sd 20-60 nats at M = 25) and needs its two marginals estimated separately, the
    power variable here is small and the within-family posterior - the part that mixes slowly - is at
    full strength at every stone. One chain, cooled from alpha = 1 to 0 (direction "down") after a warm-up
    at alpha = 1, or heated from 0 ("up"). Returns (lbf, per-stone terms, sampler).

    trace: a dict to fill with what a results explorer needs: the per-concept mean of the power variable
    at every stone (delta_c, K x M) and at alpha = 1 (the second half of the warm-up), its thermodynamic-
    integration attribution of the LBF to concepts (ti_concept, summing to ti_total), and the alpha = 1
    posterior summaries - event fractions per edge, t_root, omega, lam, the sampled root forms."""
    rng = np.random.default_rng(seed)
    # Delta grows with alpha (the join terms matter most near alpha = 1), so the stones are dense at the
    # top by default: alpha_k = 1 - (1 - k/K)^power; spacing="bottom" is the likelihood path's (k/K)^power
    if spacing == "both":
        # dense at both ends: at alpha = 0 the joined root is a free G draw and the root-edge events come
        # from their prior, so Delta has a huge spread there too (real data: mean -400, sd 240 at the bottom
        # stone of a "top" ladder, ESS ~1); a two-sided power ladder keeps d * sd(Delta) of order one at both ends
        alphas = [0.5 * (2 * k / K) ** power if 2 * k <= K else 1.0 - 0.5 * (2 * (K - k) / K) ** power for k in range(K + 1)]
    else:
        alphas = [1.0 - (1.0 - k / K) ** power for k in range(K + 1)] if spacing == "top" else [(k / K) ** power for k in range(K + 1)]
    s = Sampler(ds, theta0.copy(), rng, beta=1.0, init_from_prior=init_from_prior)
    assert s.st.joined, "join_path needs the joined (H1) forest"
    terms = [None] * K
    M = s.st.M
    at1 = None
    if direction == "down":
        n_w = warmup if warmup is not None else 2 * n_sweeps
        if trace is not None:
            at1 = _Alpha1Record(s)
        for i in range(n_w):
            s.sweep(**sweep_kw)
            if at1 is not None and i >= n_w // 2 and (i - n_w // 2) % thin == 0:
                at1.add()
        order = range(K - 1, -1, -1)
    else:
        s.st.alpha = 0.0; s.st.recompute_all()
        for _ in range(warmup if warmup is not None else 2 * n_sweeps):
            s.sweep(**sweep_kw)
        order = range(K)
    dc_mean = np.zeros((K + 1, M)); dc_sd = np.zeros((K + 1, M)); loo = np.zeros(M); stone_sweeps = [n_sweeps] * K
    for k in order:
        s.st.alpha = alphas[k]; s.st.recompute_all()
        d = alphas[k + 1] - alphas[k]
        ds_ = []; dcs = []
        for i in range(n_sweeps):
            s.sweep(**sweep_kw)
            if i >= burn and (i - burn) % thin == 0:
                dc = s.st.join_delta_by_concept()
                ds_.append(float(dc.sum())); dcs.append(dc)
        # a stone whose effective sample size is below ess_min (the transition region on real data, where
        # the family roots lock onto the joined root and Delta jumps by hundreds) is sampled up to
        # max_extend further blocks of n_sweeps, no burn-in, before its term is taken
        ext = 0
        while ess_min > 0 and ext < max_extend:
            a = np.array(ds_); w = np.exp(d * (a - a.max()))
            if w.sum() ** 2 / (w ** 2).sum() >= ess_min:
                break
            for i in range(n_sweeps):
                s.sweep(**sweep_kw)
                if i % thin == 0:
                    dc = s.st.join_delta_by_concept()
                    ds_.append(float(dc.sum())); dcs.append(dc)
            ext += 1; stone_sweeps[k] += n_sweeps
        ds_ = np.array(ds_); dcs = np.array(dcs)
        dc_mean[k] = dcs.mean(axis=0); dc_sd[k] = dcs.std(axis=0)
        m = ds_.max()
        w = np.exp(d * (ds_ - m))
        terms[k] = d * m + math.log(w.mean())
        if trace is not None:
            # leave-one-concept-out stepping stone on the same samples: how much this stone's term falls
            # when concept c's share of the power variable is removed (a marginal contribution; the M
            # values do not sum to the total, unlike the TI trapezoid, which is biased at the bottom of
            # a coarse ladder where E[Delta] is dominated by the dummy root's random inherit events)
            X = ds_[:, None] - dcs                      # (n, M): Delta without concept c
            mc = X.max(axis=0)
            loo += terms[k] - (d * mc + np.log(np.exp(d * (X - mc)).mean(axis=0)))
        if verbose:
            print(f"  stone {k:2d} alpha={alphas[k]:.4f} mean delta={ds_.mean():.2f} sd={ds_.std():.2f} term={terms[k]:.3f} "
                  f"ess={w.sum() ** 2 / (w ** 2).sum():.1f}", flush=True)
    s.st.alpha = 1.0; s.st.recompute_all()
    if trace is not None:
        if at1 is not None and at1.n:
            dc_mean[K] = at1.dc_sum / at1.n
        else:
            dc_mean[K] = dc_mean[K - 1]
        # trapezoid over the stones: the TI attribution of the evidence to concepts
        ti = np.zeros(M)
        for k in range(K):
            ti += (alphas[k + 1] - alphas[k]) * 0.5 * (dc_mean[k] + dc_mean[k + 1])
        trace.update(alphas=[float(a) for a in alphas], delta_c=dc_mean.tolist(), delta_c_sd=dc_sd.tolist(),
                     ti_concept=ti.tolist(), ti_total=float(ti.sum()), ss_total=float(sum(terms)), loo_concept=loo.tolist(),
                     stone_terms=[float(t) for t in terms], stone_sweeps=stone_sweeps)
        if at1 is not None:
            trace.update(at1.summary())
    return float(sum(terms)), terms, s


class _Alpha1Record:
    """Posterior summaries at alpha = 1 (the H1 fit) collected during the warm-up of join_path."""

    def __init__(self, s):
        self.s = s; st = s.st; f = st.f
        self.n = 0; self.dc_sum = np.zeros(st.M)
        self.edges = list(f.edges)
        self.ev_counts = {e: np.zeros(3) for e in self.edges}
        self.borrow_from = {}
        self.t_root = []; self.omega = []; self.lam = []
        self.root_forms = {c: {} for c in range(st.M)}
        self.fam_forms = {(v, c): {} for v in f.children[f.join_root] for c in range(st.M)}

    def add(self):
        st = self.s.st; f = st.f
        self.n += 1
        self.dc_sum += st.join_delta_by_concept()
        for e in self.edges:
            for c in range(st.M):
                k = st.event[(e, c)]
                self.ev_counts[e][k] += 1
                if k == ev.BORROW:
                    key = (e, st.donor[(e, c)])
                    self.borrow_from[key] = self.borrow_from.get(key, 0) + 1
        self.t_root.append(st.theta.t_root); self.omega.append(st.theta.omega); self.lam.append(st.theta.lam.copy())
        R = f.join_root
        for c in range(st.M):
            y = st.y[(R, c)]; d = self.root_forms[c]; d[y] = d.get(y, 0) + 1
            for v in f.children[R]:
                y = st.y[(v, c)]; d = self.fam_forms[(v, c)]; d[y] = d.get(y, 0) + 1

    def summary(self):
        st = self.s.st; f = st.f; n = max(self.n, 1)
        top = lambda d: sorted(((list(k), v / n) for k, v in d.items()), key=lambda kv: -kv[1])[:8]
        return dict(n_at1=self.n,
                    events_at1={str(e): dict(name=f.names.get(e, str(e)), inherit=float(self.ev_counts[e][0] / (n * st.M)),
                                             replace=float(self.ev_counts[e][1] / (n * st.M)), borrow=float(self.ev_counts[e][2] / (n * st.M)))
                                for e in self.edges},
                    borrow_from_at1={f"{e}<-{d}": v / (n * st.M) for (e, d), v in self.borrow_from.items()},
                    t_root_at1=float(np.mean(self.t_root)), t_root_sd_at1=float(np.std(self.t_root)),
                    omega_at1=float(np.mean(self.omega)), lam_at1=[float(x) for x in np.mean(self.lam, axis=0)],
                    root_forms_at1={str(c): top(self.root_forms[c]) for c in range(st.M)},
                    fam_forms_at1={f"{v}:{c}": top(self.fam_forms[(v, c)]) for (v, c) in self.fam_forms})

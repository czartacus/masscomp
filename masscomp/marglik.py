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


def _pt_worker(conn, ds, theta0, seeds, betas, sweep_kw, init=None):
    """Owns a block of PT chains. Protocol: receive list of betas for its chains (or None to stop),
    sweep each chain once at that beta, reply with the untempered log-likelihoods."""
    chains = [Sampler(ds, theta0.copy(), np.random.default_rng(sd), beta=b, init=init, init_from_prior=init is None)
              for sd, b in zip(seeds, betas)]
    while True:
        msg = conn.recv()
        if msg is None:
            break
        for s, b in zip(chains, msg):
            s.beta = b
            s.sweep(**sweep_kw)
        conn.send([s.st.log_lik() for s in chains])
    conn.close()


def stepping_stone_pt(ds, theta0, seed, K=24, n_sweeps=2000, burn=500, thin=5, power=2.0, verbose=False,
                      swap_every=1, n_procs=None, adapt=True, adapt_every=25, warmup=0, **sweep_kw):
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
        warm = Sampler(ds, theta0.copy(), np.random.default_rng(int(rng.integers(2 ** 31))), beta=1.0, init_from_prior=True)
        for _ in range(warmup):
            warm.sweep(**sweep_kw)
        theta0 = warm.st.theta.copy()
        init = (dict(warm.st.y), dict(warm.st.event), dict(warm.st.donor))
        if verbose:
            print(f"  warmup {warmup} sweeps at beta=1: ll={warm.st.log_lik():.1f}", flush=True)
    lls = [[] for _ in range(K + 1)]
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
                                                      [betas[c] for c in blk], sweep_kw, init), daemon=True)
            p.start(); procs.append(p); conns.append(parent)
        try:
            for i in range(n_sweeps):
                for blk, conn in zip(blocks, conns):
                    conn.send([betas[rank[c]] for c in blk])
                cur_by_chain = [0.0] * (K + 1)
                for blk, conn in zip(blocks, conns):
                    for c, ll in zip(blk, conn.recv()):
                        cur_by_chain[c] = ll
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
        chains = [Sampler(ds, theta0.copy(), np.random.default_rng(sd), beta=b, init=init, init_from_prior=init is None)
                  for sd, b in zip(seeds, betas)]
        for i in range(n_sweeps):
            for c, s in enumerate(chains):
                s.beta = betas[rank[c]]
                s.sweep(**sweep_kw)
            cur_by_chain = [s.st.log_lik() for s in chains]
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
    for k in range(K):
        x = np.array(lls[k]); d = betas[k + 1] - betas[k]; m = x.max()
        w = np.exp(d * (x - m))
        log_terms.append(d * m + math.log(w.mean()))
        if verbose:
            print(f"  stone {k:2d} beta={betas[k]:.4f} mean ll={x.mean():.1f} sd={x.std():.1f} term={log_terms[-1]:.3f} "
                  f"ess={w.sum() ** 2 / (w ** 2).sum():.1f} swap_acc={acc[k] / max(tries[k], 1):.2f}", flush=True)
    return float(sum(log_terms)), log_terms, acc / np.maximum(tries, 1), list(betas)

"""Geweke (2004) 'getting it right' test of the sampler.
Two ways to draw from the joint p(theta, latents, data):
  marginal-conditional   : theta ~ prior, (latents, data) ~ simulate(theta)           [exact]
  successive-conditional : alternate one sampler sweep on the current data with a redraw of the data
                           given (theta, latents) - a Gibbs pass over the leaf forms, each drawn
                           exactly from its incoming conditional times the transducers of the edges
                           that borrow from it (the FFBS lattice, <= 2 borrowers)
If every move leaves the posterior invariant the two chains have the same marginal for any statistic
of (theta, latents, data). A move with a wrong MH ratio shows up as a z-score far from 0.
Usage: PYTHONPATH=. python studies/geweke.py [N] [M] [seed] [full|params|latents]      (env MASSCOMP_REGROW_P etc. select the moves)"""
import sys, time
from dataclasses import replace
import numpy as np
from masscomp.simulate import simulate, sample_theta, Dataset
from masscomp.tree import benchmark_a, with_root_length
from masscomp.mcmc import Sampler
from masscomp import events as ev
from masscomp.ffbs import draw as ffbs_draw

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
M = int(sys.argv[2]) if len(sys.argv) > 2 else 6
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MODE = sys.argv[4] if len(sys.argv) > 4 else "full"
ONLY = sys.argv[5] if len(sys.argv) > 5 else None
SWEEPS = int(__import__("os").environ.get("MASSCOMP_GEWEKE_SWEEPS", "1"))    # params mode: test one move (lam|u|rates|troot); the other theta
                                                      # components are redrawn exactly from their prior conditional   # full | params (theta moves only; latents+data resimulated
                                                      # exactly given theta) | latents (theta fixed at one prior draw;
                                                      # latent moves only; data redrawn given theta, latents)
NAMES = ["omega", "rho_mean", "lam0", "lam1", "lam2", "lam3", "t_root", "n_u", "sum_u2", "int_len", "leaf_len",
         "f_inherit", "f_replace", "f_borrow", "log_prior", "log_lik"]


def stats(f, theta, y, event, donor, leaf_forms, lp=np.nan, ll=np.nan):
    leaves = set(f.leaves)
    kinds = np.array(list(event.values()))
    return np.array([theta.omega, theta.rho.mean(), *theta.lam, theta.t_root,
                     sum(len(d) for d in theta.u.values()), sum(w * w for d in theta.u.values() for w in d.values()),
                     np.mean([len(v) for k, v in y.items() if k[0] not in leaves]), np.mean([len(v) for v in leaf_forms.values()]),
                     (kinds == ev.INHERIT).mean(), (kinds == ev.REPLACE).mean(), (kinds == ev.BORROW).mean(), lp, ll])


def redraw_data(s, rng):
    """Gibbs pass over the leaf forms given (theta, internal forms, events, donors)."""
    st, f = s.st, s.st.f
    for l in f.topo:
        if not st.is_leaf[l]:
            continue
        for c in range(st.M):
            in_mode, tab, x, lat = s._ffbs_setup(l, c)
            y, _ = ffbs_draw(rng, in_mode, (tab.log_sub_np, tab.log_ins_np) if tab is not None else None, x,
                             [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat])
            if y and len(y) <= 8:
                st.y[(l, c)] = y
    return {(l, c): st.y[(l, c)] for l in f.leaves for c in range(st.M)}


T_GRID = np.linspace(0.02, 12, 600)


def resample_others(th, keep, rng):
    """Exact draw of the theta components other than `keep` from their prior conditional given it
    (lam, rho, omega are independent; u_e | t_root on the root edges; t_root | u on the root edges)."""
    f = f0
    new = prior_theta(rng)
    R = f.roots[0]
    if keep == "u":
        new.u = {e: dict(d) for e, d in th.u.items()}
        lp = np.array([ev.log_prior_troot(t) + sum(ev.log_prior_u(th.u.get(e, {}), t) for e in f.children[R]) for t in T_GRID])
        w = np.exp(lp - lp.max()); w /= w.sum()
        new.t_root = float(T_GRID[rng.choice(len(T_GRID), p=w)])
    else:
        if keep == "troot":
            new.t_root = th.t_root
        elif keep == "lam":
            new.lam = th.lam.copy()
        elif keep == "rates":
            new.rho, new.omega = th.rho.copy(), th.omega
        new.u = {e: ev.sample_u(new.t_root if e in f.children[R] else f.length[e], rng) for e in f.edges}
    return new


def prior_theta(rng):
    """theta ~ prior with u on the root edges drawn at t_root (sample_theta uses the forest's lengths)."""
    th = sample_theta(f0, M, rng)
    th2 = sample_theta(with_root_length(f0, th.t_root), M, rng)
    th2.t_root = th.t_root
    return th2


def ess(x):
    """Geyer's initial positive sequence estimator (batch means with b = sqrt(n) badly overstated the ESS
    of the successive-conditional chain, whose autocorrelation time for form lengths is in the thousands)."""
    x = np.asarray(x, float); n = len(x)
    x = x - x.mean()
    v = x.var()
    if v == 0 or n < 20:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (v * n)
    tau = -1.0
    for k in range(0, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return float(min(n, n / max(tau, 1.0)))


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(seed)
    f0 = benchmark_a(related=True, t_root=1.0)
    # marginal-conditional
    mc = []
    th_fix = prior_theta(rng) if MODE == "latents" else None
    if th_fix is not None:
        th_fix.lam[3] = 7.0; th_fix.t_root = 1.0     # keep forms short: the sampler truncates at MAX_FORM_LEN
    for i in range(N):
        th = th_fix if th_fix is not None else prior_theta(rng)
        ds = simulate(with_root_length(f0, th.t_root), th, M, rng)
        t = ds.truth
        mc.append(stats(ds.forest, th, t["y"], t["event"], t["donor"], ds.leaf_forms))
    mc = np.array(mc)
    print(f"marginal-conditional done ({time.time() - t0:.0f}s)", flush=True)
    # successive-conditional
    th = th_fix if th_fix is not None else prior_theta(rng)
    ds = simulate(with_root_length(f0, th.t_root), th, M, rng)
    t = ds.truth
    leaves = set(f0.leaves)
    y = {k: v for k, v in t["y"].items() if k[0] not in leaves}; event, donor = dict(t["event"]), dict(t["donor"])
    sc = []; acc = {}
    for i in range(N):
        s = Sampler(Dataset(f0, M, ds.leaf_forms), th, rng, init=(y, event, donor))
        if ONLY:
            s.PARAM_MOVES = {ONLY}
        for _ in range(SWEEPS):
            s.sweep(clamp_latents=(MODE == "params"), clamp_params=(MODE == "latents"))
        st = s.st
        lp, ll = st.log_prior(), st.log_lik()
        y = {k: v for k, v in st.y.items() if k[0] not in leaves}; event, donor = dict(st.event), dict(st.donor); th = st.theta.copy()
        sc.append(stats(st.f, th, st.y, event, donor, {k: v for k, v in st.y.items() if k[0] in leaves}, lp, ll))
        for k, (a, b) in s.acc.items():
            acc[k] = [acc.get(k, [0, 0])[0] + a, acc.get(k, [0, 0])[1] + b]
        if MODE == "params":
            if ONLY:
                th = resample_others(th, ONLY, rng)
            ds = simulate(with_root_length(f0, th.t_root), th, M, rng)
            t = ds.truth
            y = {k: v for k, v in t["y"].items() if k[0] not in leaves}; event, donor = dict(t["event"]), dict(t["donor"])
        else:
            leaf_forms = redraw_data(s, rng)
            ds = Dataset(f0, M, leaf_forms)
    sc = np.array(sc)
    np.save(__import__("os").environ.get("MASSCOMP_GEWEKE_OUT", "/tmp/geweke_last.npy"), dict(mc=mc, sc=sc, names=NAMES), allow_pickle=True)
    print(f"successive-conditional done ({time.time() - t0:.0f}s)   moves: " + " ".join(f"{k}={a}/{b}" for k, (a, b) in acc.items() if b), flush=True)
    print(f"{'stat':10s} {'mc mean':>10s} {'sc mean':>10s} {'z':>7s} {'ess_sc':>7s}")
    for j, nm in enumerate(NAMES):
        if nm in ("log_prior", "log_lik"):
            continue
        a, b = mc[:, j], sc[:, j]
        e = ess(b)
        z = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / e)
        print(f"{nm:10s} {a.mean():10.3f} {b.mean():10.3f} {z:7.2f} {e:7.0f}")
    print(f"({time.time() - t0:.0f}s)")

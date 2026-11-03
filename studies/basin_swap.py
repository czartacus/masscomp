"""Where does the basin gap live? (root_diag, 3 Nov 2026: copy-start and truth-start chains settle ~130
nats apart in log posterior and neither likelihood-only nor full tempering crosses.)
Fit the copy-start and truth-start chains, take their end states, and score hybrids:
  * latents of one, theta of the other           -> is the gap carried by the parameters?
  * concept c of one spliced into the other      -> is it a sum of per-concept gaps (then a per-concept
                                                    move that can find the other mode would do), or do
                                                    the concepts only pay off together (coupled through
                                                    theta or through the borrow graph)?
Usage: PYTHONPATH=. python studies/basin_swap.py <seed> [sweeps] [burn]"""
import sys, time
from dataclasses import replace
import numpy as np
from masscomp.simulate import simulate_pair, sample_theta
from masscomp.mcmc import run_chain
from masscomp.likelihood import State
from masscomp.tree import with_root_length

seed = int(sys.argv[1]); SW = int(sys.argv[2]) if len(sys.argv) > 2 else 600; BURN = int(sys.argv[3]) if len(sys.argv) > 3 else 400
M, DEPTH = 25, 1.5
ds = simulate_pair(related=True, depth=DEPTH, M=M, seed=seed)
leaves = set(ds.forest.leaves)
internal = [v for v in range(ds.forest.n) if v not in leaves]


def fit(mode):
    rng = np.random.default_rng(seed * 100 + 7)
    if mode == "truth":
        t = ds.truth
        y = {k: v for k, v in t["y"].items() if k[0] not in leaves}
        s, _ = run_chain(ds, t["theta"].copy(), seed=seed * 100 + 60, n_sweeps=SW, burn=BURN, thin=5, record=False,
                         init=(y, dict(t["event"]), dict(t["donor"])))
    else:
        s, _ = run_chain(ds, sample_theta(ds.forest, ds.M, rng), seed=seed * 100 + 61, n_sweeps=SW, burn=BURN, thin=5, record=False)
    st = s.st
    lat = ({k: v for k, v in st.y.items() if k[0] not in leaves}, dict(st.event), dict(st.donor))
    return st.theta.copy(), lat


def score(theta, lat):
    d2 = replace(ds, forest=with_root_length(ds.forest, theta.t_root))
    st = State(d2, theta, *lat)
    return st.log_prior() + st.log_lik(), st.log_prior(), st.log_lik()


def splice(base, other, c):
    """base latents with concept c taken from other."""
    y, e, d = ({k: v for k, v in base[0].items() if k[1] != c}, {k: v for k, v in base[1].items() if k[1] != c},
               {k: v for k, v in base[2].items() if k[1] != c})
    y.update({k: v for k, v in other[0].items() if k[1] == c}); e.update({k: v for k, v in other[1].items() if k[1] == c})
    d.update({k: v for k, v in other[2].items() if k[1] == c})
    return y, e, d


if __name__ == "__main__":
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2) as ex:
        (th_c, lat_c), (th_t, lat_t) = list(ex.map(fit, ["copy", "truth"]))
    print(f"fits done ({time.time() - t0:.0f}s)", flush=True)
    Pc, Pt = score(th_c, lat_c), score(th_t, lat_t)
    print(f"copy  : post={Pc[0]:.1f} prior={Pc[1]:.1f} lik={Pc[2]:.1f}  omega={th_c.omega:.3f} rho={th_c.rho.mean():.3f} t_root={th_c.t_root:.2f} lam={np.round(th_c.lam, 2)}")
    print(f"truth : post={Pt[0]:.1f} prior={Pt[1]:.1f} lik={Pt[2]:.1f}  omega={th_t.omega:.3f} rho={th_t.rho.mean():.3f} t_root={th_t.t_root:.2f} lam={np.round(th_t.lam, 2)}")
    print(f"gap (truth - copy) = {Pt[0] - Pc[0]:.1f}")
    x = score(th_t, lat_c); y = score(th_c, lat_t)
    print(f"copy latents + truth theta : post={x[0]:.1f}  (vs copy {x[0] - Pc[0]:+.1f})")
    print(f"truth latents + copy theta : post={y[0]:.1f}  (vs truth {y[0] - Pt[0]:+.1f}, vs copy {y[0] - Pc[0]:+.1f})")
    dc, dt = [], []
    for c in range(M):
        dc.append(score(th_c, splice(lat_c, lat_t, c))[0] - Pc[0])   # truth concept into copy basin
        dt.append(score(th_t, splice(lat_t, lat_c, c))[0] - Pt[0])   # copy concept into truth basin
    dc, dt = np.array(dc), np.array(dt)
    print("per-concept: truth concept spliced into copy state (under copy theta), delta post:")
    print("   ", np.round(dc, 1), f" sum={dc.sum():.1f} n_improve={(dc > 0).sum()}")
    print("per-concept: copy concept spliced into truth state (under truth theta), delta post:")
    print("   ", np.round(dt, 1), f" sum={dt.sum():.1f} n_improve={(dt > 0).sum()}")
    # what do the two states disagree on, per concept
    ev_diff = [sum(lat_c[1][(e, c)] != lat_t[1][(e, c)] for e in ds.forest.edges) for c in range(M)]
    y_diff = [sum(lat_c[0][(v, c)] != lat_t[0][(v, c)] for v in internal) for c in range(M)]
    print("edges with different event per concept:", ev_diff)
    print("internal nodes with different form per concept:", y_diff, f"(of {len(internal)})")
    print(f"({time.time() - t0:.0f}s)")

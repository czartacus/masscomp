"""One diagnostic cell, for the `diag` workflow (one runner per cell) or local use.

Usage: python studies/diag.py "<spec>" <out.json>

<spec> is comma-separated key=value pairs. `kind` selects the experiment:

  kind=basin   root_diag.basin on one recovery dataset: seed, depth, modes (joined with '+', e.g.
               copy+truth), sweeps, burn, chains (3), M (25)
  kind=geweke  studies/geweke.py in a subprocess: n (iterations), m (concepts), seed, mode (full |
               params | latents), only (one param move), sweeps (sampler sweeps per iteration)
  kind=essrate mixing per CPU-second of one chain: seed, depth, M, start (truth | copy | prior), sweeps,
               burn; records the wall time per sweep and Geyer ESS per second of log_lik, log_prior,
               omega, lam, t_root, |u| and the mean internal form length, plus how often the joined
               root's forms change between sweeps. Compare across latent_moves / ffbs_p / regrow_p.

Sampler knobs go in the same spec and become MASSCOMP_* environment variables before the sampler is
imported: regrow_p, regrow_events, regrow_borrow, temper_p, ffbs_p, latent_moves, param_moves, procs,
or any key written env_<name> (-> MASSCOMP_<NAME>). List values are joined with '+' (latent_moves=
gibbs+ffbs+events). Everything in the spec is recorded in the output file.

Examples:
  kind=basin,seed=7,depth=0.25,modes=copy+truth,sweeps=1500,burn=500,regrow_p=0.1,regrow_events=1,regrow_borrow=1
  kind=geweke,n=10000,m=2,seed=1,mode=latents,sweeps=2,regrow_p=0.3,latent_moves=regrow
"""
import sys, os, json, time, subprocess, re
import numpy as np

KNOBS = {"regrow_p": "MASSCOMP_REGROW_P", "regrow_events": "MASSCOMP_REGROW_EVENTS", "regrow_borrow": "MASSCOMP_REGROW_BORROW",
         "temper_p": "MASSCOMP_TEMPER_P", "latent_moves": "MASSCOMP_LATENT_MOVES", "param_moves": "MASSCOMP_PARAM_MOVES",
         "procs": "MASSCOMP_PROCS", "ffbs_p": "MASSCOMP_FFBS_P", "init": "MASSCOMP_INIT",
         "root_regrow_p": "MASSCOMP_ROOT_REGROW_P", "max_form_len": "MASSCOMP_MAX_FORM_LEN", "geweke_omega": "MASSCOMP_GEWEKE_OMEGA", "geweke_restart": "MASSCOMP_GEWEKE_RESTART",
         "g": "MASSCOMP_G", "inventory": "MASSCOMP_INVENTORY", "lam_sd": "MASSCOMP_LAM_SD",
         "u_context": "MASSCOMP_U_CONTEXT", "u_rules": "MASSCOMP_U_RULES", "u_per_t": "MASSCOMP_U_PER_T"}


def parse(spec):
    params = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.partition("=")
        params[k.strip()] = v.strip()
    return params


def set_env(params):
    env = {}
    for k, v in params.items():
        if k in KNOBS:
            env[KNOBS[k]] = v.replace("+", ",")        # list-valued knobs are written a+b+c in a spec
        elif k.startswith("env_"):
            env["MASSCOMP_" + k[4:].upper()] = v
    os.environ.update(env)
    return env


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def run_basin(p):
    from studies.root_diag import basin
    modes = p.get("modes", "copy+truth").split("+")
    n_procs = int(os.environ.get("MASSCOMP_PROCS", "1"))
    return basin(int(p.get("seed", 0)), float(p.get("depth", 1.5)), modes, int(p.get("sweeps", 1500)), int(p.get("burn", 500)),
                 M=int(p.get("M", 25)), n_chains=int(p.get("chains", 3)), n_procs=n_procs)


def _ess(x):
    x = np.asarray(x, float); n = len(x); x = x - x.mean(); v = x.var()
    if v == 0 or n < 20:
        return float(n)
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n] / (v * n)
    tau = -1.0
    for k in range(0, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return float(min(n, n / max(tau, 1.0)))


def run_essrate(p):
    from masscomp.simulate import simulate_pair, sample_theta
    from masscomp.mcmc import run_chain, Sampler
    seed = int(p.get("seed", 7)); depth = float(p.get("depth", 0.25)); M = int(p.get("M", 25))
    SW = int(p.get("sweeps", 600)); BURN = int(p.get("burn", 100)); start = p.get("start", "truth")
    ds = simulate_pair(related=True, depth=depth, M=M, seed=seed)
    rng = np.random.default_rng(seed * 100 + 9)
    kw = {}
    if start == "truth":
        t = ds.truth; leaves = set(ds.forest.leaves)
        th0 = t["theta"].copy(); kw["init"] = ({k: v for k, v in t["y"].items() if k[0] not in leaves}, dict(t["event"]), dict(t["donor"]))
    else:
        th0 = sample_theta(ds.forest, M, rng)
        if start == "prior":
            kw["init_from_prior"] = True
    t0 = time.time()
    s, out = run_chain(ds, th0, seed=seed * 100 + 59, n_sweeps=SW, burn=BURN, thin=1, **kw)
    secs = time.time() - t0
    series = dict(log_lik=[o["log_lik"] for o in out], log_prior=[o["log_prior"] for o in out], omega=[o["omega"] for o in out],
                  t_root=[o["t_root"] for o in out], n_u=[sum(len(d) for d in o["u"].values()) for o in out])
    for i in range(4):
        series[f"lam{i}"] = [o["lam"][i] for o in out]
    R = ds.forest.roots[0]
    series["root_len"] = [np.mean([len(v) for (vv, c), v in o["roots"].items() if vv == R]) for o in out]
    changes = [np.mean([o["roots"][k] != prev["roots"][k] for k in o["roots"] if k[0] == R]) for prev, o in zip(out, out[1:])]
    ess = {k: _ess(v) for k, v in series.items()}
    res = dict(secs=secs, secs_per_sweep=secs / SW, n_recorded=len(out), ess=ess, ess_per_sec={k: v / secs for k, v in ess.items()},
               root_change_rate=float(np.mean(changes)) if changes else None,
               means={k: float(np.mean(v)) for k, v in series.items()}, acc={k: list(map(int, v)) for k, v in s.acc.items()},
               truth=dict(omega=float(ds.truth["theta"].omega), t_root=float(ds.truth["theta"].t_root), lam=[float(x) for x in ds.truth["theta"].lam]))
    return res


def run_geweke(p):
    args = [sys.executable, "studies/geweke.py", p.get("n", "2000"), p.get("m", "6"), p.get("seed", "1"), p.get("mode", "full")]
    if p.get("only"):
        args.append(p["only"])
    env = dict(os.environ, MASSCOMP_GEWEKE_SWEEPS=p.get("sweeps", "1"), PYTHONPATH=os.environ.get("PYTHONPATH", "."))
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    out = r.stdout
    stats = []
    for line in out.splitlines():
        m = re.match(r"^(\w+)\s+(-?[\d.]+(?:e[-+]?\d+)?|nan)\s+(-?[\d.]+(?:e[-+]?\d+)?|nan)\s+(-?[\d.]+|nan)\s+(\d+)\s*$", line.strip())
        if m:
            stats.append(dict(stat=m.group(1), mc=float(m.group(2)), sc=float(m.group(3)), z=float(m.group(4)), ess=int(m.group(5))))
    moves = next((l for l in out.splitlines() if "moves:" in l), None)
    zs = [abs(s["z"]) for s in stats if s["ess"] >= 20 and s["z"] == s["z"]]
    return dict(stats=stats, moves=moves, max_abs_z=max(zs) if zs else None, returncode=r.returncode,
                stdout=out[-20000:], stderr=r.stderr[-5000:])


if __name__ == "__main__":
    spec, out_path = sys.argv[1], sys.argv[2]
    p = parse(spec)
    env = set_env(p)
    t0 = time.time()
    kind = p.get("kind", "basin")
    if kind == "basin":
        result = run_basin(p)
    elif kind == "geweke":
        result = run_geweke(p)
    elif kind == "essrate":
        result = run_essrate(p)
    elif kind == "realdata":
        from studies.realdata_probe import probe, summary_lines
        result = probe(p); result["summary"] = summary_lines(result)
    elif kind == "realdata_lbf":
        from studies.realdata_lbf import lbf, summary_line
        result = lbf(p); result["summary"] = summary_line(result)
    else:
        raise SystemExit(f"unknown kind {kind}")
    rec = dict(study="diag", kind=kind, spec=spec, params=p, env=env, git=git_sha(), secs=time.time() - t0, result=result)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "result"}, indent=1))
    if kind == "basin":
        for m, s in result.items():
            print(f"{m:6s} root cov R={s['root_cov_R']:.2f} fam={s['root_cov_fam']:.2f} ll={s['mean_ll']:.1f} omega x{s['omega_ratio']:.2f} t_root={s['t_root']:.2f} ({s['secs']:.0f}s)")
    elif kind == "geweke":
        print(result.get("moves")); print("max |z| (ess>=20):", result.get("max_abs_z"))
    elif kind == "realdata":
        print("\n".join(result["summary"]))
    elif kind == "realdata_lbf":
        print(result["summary"])
    else:
        print(f"{result['secs_per_sweep']*1000:.0f} ms/sweep; ESS/s:", {k: round(v, 3) for k, v in result["ess_per_sec"].items()}, "root change rate", result["root_change_rate"])

"""One diagnostic cell, for the `diag` workflow (one runner per cell) or local use.

Usage: python studies/diag.py "<spec>" <out.json>

<spec> is comma-separated key=value pairs. `kind` selects the experiment:

  kind=basin   root_diag.basin on one recovery dataset: seed, depth, modes (joined with '+', e.g.
               copy+truth), sweeps, burn, chains (3), M (25)
  kind=geweke  studies/geweke.py in a subprocess: n (iterations), m (concepts), seed, mode (full |
               params | latents), only (one param move), sweeps (sampler sweeps per iteration)

Sampler knobs go in the same spec and become MASSCOMP_* environment variables before the sampler is
imported: regrow_p, regrow_events, regrow_borrow, temper_p, latent_moves, param_moves, procs, or any
key written env_<name> (-> MASSCOMP_<NAME>). Everything in the spec is recorded in the output file.

Examples:
  kind=basin,seed=7,depth=0.25,modes=copy+truth,sweeps=1500,burn=500,regrow_p=0.1,regrow_events=1,regrow_borrow=1
  kind=geweke,n=10000,m=2,seed=1,mode=latents,sweeps=2,regrow_p=0.3,latent_moves=regrow
"""
import sys, os, json, time, subprocess, re

KNOBS = {"regrow_p": "MASSCOMP_REGROW_P", "regrow_events": "MASSCOMP_REGROW_EVENTS", "regrow_borrow": "MASSCOMP_REGROW_BORROW",
         "temper_p": "MASSCOMP_TEMPER_P", "latent_moves": "MASSCOMP_LATENT_MOVES", "param_moves": "MASSCOMP_PARAM_MOVES",
         "procs": "MASSCOMP_PROCS"}


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
            env[KNOBS[k]] = v
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
    else:
        raise SystemExit(f"unknown kind {kind}")
    rec = dict(study="diag", kind=kind, spec=spec, params=p, env=env, git=git_sha(), secs=time.time() - t0, result=result)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(rec, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "result"}, indent=1))
    if kind == "basin":
        for m, s in result.items():
            print(f"{m:6s} root cov R={s['root_cov_R']:.2f} fam={s['root_cov_fam']:.2f} ll={s['mean_ll']:.1f} omega x{s['omega_ratio']:.2f} t_root={s['t_root']:.2f} ({s['secs']:.0f}s)")
    else:
        print(result.get("moves")); print("max |z| (ess>=20):", result.get("max_abs_z"))

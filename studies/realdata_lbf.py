"""Join-path LBF (marglik.join_path) on real family pairs from NorthEuraLex: the tier-1 question, "does the
model separate pairs known to be related from pairs known not to be?", asked of the prototype with the
caveats of README "Real-data groundwork". Families are fixed 3-leaf binary trees (FAMILIES); a pair is
joined at a root whose edges carry t_root (prior Gamma(2, 1)); n random concepts with a single-word form
in all six languages; G flat or fitted to the six languages' other concepts.
Usage: PYTHONPATH=. MASSCOMP_INVENTORY=extended python studies/realdata_lbf.py fam1=ibero,fam2=finnic,n=50,...
  keys: fam1, fam2, n (50), seed (1), K (16), sweeps (600), burn (200), warmup (800), power (2), reps (1),
        g (flat|fit), len_edge (0.3), thin (4), init (copy|prior)
Also a diag cell: kind=realdata_lbf,fam1=...,fam2=... (studies/diag.py)."""
import sys, os, time, json
import numpy as np
from masscomp import events as ev
from masscomp.segments import N_SEG
from masscomp.simulate import Dataset, sample_theta
from masscomp.marglik import join_path
from masscomp.data import load_cldf
from studies.common import N_PROCS
from studies.realdata_probe import build_forest, leaf_langs

FAMILIES = {"ibero": (("spa", "por"), "cat"), "italo": (("ita", "fra"), "ron"), "germanic": (("deu", "nld"), "eng"),
            "nordic": (("swe", "dan"), "nor"), "slavic": (("rus", "ukr"), "pol"), "finnic": (("fin", "krl"), "ekk"),
            "saami": (("sme", "smn"), "sma"), "turkic": (("tur", "azj"), "kaz"), "mongolic": (("khk", "bua"), "xal")}
# what is known (for the report only; the model never sees it)
RELATED = {frozenset(p) for p in [("ibero", "italo"), ("germanic", "nordic"), ("finnic", "saami"),
                                  ("ibero", "germanic"), ("ibero", "nordic"), ("ibero", "slavic"), ("italo", "germanic"), ("italo", "nordic"),
                                  ("italo", "slavic"), ("germanic", "slavic"), ("nordic", "slavic")]}
DEEP = {frozenset(p) for p in [("ibero", "germanic"), ("ibero", "nordic"), ("ibero", "slavic"), ("italo", "germanic"), ("italo", "nordic"),
                               ("italo", "slavic"), ("germanic", "slavic"), ("nordic", "slavic")]}


def _estimate(args):
    ds, th, seed, K, sweeps, burn, thin, warmup, power, init_prior, g_tables = args
    ev.set_G(*g_tables)
    t0 = time.time()
    trace = {}
    val, terms, s = join_path(ds, th, seed, K=K, n_sweeps=sweeps, burn=burn, thin=thin, warmup=warmup, power=power, init_from_prior=init_prior,
                              trace=trace)
    from masscomp.data import to_str
    for key in ("root_forms_at1", "fam_forms_at1"):
        trace[key] = {k: [[to_str(form), round(p_, 3)] for form, p_ in v] for k, v in trace.get(key, {}).items()}
    return dict(lbf=float(val), terms=[float(t) for t in terms], secs=time.time() - t0, t_root_end=float(s.st.theta.t_root),
                omega_end=float(s.st.theta.omega), lam_end=[round(float(x), 2) for x in s.st.theta.lam], trace=trace)


def lbf(p):
    data = p.get("data", "data/northeuralex_sel.csv"); n = int(p.get("n", 50)); seed = int(p.get("seed", 1))
    f1, f2 = p["fam1"], p["fam2"]
    K = int(p.get("K", 16)); sweeps = int(p.get("sweeps", 600)); burn = int(p.get("burn", 200)); warmup = int(p.get("warmup", 800))
    power = float(p.get("power", 2.0)); reps = int(p.get("reps", 1)); g_mode = p.get("g", "flat"); len_edge = float(p.get("len_edge", 0.3))
    thin = int(p.get("thin", 4)); init_prior = p.get("init", "copy") == "prior"
    families = [(f1, FAMILIES[f1]), (f2, FAMILIES[f2])]
    langs = leaf_langs(families)
    forms_by, common, report = load_cldf(data, langs)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(common, size=min(n, len(common)), replace=False).tolist())
    rest = [c for c in common if c not in set(chosen)]
    if g_mode == "fit":
        ev.fit_G([forms_by[L][c] for L in langs for c in rest])
    else:
        ev.reset_G()
    f = build_forest(families, len_edge)
    name2v = {v: k for k, v in f.names.items()}
    M = len(chosen)
    ds = Dataset(f, M, {(name2v[L], ci): forms_by[L][c] for L in langs for ci, c in enumerate(chosen)})
    jobs = []
    for r in range(reps):
        th = sample_theta(f, M, np.random.default_rng(seed * 100 + r))
        jobs.append((ds, th, seed * 1000 + r, K, sweeps, burn, thin, warmup, power, init_prior, (ev.G_LEN_RAW.copy(), ev.G_SEG.copy())))
    t0 = time.time()
    if N_PROCS > 1 and reps > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(N_PROCS, reps)) as ex:
            ests = list(ex.map(_estimate, jobs))
    else:
        ests = [_estimate(j) for j in jobs]
    vals = [e["lbf"] for e in ests]
    key = frozenset((f1, f2))
    from masscomp.data import to_str
    tree = dict(parent=list(f.parent), length=[float(x) for x in f.length], family=list(f.family), leaves=list(f.leaves),
                names={str(k): v for k, v in f.names.items()})
    leaves = {L: {c: to_str(forms_by[L][c]) for c in chosen} for L in langs}
    return dict(concepts=chosen, tree=tree, leaf_forms=leaves, config=dict(fam1=f1, fam2=f2, families=families, n=M, seed=seed, K=K, sweeps=sweeps, burn=burn, warmup=warmup, power=power,
                            reps=reps, g=g_mode, len_edge=len_edge, init="prior" if init_prior else "copy", inventory_size=N_SEG,
                            lam_sd=ev.LAM_SD, u_context=ev.U_CONTEXT, u_rules=ev.U_RULES, u_per_t=ev.U_PER_T),
                known="related" if key in RELATED else "unrelated", deep=key in DEEP, common_concepts=len(common),
                lbf_mean=float(np.mean(vals)), lbf_sd=float(np.std(vals)) if reps > 1 else None, estimates=ests, secs=time.time() - t0)


def summary_line(r):
    c = r["config"]
    return (f"{c['fam1']}-{c['fam2']} ({r['known']}{', deep' if r['deep'] else ''}) G={c['g']} n={c['n']} K={c['K']}x{c['sweeps']}: "
            f"LBF {r['lbf_mean']:+.1f}" + (f" sd {r['lbf_sd']:.1f}" if r['lbf_sd'] is not None else "") +
            f"  [{', '.join(f'{e['lbf']:+.1f}' for e in r['estimates'])}] t_root_end {[round(e['t_root_end'], 2) for e in r['estimates']]} ({r['secs']:.0f}s)")


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else ""
    p = dict(kv.split("=", 1) for kv in spec.split(",") if kv)
    r = lbf(p)
    os.makedirs("results/realdata", exist_ok=True)
    tag = p.get("tag", f"lbf_{p['fam1']}_{p['fam2']}_{p.get('g', 'flat')}_{p.get('seed', 1)}")
    json.dump(r, open(f"results/realdata/{tag}.json", "w"), indent=1)
    print(summary_line(r))

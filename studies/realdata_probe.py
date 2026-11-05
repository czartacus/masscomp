"""Real-data reconstruction probe (README, "Real-data groundwork"): a known family on a fixed tree, the
attested proto-language held out, and the question of whether the model reconstructs it — never a
claim, a way to find what the prototype gets wrong on real words.

Default configuration: six Romance languages from NorthEuraLex on a two-family tree (Ibero-Romance
spa/por/cat; ita/fra/ron, star within each family, daughter edges 0.3), Latin held out, n random
concepts (seeded), the innovation prior G either the plan's flat one or fitted (events.fit_G) to the
daughters' forms for the concepts NOT in the probe, two chains, extended inventory expected
(MASSCOMP_INVENTORY=extended). Reports per chain the posterior means of t_root, omega, lam and the
event fractions on the root edges; per concept the Latin form, the Rao-Blackwellised top root
forms (studies.common.node_conditional over a subsample of states), log p(Latin) under that RB
marginal, and the same for the family roots; and the root-length / Latin-length ratio.

Usage (CLI):  PYTHONPATH=. MASSCOMP_INVENTORY=extended python studies/realdata_probe.py key=value,...
   keys: data (csv or CLDF dir; default data/northeuralex_sel.csv), n (50), seed (1), sweeps (1500),
         burn (500), chains (2), g (flat|fit), len_edge (0.3), tag (output name), thin (5), n_states (40)
Also a diag cell: kind=realdata,n=50,seed=1,g=fit,... (studies/diag.py)."""
import sys, os, time, math, json
import numpy as np
from masscomp import events as ev
from masscomp.segments import N_SEG
from masscomp.tree import Forest
from masscomp.simulate import Theta, Dataset, sample_theta
from masscomp.mcmc import run_chain, Sampler
from masscomp.data import load_cldf, to_str
from studies.common import node_conditional, _lse_list, N_PROCS

PROTO = "lat"
# binary trees (the lattice marginalises at most two inheriting out-edges per node): ((spa, por), cat), ((ita, fra), ron)
FAMILIES = [("ibero", (("spa", "por"), "cat")), ("italo", (("ita", "fra"), "ron"))]


def build_forest(families, len_edge, donors=()):
    """The joined families under one root R (family roots are R's children); every leaf sits at depth
    len_edge below its family root (a nested tuple is an internal node; internal edges split the budget
    in half). `donors` are families that are never joined: each is a separate tree of the forest, built
    first so that its nodes precede the joined families in the breadth-first order that
    Forest.donor_candidates uses, and so can be borrowed from at every level, the family roots included.
    A donor given as a bare language name is a root with that one leaf."""
    parent, length, family, leaves, names = [], [], [], [], {}

    def add(sub, par, fi, budget):
        v = len(parent)
        if isinstance(sub, tuple):
            parent.append(par); length.append(budget / 2); family.append(fi); names[v] = f"{names[par]}/{len(names)}"
            for s in sub:
                add(s, v, fi, budget / 2)
        else:
            parent.append(par); length.append(budget); family.append(fi); leaves.append(v); names[v] = sub

    def subtree(sub):
        return sub if isinstance(sub, tuple) else (sub,)

    fi = 0
    for dname, sub in donors:                       # each donor: its root, then its subtree
        r = len(parent); parent.append(-1); length.append(0.0); family.append(fi); names[r] = dname
        for s in subtree(sub):
            add(s, r, fi, len_edge)
        fi += 1
    R = len(parent)                                 # the joined root, then the family roots 1..F below it, then the subtrees
    parent.append(-1); length.append(0.0); family.append(-1)
    first = fi
    for fname, _ in families:
        r = len(parent); parent.append(R); length.append(1.0); family.append(fi); names[r] = fname; fi += 1
    for k, (fname, sub) in enumerate(families):
        for s in subtree(sub):
            add(s, R + 1 + k, first + k, len_edge)
    return Forest(parent, length, family, leaves, names=names)


def leaf_langs(families):
    out = []

    def walk(sub):
        if isinstance(sub, tuple):
            for s in sub:
                walk(s)
        else:
            out.append(sub)
    for _, sub in families:
        walk(sub)
    return out


def edit_distance(a, b):
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[m]


def rb_marginals(ds, chain, nodes, n_states, seed):
    """RB marginal over the forms seen in the chain at each (node, concept): p(f) = mean_s p_s(f)."""
    samples = [o for o in chain if "latents" in o]
    rng = np.random.default_rng(seed)
    sub = [samples[i] for i in sorted(rng.choice(len(samples), size=min(n_states, len(samples)), replace=False))]
    keys = [(v, c) for v in nodes for c in range(ds.M)]
    forms = {k: sorted({o["roots"][k] for o in samples}) for k in keys}
    acc = {k: np.full(len(forms[k]), -np.inf) for k in keys}; n_used = {k: 0 for k in keys}
    for o in sub:
        th = Theta(np.array(o["lam"]), {e: dict(d) for e, d in o["u"].items()}, np.array(o["rho"]), float(o["omega"]), float(o["t_root"]))
        y, event, donor = o["latents"]
        s = Sampler(ds, th, np.random.default_rng(0), init=(dict(y), dict(event), dict(donor)))
        for k in keys:
            lp = node_conditional(s, k[0], k[1])
            if lp is None:
                continue
            acc[k] = np.logaddexp(acc[k], np.array([lp(f) for f in forms[k]])); n_used[k] += 1
    out = {}
    for k in keys:
        if n_used[k] == 0:
            continue
        lp = acc[k] - math.log(n_used[k])
        out[k] = (forms[k], lp, n_used[k])
    return out


def rb_logp(ds, chain, k, form, n_states, seed):
    """log of the RB marginal probability of one particular form at (node, concept)."""
    samples = [o for o in chain if "latents" in o]
    rng = np.random.default_rng(seed)
    sub = [samples[i] for i in sorted(rng.choice(len(samples), size=min(n_states, len(samples)), replace=False))]
    vals = []
    for o in sub:
        th = Theta(np.array(o["lam"]), {e: dict(d) for e, d in o["u"].items()}, np.array(o["rho"]), float(o["omega"]), float(o["t_root"]))
        y, event, donor = o["latents"]
        s = Sampler(ds, th, np.random.default_rng(0), init=(dict(y), dict(event), dict(donor)))
        lp = node_conditional(s, k[0], k[1])
        if lp is not None:
            vals.append(lp(form))
    return (_lse_list(vals) - math.log(len(vals))) if vals else float("nan")


def _chain(args):
    ds, th0, seed, sweeps, burn, thin, g_tables = args
    ev.set_G(*g_tables)                 # a pool worker is a fresh process with the flat default
    t0 = time.time()
    s, out = run_chain(ds, th0, seed=seed, n_sweeps=sweeps, burn=burn, thin=thin, record_latents=True)
    return out, time.time() - t0, {k: list(map(int, v)) for k, v in s.acc.items()}


def probe(p):
    data = p.get("data", "data/northeuralex_sel.csv"); n = int(p.get("n", 50)); seed = int(p.get("seed", 1))
    sweeps = int(p.get("sweeps", 1500)); burn = int(p.get("burn", 500)); n_chains = int(p.get("chains", 2))
    g_mode = p.get("g", "flat"); len_edge = float(p.get("len_edge", 0.3)); thin = int(p.get("thin", 5)); n_states = int(p.get("n_states", 40))
    families = FAMILIES
    langs = leaf_langs(families)
    forms_by, common, report = load_cldf(data, [PROTO] + langs)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(common, size=min(n, len(common)), replace=False).tolist())
    rest = [c for c in common if c not in set(chosen)]
    if g_mode == "fit":
        ev.fit_G([forms_by[L][c] for L in langs for c in rest])
    elif g_mode.startswith("random:"):
        ev.random_G(int(g_mode.split(":")[1]))
    else:
        ev.reset_G()
    g_desc = dict(mode=g_mode, len_p={L: round(float(v), 4) for L, v in enumerate(ev.G_LEN_P) if v > 0},
                  seg_p=[round(float(v), 4) for v in ev.G_SEG_P], fit_forms=len(rest) * len(langs) if g_mode == "fit" else 0)
    f = build_forest(families, len_edge)
    name2v = {v: k for k, v in f.names.items()}
    M = len(chosen)
    leaf_forms = {(name2v[L], ci): forms_by[L][c] for L in langs for ci, c in enumerate(chosen)}
    ds = Dataset(f, M, leaf_forms)
    latin = {ci: forms_by[PROTO][c] for ci, c in enumerate(chosen)}
    jobs = []
    for ch in range(n_chains):
        th0 = sample_theta(f, M, np.random.default_rng(seed * 100 + ch))
        jobs.append((ds, th0, seed * 1000 + ch, sweeps, burn, thin, (ev.G_LEN_RAW.copy(), ev.G_SEG.copy())))
    t0 = time.time()
    if N_PROCS > 1 and n_chains > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(N_PROCS, n_chains)) as ex:
            runs = list(ex.map(_chain, jobs))
    else:
        runs = [_chain(j) for j in jobs]
    chains = [r[0] for r in runs]
    R = f.join_root; fam_roots = list(f.children[R])
    per_chain = []
    for out, secs, acc in runs:
        lam = np.array([o["lam"] for o in out]); kinds = {}
        for e in fam_roots + f.leaves:
            ks = [o["latents"][1][(e, c)] for o in out for c in range(M)]
            kinds[f.names.get(e, str(e))] = dict(inherit=float(np.mean([k == ev.INHERIT for k in ks])), replace=float(np.mean([k == ev.REPLACE for k in ks])),
                                                 borrow=float(np.mean([k == ev.BORROW for k in ks])))
        per_chain.append(dict(secs=secs, t_root=float(np.mean([o["t_root"] for o in out])), omega=float(np.mean([o["omega"] for o in out])),
                              lam=[round(float(x), 2) for x in lam.mean(0)], lam_sd=[round(float(x), 2) for x in lam.std(0)],
                              n_u=float(np.mean([sum(len(d) for d in o["u"].values()) for o in out])),
                              root_len=float(np.mean([len(o["roots"][(R, c)]) for o in out for c in range(M)])),
                              fam_root_len=float(np.mean([len(o["roots"][(v, c)]) for o in out for v in fam_roots for c in range(M)])),
                              events=kinds, log_lik=float(np.mean([o["log_lik"] for o in out]))))
    # RB reconstructions from the first chain
    marg = rb_marginals(ds, chains[0], [R] + fam_roots, n_states, seed)
    concepts = []
    for ci, c in enumerate(chosen):
        rec = dict(concept=c, latin=to_str(latin[ci]), latin_len=len(latin[ci]))
        for v, label in [(R, "root")] + [(v, f.names[v]) for v in fam_roots]:
            if (v, ci) not in marg:
                continue
            forms, lp, nu = marg[(v, ci)]
            order = np.argsort(-lp)[:3]
            lpl = rb_logp(ds, chains[0], (v, ci), latin[ci], n_states, seed)
            mode = forms[order[0]]
            rec[label] = dict(top=[(to_str(forms[i]), round(float(math.exp(lp[i])), 3)) for i in order], logp_latin=round(float(lpl), 2),
                              logp_mode=round(float(lp[order[0]]), 2), mode_len=len(mode),
                              ned=round(edit_distance(mode, latin[ci]) / max(len(mode), len(latin[ci])), 3))
        rec["leaves"] = {L: to_str(forms_by[L][c]) for L in langs}
        # the daughters' own distance to Latin, as the yardstick a reconstruction has to beat
        rec["leaf_ned"] = round(float(np.mean([edit_distance(forms_by[L][c], latin[ci]) / max(len(forms_by[L][c]), len(latin[ci])) for L in langs])), 3)
        concepts.append(rec)
    latin_len = float(np.mean([len(v) for v in latin.values()]))
    hit1 = float(np.mean([r["root"]["top"][0][0] == r["latin"] for r in concepts if "root" in r])) if concepts else float("nan")
    ned = {label: float(np.mean([r[label]["ned"] for r in concepts if label in r])) for label in ["root"] + [f.names[v] for v in fam_roots]}
    ned["leaves"] = float(np.mean([r["leaf_ned"] for r in concepts]))
    # the NorthEuraLex Latin form is often not the etymon of the Romance words (aridus / siccus, imber /
    # pluvia); the "cognate" subset keeps the concepts whose daughters are within 0.6 of the Latin form
    cog = [r for r in concepts if r["leaf_ned"] < 0.6]
    ned_cog = {label: float(np.mean([r[label]["ned"] for r in cog if label in r])) for label in ["root"] + [f.names[v] for v in fam_roots]} if cog else {}
    if cog:
        ned_cog["leaves"] = float(np.mean([r["leaf_ned"] for r in cog])); ned_cog["n"] = len(cog)
    res = dict(config=dict(data=data, n=M, seed=seed, sweeps=sweeps, burn=burn, chains=n_chains, len_edge=len_edge, families=families, proto=PROTO,
                           inventory_size=N_SEG, lam_sd=ev.LAM_SD, lam_mu=[float(x) for x in ev.LAM_MU]),
               G=g_desc, report=dict(common_concepts=len(common), kept=report["kept"]), per_chain=per_chain,
               latin_len=latin_len, root_len_ratio=per_chain[0]["root_len"] / latin_len, fam_root_len_ratio=per_chain[0]["fam_root_len"] / latin_len,
               root_mode_is_latin=hit1, ned=ned, ned_cognate=ned_cog, mean_logp_latin_root=float(np.mean([r["root"]["logp_latin"] for r in concepts if "root" in r])),
               mean_logp_mode_root=float(np.mean([r["root"]["logp_mode"] for r in concepts if "root" in r])),
               concepts=concepts, secs=time.time() - t0)
    return res


def summary_lines(res):
    L = []
    g = res["G"]; c = res["config"]
    L.append(f"G={g['mode']} n={c['n']} seed={c['seed']} sweeps={c['sweeps']}x{c['chains']} lam_sd={c['lam_sd']} inventory={c['inventory_size']} ({res['secs']:.0f}s)")
    for i, pc in enumerate(res["per_chain"]):
        evs = " ".join(f"{k}:{v['inherit']:.2f}/{v['replace']:.2f}/{v['borrow']:.2f}" for k, v in pc["events"].items())
        L.append(f"chain {i}: t_root {pc['t_root']:.2f} omega {pc['omega']:.3f} lam {pc['lam']} n_u {pc['n_u']:.0f} root_len {pc['root_len']:.2f} "
                 f"fam_root_len {pc['fam_root_len']:.2f} ll {pc['log_lik']:.0f} ({pc['secs']:.0f}s) events inherit/replace/borrow {evs}")
    L.append(f"latin_len {res['latin_len']:.2f}; root/latin {res['root_len_ratio']:.2f}; fam/latin {res['fam_root_len_ratio']:.2f}; "
             f"root mode == latin {res['root_mode_is_latin']:.2f}; mean log p(latin | root) {res['mean_logp_latin_root']:.1f} vs mode {res['mean_logp_mode_root']:.1f}")
    L.append("normalised edit distance of the modal form to Latin: " + ", ".join(f"{k} {v:.3f}" for k, v in res["ned"].items()))
    if res["ned_cognate"]:
        L.append("  same on the concepts whose daughters are within 0.6 of Latin: " + ", ".join(f"{k} {v:.3f}" if k != "n" else f"n {v}" for k, v in res["ned_cognate"].items()))
    for r in res["concepts"]:
        L.append(f"  {r['concept']:<16} {r['latin']:<10} root {r['root']['top']} lp(lat) {r['root']['logp_latin']} ned {r['root']['ned']} | "
                 f"ibero {r.get('ibero', {}).get('top', [])[:2]} | italo {r.get('italo', {}).get('top', [])[:2]} | leaves ned {r['leaf_ned']}")
    return L


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else ""
    p = dict(kv.split("=", 1) for kv in spec.split(",") if kv)
    res = probe(p)
    tag = p.get("tag", f"realdata_{p.get('g', 'flat')}_{p.get('seed', 1)}")
    os.makedirs("results/realdata", exist_ok=True)
    json.dump(res, open(f"results/realdata/{tag}.json", "w"), indent=1)
    lines = summary_lines(res)
    open(f"results/realdata/{tag}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

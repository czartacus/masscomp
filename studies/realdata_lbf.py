"""Join-path LBF (marglik.join_path) on real family pairs from NorthEuraLex: the tier-1 question, "does the
model separate pairs known to be related from pairs known not to be?", asked of the prototype with the
caveats of README "Real-data groundwork". Families are fixed 3-leaf binary trees (FAMILIES); a pair is
joined at a root whose edges carry t_root (prior Gamma(2, 1)); n random concepts with a single-word form
in all six languages; G flat or fitted to the six languages' other concepts.
Usage: PYTHONPATH=. MASSCOMP_INVENTORY=extended python studies/realdata_lbf.py fam1=ibero,fam2=finnic,n=50,...
  keys: fam1, fam2, n (50), seed (1), K (16), sweeps (600), burn (200), warmup (800), power (2), reps (1),
        g (flat|fit), len_edge (0.3), thin (4), init (copy|prior), concepts (all|basic: Swadesh-100 + Leipzig-Jakarta),
        donors (none, or DONORS names joined by +: separate trees that are never joined but can be borrowed from),
        spacing (top|both: where the ladder is dense; "both" keeps the bottom stone's power variable in range), power (2),
        direction (down|up|alt: cooled from alpha = 1 or heated from 0; alt = replicates alternate, so a pair brackets
        the hysteresis of the transition region), ess_min (0: a stone with fewer effective samples is sampled up to
        twice as long again), boost (1: sweeps multiplier for the stones with alpha in 0.7-0.98, the transition),
        estimator (ss: one chain walked along the ladder | re: replica exchange, all stones as coupled chains; the
        remedy for the ladder's hysteresis, README "The hysteresis is a first-order transition")
Also a diag cell: kind=realdata_lbf,fam1=...,fam2=... (studies/diag.py)."""
import sys, os, time, json
import numpy as np
from masscomp import events as ev
from masscomp.segments import N_SEG
from masscomp.transducer import RATE_TIME, T_REF, DIRECTIONAL
from masscomp.simulate import Dataset, sample_theta
from masscomp.marglik import join_path, join_path_re
from masscomp.data import load_cldf
from studies.common import N_PROCS
from studies.realdata_probe import build_forest, leaf_langs, require_extended_inventory

FAMILIES = {"ibero": (("spa", "por"), "cat"), "italo": (("ita", "fra"), "ron"), "germanic": (("deu", "nld"), "eng"),
            "romlat": ((("spa", "por"), "cat"), "lat"),          # Ibero-Romance with Latin as an attested older stage in the tree
            "nordic": (("swe", "dan"), "nor"), "slavic": (("rus", "ukr"), "pol"), "finnic": (("fin", "krl"), "ekk"),
            "saami": (("sme", "smn"), "sma"), "turkic": (("tur", "azj"), "kaz"), "mongolic": (("khk", "bua"), "xal")}
# donor families: separate trees in the forest under both hypotheses, never joined, so that "shared from Latin" is a
# borrowing path the model can take instead of evidence for a common ancestor; a bare language name is a root with one leaf
DONORS = {"latin": "lat", "hungarian": "hun", "basque": "eus", "livonian": "liv"}
# what is known (for the report only; the model never sees it)
RELATED = {frozenset(p) for p in [("ibero", "italo"), ("germanic", "nordic"), ("finnic", "saami"), ("romlat", "italo"),
                                  ("ibero", "germanic"), ("ibero", "nordic"), ("ibero", "slavic"), ("italo", "germanic"), ("italo", "nordic"),
                                  ("italo", "slavic"), ("germanic", "slavic"), ("nordic", "slavic"),
                                  ("romlat", "germanic"), ("romlat", "nordic"), ("romlat", "slavic")]}
DEEP = {frozenset(p) for p in [("ibero", "germanic"), ("ibero", "nordic"), ("ibero", "slavic"), ("italo", "germanic"), ("italo", "nordic"),
                               ("italo", "slavic"), ("germanic", "slavic"), ("nordic", "slavic"),
                               ("romlat", "germanic"), ("romlat", "nordic"), ("romlat", "slavic")]}

# Basic vocabulary: the Swadesh 100-item list and the Leipzig-Jakarta list (Tadmor 2009), mapped onto the NorthEuraLex
# concepts in data/northeuralex_sel.csv (identifier = NorthEuraLex concept number + gloss). Items with no NorthEuraLex
# counterpart are left out (all, many, suck, in, crush/grind, rope -> string); "hot/warm" is taken as warm; "claw" as nail.
# The point of the subset: the full 1,016-concept list carries months, "connect", "cup" etc., where shared vocabulary
# across families is largely borrowed (Latin, Low German), so a relationship test on the full list credits contact
# as inheritance. Basic vocabulary is where the comparative method looks because it is borrowed least.
BASIC = """640_i 641_thou 642_he 643_we 637_this 638_that 647_who 646_what 620_not 653_one 654_two 481_big 483_long 482_little
194_woman 193_man 192_human 195_child 160_fish 148_bird 139_dog 167_louse 169_fly 166_ant 112_tree 175_seed 117_leaf 111_root 114_bark
299_wood 40_skin 350_meat 41_blood 44_bone 355_fat 129_egg 122_horn 128_tail 123_feather 125_wing 11_hair 19_head 2_ear 1_eye 3_nose
4_mouth 5_tooth 6_tongue 32_nail 34_foot 38_leg 37_thigh 36_knee 28_hand 21_belly 22_navel 20_back 17_neck 24_breast 46_heart 48_liver
736_drink 737_eat 854_bite 800_see 803_hear 1000_know 741_sleep 750_die 944_kill 768_swim 772_fly 757_walk 773_go 779_come 679_lie
678_sit 677_stand 898_give 981_say 834_make 861_hit 791_laugh 793_weepcry 941_conceal 873_carry 887_take 763_run 696_fall 690_blow
857_tie 57_sun 58_moon 59_star 63_water 76_rain 61_wind 64_stone 191_sand 66_earth 65_ground 75_cloud 68_smoke 70_fire 183_ash 697_burn
402_path 96_mountain 307_house 385_string 72_shadow 358_salt 526_red 529_green 527_yellow 525_white 524_black 454_night 595_yesterday
503_warm 504_cold 510_full 564_new 563_old 558_good 496_round 509_dry 575_far 500_heavy 488_thick 485_wide 494_hard 519_sweet 520_bitter
230_name""".split()


def _estimate(args):
    ds, th, seed, K, sweeps, burn, thin, warmup, power, init_prior, g_tables, spacing, direction, ess_min, boost, estimator = args
    ev.set_G(*g_tables)
    t0 = time.time()
    trace = {}
    if estimator == "re":
        val, terms, s = join_path_re(ds, th, seed, K=K, n_sweeps=sweeps, burn=burn, thin=thin, warmup=warmup, power=power,
                                     init_from_prior=init_prior, trace=trace, spacing=spacing)
        trace["direction"] = "re"
    else:
        val, terms, s = join_path(ds, th, seed, K=K, n_sweeps=sweeps, burn=burn, thin=thin, warmup=warmup, power=power, init_from_prior=init_prior,
                                  trace=trace, spacing=spacing, direction=direction, ess_min=ess_min, boost=boost)
        trace["direction"] = direction
    from masscomp.data import to_str
    for key in ("root_forms_at1", "fam_forms_at1"):
        trace[key] = {k: [[to_str(form), round(p_, 3)] for form, p_ in v] for k, v in trace.get(key, {}).items()}
    return dict(lbf=float(val), terms=[float(t) for t in terms], secs=time.time() - t0, t_root_end=float(s.st.theta.t_root),
                omega_end=float(s.st.theta.omega), lam_end=[round(float(x), 2) for x in s.st.theta.lam], trace=trace)


def lbf(p):
    require_extended_inventory()
    data = p.get("data", "data/northeuralex_sel.csv"); n = int(p.get("n", 50)); seed = int(p.get("seed", 1))
    f1, f2 = p["fam1"], p["fam2"]
    K = int(p.get("K", 16)); sweeps = int(p.get("sweeps", 600)); burn = int(p.get("burn", 200)); warmup = int(p.get("warmup", 800))
    power = float(p.get("power", 2.0)); reps = int(p.get("reps", 1)); g_mode = p.get("g", "flat"); len_edge = float(p.get("len_edge", 0.3))
    thin = int(p.get("thin", 4)); init_prior = p.get("init", "copy") == "prior"; spacing = p.get("spacing", "top")
    direction = p.get("direction", "down"); ess_min = float(p.get("ess_min", 0)); boost = int(p.get("boost", 1))
    estimator = p.get("estimator", "ss")
    concept_set = p.get("concepts", "all")                      # all | basic (BASIC above)
    donors = [(d, DONORS[d]) for d in p.get("donors", "").replace("+", ",").split(",") if d]
    families = [(f1, FAMILIES[f1]), (f2, FAMILIES[f2])]
    fam_langs = leaf_langs(families); donor_langs = leaf_langs(donors); langs = fam_langs + donor_langs
    forms_by, _, report = load_cldf(data, langs)
    common = sorted(set.intersection(*[set(forms_by[L]) for L in fam_langs]))   # the families decide the pool, so the
    pool = [c for c in common if c in set(BASIC)] if concept_set == "basic" else list(common)   # draw matches a run without donors
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    dropped = []
    if donor_langs:                                             # a concept must have a form in the donors too: swap the
        has = lambda c: all(c in forms_by[L] for L in donor_langs)   # few that lack one for further draws from the pool
        dropped = [c for c in chosen if not has(c)]
        spare = [c for c in pool if c not in set(chosen) and has(c)]
        extra = rng.choice(spare, size=min(len(dropped), len(spare)), replace=False).tolist() if dropped else []
        chosen = sorted([c for c in chosen if has(c)] + extra)
    rest = [c for c in common if c not in set(chosen)]          # G is fitted to every other concept, basic or not
    if g_mode == "fit":
        ev.fit_G([forms_by[L][c] for L in fam_langs for c in rest])   # the joined families' other concepts, as without donors
    else:
        ev.reset_G()
    f = build_forest(families, len_edge, donors=donors)
    name2v = {v: k for k, v in f.names.items()}
    M = len(chosen)
    ds = Dataset(f, M, {(name2v[L], ci): forms_by[L][c] for L in langs for ci, c in enumerate(chosen)})
    jobs = []
    for r in range(reps):
        th = sample_theta(f, M, np.random.default_rng(seed * 100 + r))
        jobs.append((ds, th, seed * 1000 + r, K, sweeps, burn, thin, warmup, power, init_prior, (ev.G_LEN_RAW.copy(), ev.G_SEG.copy()), spacing,
                     ("down" if r % 2 == 0 else "up") if direction == "alt" else direction, ess_min, boost, estimator))
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
                            lam_sd=ev.LAM_SD, lam_min=[None if not np.isfinite(x) else float(x) for x in ev.LAM_MIN],
                            rate_time=RATE_TIME, t_ref=T_REF, directional=DIRECTIONAL,
                            u_env_rules=ev.U_ENV_RULES, u_rules=ev.U_RULES, u_per_t=ev.U_PER_T, concepts=concept_set, pool=len(pool),
                            donors=donors, dropped_for_donors=dropped, spacing=spacing, direction=direction, ess_min=ess_min, boost=boost,
                            estimator=estimator),
                known="related" if key in RELATED else "unrelated", deep=key in DEEP, common_concepts=len(common),
                lbf_mean=float(np.mean(vals)), lbf_sd=float(np.std(vals)) if reps > 1 else None, estimates=ests, secs=time.time() - t0)


def summary_line(r):
    c = r["config"]
    return (f"{c['fam1']}-{c['fam2']} ({r['known']}{', deep' if r['deep'] else ''}) G={c['g']} n={c['n']}/{c.get('pool', '?')} {c.get('concepts', 'all')}"
            f"{' donors ' + '+'.join(d for d, _ in c['donors']) if c.get('donors') else ''} K={c['K']}x{c['sweeps']}"
            f"{' ' + c['spacing'] + '^' + str(c['power']).rstrip('0').rstrip('.') if c.get('spacing', 'top') != 'top' or c.get('power', 2) != 2 else ''}"
            f"{' RE' if c.get('estimator') == 're' else ''}: "
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

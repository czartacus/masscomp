# masscomp-proto

Prototype of the core model in the *joint-forest* specification (roadmap step 1): sound change,
lexical replacement and borrowing on a fixed two-family forest, with MCMC inference and a
stepping-stone log Bayes factor for "one root vs two roots". Built to the brief in the
spec's **Prototype plan** tab.

## Run

```
pip install numpy scipy pytest
python -m pytest tests -q                       # ~4 min; the two MCMC tests dominate
PYTHONPATH=. python studies/profiling.py        # seconds per sweep at benchmark A and B
PYTHONPATH=. python studies/recovery.py 50 0    # §8.1  (args: n_reps seed0 [sweeps burn])
PYTHONPATH=. python studies/confound.py 20 0    # §8.2
PYTHONPATH=. python studies/lbf.py 20 0 1.0,2.0,3.0   # §8.3 (args: n_per_cell seed0 depths [K sweeps burn])
```

Each study writes `results/<study>_<seed0>.json`; nothing in `results/` is edited by hand.

## Layout

| Module | Plan section | Notes |
| --- | --- | --- |
| `masscomp/segments.py` | §2 | 24 consonants + 5 vowels, 12-bit features, Hamming distance |
| `masscomp/tree.py` | §2 | benchmark-A forest; `donor_candidates` |
| `masscomp/transducer.py` | §3.1–3.2 | log-linear edge transducer; edit DP with insertion slots; memoised |
| `masscomp/events.py` | §3.3, §3.5 | competing hazards (single event per edge); priors |
| `masscomp/simulate.py` | §4 | forward simulator; `simulate_matched_pair` for §8.3 |
| `masscomp/likelihood.py` | §3.6 | cached edge terms with a prior / likelihood split for tempering |
| `masscomp/mcmc.py` | §5 | moves 1–7 plus a joint (event, form) move |
| `masscomp/marglik.py` | §6 | stepping-stone, K stones at β = (k/K)³ |

## Deviations from the plan, and why

- **Prior means for λ** are (4.5, 3, 2, 6), not N(0,1). With λ₀ ~ N(0,1) the identity probability
  per segment is ~50% and simulated families carry no lineage signal to recover. The chosen
  means give ~90% identity per segment per edge and ~7% insertion per slot.
- **Borrowing is acyclic.** Donors are restricted to nodes earlier in breadth-first order.
  Allowing A→B and B→A simultaneously made the joint density of forms unnormalised and the
  prior-only chain drifted to 60% borrowing. This is the one modelling change made during the
  build; it is a correctness fix, not a tuning.
- **Donor choice (1/#candidates) is part of the event prior**, not the likelihood. Putting it in
  the likelihood inflated borrowing by a factor of #candidates at low β.
- **ρ_c, ω, t_root** are updated by log-scale random-walk MH, not Gibbs / slice: the hazard
  probabilities are not conjugate.
- **A joint (event, form) move** was added: propose a new event on an internal node's incoming
  edge together with a form drawn from that event's own distribution. Without it the chain
  could not leave states where sibling forms were pinned together by borrowing.
- **Tempered chains start from a prior simulation**, not from copies of the data.
- **The λ step is 0.1 + 0.5(1−β)**: the prior is ten times wider than any posterior.
- **u_e recovery** is reported as posterior inclusion probability plus interval coverage of
  entries with |w| > 1. Most prior-drawn entries have |w| < 0.5 and are invisible to 25 concepts;
  a 90% interval that is exactly {0} cannot "cover" them, and treating that as failure would be
  a test of the prior, not of the sampler.
- **Chain lengths** in the studies default to 1,500 sweeps / 500 burn (plan: 5,000 / 1,000) and
  stepping-stone to K=32 × 2,000 sweeps as planned but overridable, because of the timings below.

## Timings (profiling.py, one core)

| Config | s / sweep | 3 chains × 5,000 sweeps |
| --- | --- | --- |
| A: 10 languages × 25 concepts | 0.33 | 1.4 h |
| B: 20 languages × 50 concepts | 1.9 | 7.9 h |

A full §8.3 at plan settings (K=32, 2,000 sweeps, 40 datasets × 2 hypotheses) is ~1,000 CPU-hours
in this implementation. The DP is memoised and already pure Python; the next step the plan names
(vectorised NumPy DP) is the right one if the studies are to run at plan scale.

## Study results

Settings and raw numbers are in `results/` (local) and `results/cloud/` (GitHub Actions, one job per
replicate; see `.github/workflows/study.yml`). Status as of 2026-11-03.

### Model change on 2026-11-02: cross-family borrowing donors, collapsed u move

A planted-rule trace (`studies/u_planted.py`) showed the chain preferring "every concept borrowed
from an uncle, ω large" to "one branch rule": with donors inside the recipient's own family,
borrow-from-uncle is the same likelihood as inheritance with the rule, and the event prior does not
pay for the rule. Two changes (commit 66348e4): `tree.CROSS_FAMILY_ONLY = True` restricts donors to
the other family (within-family contact is left to the sound-change model, as the spec's §4.4
intends); and `mcmc.Sampler.move_u` now marginalises the edge's events when proposing a rule, so a
rule can be found while the concepts it explains are still labelled "borrowed". Everything below
that is marked **v3** ran under this model; earlier numbers did not.

### §8.3 LBF separation — estimator settled (parallel tempering), grid rerunning under v3

- Downward stepping stone, β_k = (k/K)², K=24 (see earlier notes below). Single-chain replicate
  noise ~30 nats; 4 chains pooled stone by stone (`results/cloud/lbf_41455207637.json`) 17 nats per
  chain and the pooled value dominated by the best chain: four chains, four modes.
- **Parallel-tempered stepping stone** (`marglik.stepping_stone_pt`: all 25 stones run together with
  neighbour swaps) on the same seed-2101 pair, 4 replicates each
  (`results/cloud/pt_41469491586.json`): replicate sd **9.4 nats** at the same sweep budget. Swap
  acceptance is 10–50% for β < 0.3 and 1–2% for the top ten stones, so the schedule still needs more
  stones near β = 1; left as is for now. PT is the standard estimator from here.
- The pooled grid and the PT check both ran on the pre-v3 model, where unrelated pairs scored up
  to +68 (H1 offers more donors to abuse). Run 41550036592 = the depth 0.25 / 0.5 grid, 5 related +
  5 unrelated per depth, 2 PT replicates per cell, under v3.

### §8.1 recovery — v3 fixes most of the ω bias; ancestral forms still the bottleneck

3 chains × 3,000 sweeps / 1,500 burn (`results/cloud/recovery_41515150819.json` M=25,
`recovery_41515170630.json` M=100), n = 49 each:

| | v2, M=25 | pre-v3, M=100 | **v3, M=25** | **v3, M=100** |
| --- | --- | --- | --- | --- |
| ω coverage / median over-estimate | 0.48 / 1.9× | 0.14 / 2.7× | 0.78 / 1.5× | 0.57 / 1.3× |
| ω coverage, true ω < 0.03 | 0.08 | — | 0.81 | — |
| λ₀–λ₃, ρ coverage | 0.80–0.92 | 0.71–0.96 | 0.84–0.94 | 0.84–0.94 |
| t_root coverage | 0.92 | 0.55 | 0.94 | 0.67 |
| root-set coverage / chains R̂ ≤ 1.25 | 0.62 / 0.46 | 0.59 / 0.37 | 0.60 / 0.39 | 0.60 / 0.37 |
| detectable \|w\| > 2.5 rule inclusion | 0 | 0 | 0 | 0.01 |

The donor change removed most of the ω inflation; what remains (1.3× at M=100, where the intervals
are narrow enough to miss) and the still-undetected planted rules both point at the ancestral-form
sampler, whose coverage and convergence have not moved across any version. `studies/u_diag.py`
fits one detectable-rule dataset with latents clamped at truth / started at truth / from the prior
to separate the u move from the form sampler.

### Pre-v3 notes (kept for the record)

#### §8.3 LBF separation — blocked on estimator noise

- The stepping-stone estimator must be run **downward** (fit at β=1, cool to the prior). Run upward
  it is biased 400–700 nats low with ±150 nat error, which produced a spurious sign flip (−63 vs +87)
  on one dataset. Spacing β_k = (k/K)² (not cubic), K=24, warm-up 1,200 sweeps, 2,000 per stone.
- **Horizon.** At t_root = 1.0 (prototype units) only 5 of 25 concepts remain cross-family cognates
  and the LBF is ≈ +1 for related pairs; the grid now uses depths 0.25 / 0.5 / 1.0.
- **Grid, 5 related + 5 unrelated per depth, one chain** (`results/cloud/lbf_41344502324.json`):
  depth 0.25 related +13 ± 26, unrelated +4 ± 29 (max **+50.6**); depth 0.5 related +19 ± 20,
  unrelated −19 ± 23; depth 1.0 no separation.
- **Replicates** of the two seed-2101 cells with fresh stepping-stone seeds
  (`results/cloud/lbf_41355534462.json`): related +28.5 / +14.6 / +40.0, unrelated +7.8 / −24.1 / −11.7.
  The +50.6 "false positive" is estimator noise; the per-estimate noise is ~30 nats, as large as the
  signal, so the grid above measures estimator variance, not separation. Per-stone diagnostics show
  ESS 1–2 at mid-β stones: the chain sits in one forest/root mode per run.
- **Fix in progress:** R independent chains per estimate (one job each), pooled stone by stone in
  `studies/merge.py` (log-mean-exp of per-stone terms). Run 41455207637 = 5 seeds × depths 0.25, 0.5
  × 4 chains. If pooling does not bring the replicate sd under ~10 nats, the next step is parallel
  tempering inside the sampler.

#### §8.1 recovery (pre-v3) — continuous parameters fine; ω and roots need more data, not more sampler

n = 50 at M = 25 concepts, 3 chains × 1,500 sweeps (`results/recovery_all.json`, then
`results/cloud/recovery_41346586750.json` after the u-proposal change):

| | before | after data-driven u birth proposal |
| --- | --- | --- |
| λ₀–λ₃, t_root, mean ρ coverage | 0.80–0.92 | 0.80–0.92 |
| ω coverage / median over-estimate | 0.50 / 2.4× | 0.48 / 1.9× |
| root-set coverage (converged chains) | 0.58 | 0.62 |
| chains with R̂ ≤ 1.25 | 50% | 46% |
| planted \|w\| > 2.5 terms with ≥ 3 inherited forms to act on | 0 of 50 datasets | 0 of 50 |

The large branch terms are undetectable at 25 concepts, so their 0 coverage is a property of the
data, and the ω inflation is the model absorbing mismatches that u cannot explain. The u proposal
(`mcmc.Sampler.move_u`: slot from mismatch counts, value from a N(2,1) mixture) is kept because
the planted-term unit test went from 0 to 1.0 inclusion. Run 41467498208 repeats the study at
M = 100 (`concepts` workflow input), where planted terms do become detectable.

#### §8.2 confound — fails as specified

20 replicates: 1 of 20 flags the borrowing-vs-inheritance ambiguity (target ≥ 0.8). The model
resolves the confound by parsimony rather than reporting it. Detection will have to come from the
prior-sensitivity grid (spec §9), not from the posterior of a single fit.

#### Earlier reduced-scale smoke results (2026-11-01, two contended cores)

Run on two contended cores inside one session; settings far below the plan's, so these are a
check that the machinery works, not the study.

| Study | Settings | Result |
| --- | --- | --- |
| §8.1 recovery | 4 replicates, 3 chains × 1,200 sweeps (plan: 50, 3 × 5,000) | continuous parameters (λ₀–λ₃, ω, mean ρ, t_root): 24 of 28 intervals cover the truth (0.86, target 0.85–0.95). Root forms: 8–24% in the 90% set, R̂ 1.64–1.71 → not converged. Large u_e entries: not recovered. |
| §8.2 confound | not run | — |
| §8.3 LBF | depth 1.0, one related + one unrelated pair, K=12 stones × 400 sweeps, one chain (plan: 20 per cell, K=32 × 2,000, three chains) | related **+86.7** nats; unrelated **−64.1** nats. One pair each, no MC error estimate. |

Reading: the parameter layer recovers and the LBF separates the one related pair from the one
unrelated pair by 150 nats at depth 1.0. The ancestral-form layer does not mix in 1,200 sweeps
with the current MH proposals; that is the known weakest part of the inference and the reason the
next change is the lattice-FFBS form sampler (spec §12.4), not more replicates. Logs:
`results/recovery_100.log`, `results/lbf_100.log`, `results/lbf_100.json`.

Two sentinel bugs were found by the stepping-stone run and fixed before the LBF numbers above:
impossible transducer outputs and innovation forms outside length 2–6 returned −1e30, which the
prior stage (β = 0) samples freely; both now carry a soft floor (see `transducer.FLOOR`, `events.log_G`).

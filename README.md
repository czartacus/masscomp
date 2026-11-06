# masscomp-proto

Prototype of the core model in the *joint-forest* specification (roadmap step 1): sound change,
lexical replacement and borrowing on a fixed two-family forest, with MCMC inference and a
stepping-stone log Bayes factor for "one root vs two roots". Built to the brief in the
spec's **Prototype plan** tab.

## Run

```
pip install numpy scipy pytest
python -m pytest tests -q                       # ~2 min; the prior-only MCMC test dominates
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

| Config | s / sweep (2026-11-04) | before | 3 chains × 5,000 sweeps |
| --- | --- | --- | --- |
| A: 10 languages × 25 concepts | 0.16 | 0.33 (v2), ~0.75 (v4 with the lattice moves) | 0.7 h |
| B: 20 languages × 50 concepts | 0.79 | 1.9 (v2) | 3.3 h |

On the recovery dataset used for the basin checks (seed 7, depth 0.25, M=25, regrow at 0.1) a
sweep went from 1.58 s to 0.18 s on 2026-11-04 (8.8×; `results/basin/` runs before that date are
at the old speed). Where it came from, in order of size:

- **Lattice forward in linear space** (`ffbs.forward_lin`): the alignment-lattice forward pass was
  62% of the sweep and did ~1.6 million log-add-exps per call. It now runs in linear space with
  one scale factor per layer, with the out-edge step written as a (symbol × 16 out-edge options)
  matrix product so the inner loops vectorise: 6.6 → 0.29 ms per call at two out-edges, identical
  to the log-space loops to 1e-13 (`forward` is kept as the reference; `ffbs.USE_LIN = False`).
- **Rule move** (`move_u`, 57 → 10 ms): a proposal on a substitution/deletion slot (s, o) changes
  row s of one table (`EdgeParams.derive`), so only the concepts whose parent form contains s have
  their inherit term recomputed; the replace/borrow options come from per-sweep tables shared with
  the rate move.
- **Rate move** (32 → 9 ms): the per-(edge, concept) option tables are built once per sweep as
  padded (concept × edge × option) tensors and the normalisers are one vectorised log-sum-exp.
- **Gibbs form move** (52 → 33 ms): the per-edge term evaluators are built once per site instead of
  going through `edge_terms` for every candidate; the favoured parent segments per child segment
  are tabulated once per transducer.
- Lazy transducer tables, O(present) rule-slot proposals, `fastmath` on the lattice.

Tried and not kept (no measurable gain): batching the edit-DP calls for the Gibbs weights (the
cost was the surrounding Python, not the compiled DP) and skipping dead lattice cells (most cells
are live). What remains (0.18 s): lattice forward ≈ 30%, compiled edit DP ≈ 10%, the rest Python
bookkeeping spread over the latent moves (Gibbs 34 ms, joint-FFBS 20, joint 19, FFBS 16, events 11,
forms 7; regrow at 0.1 adds ~80). An array-state rewrite of the latent moves would buy ~2× more,
not the 5× first estimated, because the compiled parts do not shrink; the move-mix tuning
(`kind=essrate` diag cells) is the cheaper lever.

## Diagnostics in the cloud (`.github/workflows/diag.yml`)

Any diagnostic that used to run serially on the development machine — a basin check, a Geweke run,
a mixing-rate measurement — is one cell of `studies/diag.py`, run one per runner (4 cores) through
the `diag` workflow and merged into `results/cloud/diag_<run_id>.json` with the exact spec of
every cell. Dispatch (REST; `gh workflow run` is not available from these sessions):

```
gh api -X POST repos/czartacus/masscomp/actions/workflows/diag.yml/dispatches -f ref=main \
  -f "inputs[label]=..." \
  -f "inputs[cells]=kind=basin,seed=7,depth=0.25,modes=copy+lattice+truth,regrow_p=0.1;kind=geweke,n=10000,m=2,seed=1,mode=latents,sweeps=2,regrow_p=0.3,latent_moves=regrow;kind=essrate,seed=7,depth=0.25,start=truth,sweeps=3000,burn=500,latent_moves=ffbs+jffbs+events"
```

Cell kinds: `basin` (`root_diag.basin`: copy / lattice / truth / prior / ptK starts on one recovery
dataset), `geweke` (`studies/geweke.py`), `essrate` (one chain; Geyer ESS per CPU-second of the
log-likelihood, ω, λ, t_root, |u| and the root forms' change rate, for comparing move mixes),
`realdata` (`studies/realdata_probe.py`: Romance reconstruction with Latin held out) and
`realdata_lbf` (`studies/realdata_lbf.py`: join-path LBF of a named family pair, `fam1=`, `fam2=`).
Sampler and model knobs (`regrow_p`, `regrow_events`, `regrow_borrow`, `ffbs_p`, `latent_moves`,
`init`, `g`, `inventory`, `lam_sd`, `u_context`, `u_rules`, `u_per_t`, …; any other as
`env_<name>=`) go in the spec and become the `MASSCOMP_*` environment variables. The account allows 20 concurrent
jobs (public repository: minutes are not metered), shared with the `study` workflow.

## Study results

Settings and raw numbers are in `results/` (local) and `results/cloud/` (GitHub Actions, one job per
replicate; see `.github/workflows/study.yml`). The sections below are chronological; this is where
things stand as of 2026-11-05 (the plan's §10 verdict is in the spec, Prototype plan §12).

| | verdict | the number | where |
| --- | --- | --- | --- |
| §8.1 recovery | passes | 144 datasets: λ₀–λ₃ covered at 0.87–0.89, ρ 0.90, ω 0.875, t_root 0.99; root forms 0.95 (Rao-Blackwellised) | §8.1 v4, Findings of 11-04 |
| §8.2 confound | fails as specified | 1 of 20 replicates flags the ambiguity; detection moves to prior-sensitivity refits | Pre-v3 notes |
| §8.3 separation | passes on the join path | K = 16, two replicates: power 0.90 / 1.00 / 0.90 at depths 0.25 / 0.5 / 1.0; K = 24, single replicates: 0.90 / 1.00 at 0.25 / 1.0, unrelated 99th pct +2.4 | Findings of 11-04 |
| sampler | correct | Geweke restart batteries: max \|z\| ≤ 2.6 under the default moves, a random non-flat G and the context rules | Sampler correctness; 11-05 sections |
| real data | separates known pairs | NorthEuraLex, 11 family pairs, fitted G, basic vocabulary: related within a family +45 to +89, unrelated −7 to −29; one of two PIE-depth pairs at +7.6; flat G breaks the test (+143 on an unrelated pair) | Real-data relationship tests |

Protocol for real data from here: the extended inventory, G fitted to the languages' other concepts,
λ free, all available basic vocabulary, ten context-free rules per edge (`u_rules=10`), the join-path
estimator on the two-sided ladder (`spacing=both,power=3`, K = 24, one replicate cooled and one
heated, the bracket reported), donor families in the forest for known contact sources that are not
ancestors, and the per-word attribution as the audit. Reference numbers under this protocol:
Ibero-Romance–Germanic +35 (+22…+48), Ibero-Romance–Slavic +35 (+23…+48), Germanic–Slavic 0 (±3),
Ibero–Finnic +3 (−3…+8), Germanic–Finnic −6.

### Model change on 2026-11-02: cross-family borrowing donors, collapsed u move

A planted-rule trace (`studies/u_planted.py`) showed the chain preferring "every concept borrowed
from an uncle, ω large" to "one branch rule": with donors inside the recipient's own family,
borrow-from-uncle is the same likelihood as inheritance with the rule, and the event prior does not
pay for the rule. Two changes (commit 66348e4): `tree.CROSS_FAMILY_ONLY = True` restricts donors to
the other family (within-family contact is left to the sound-change model, as the spec's §4.4
intends); and `mcmc.Sampler.move_u` now marginalises the edge's events when proposing a rule, so a
rule can be found while the concepts it explains are still labelled "borrowed". Everything below
that is marked **v3** ran under this model; earlier numbers did not.

### §8.3 LBF separation — first v3 grid: no separation; the PT estimator rebuilt

Run 41550036592 (v3 model, M=25, K=24, 1,500 sweeps / 400 burn per stone, 2 PT replicates per
cell, 5 related + 5 unrelated seeds, depths 0.25 and 0.5; `results/cloud/pt_41550036592.json`):

| depth | related LBF, mean (sd) | unrelated LBF, mean (sd, max) | power at the 99% null quantile |
| --- | --- | --- | --- |
| 0.25 | +9 (10) | +6 (27, +57) | 0 |
| 0.5 | −2 (15) | same data as above | 0 |

Replicates of the same marginal differ by up to 80 nats (sd of a single estimate ~19 nats, versus
9.4 in the earlier noise check), swap acceptance at the cold end of the ladder is 0.00 in every
cell, and the +57/+77 unrelated outliers are single replicates: the estimator, not the model, is
what blocks this study. (Two bugs found in the same read: `merge.py` pooled related cells across
depths because its key omitted depth — the numbers above are from the corrected merge — and
unrelated data were simulated per depth although depth does not enter them.)

What was wrong, and the fixes (commits 629a327, 3ca97e1, d29da3c):

- **Stones started from the prior and had not equilibrated.** A local trace showed stones still
  drifting by hundreds of nats after 200 sweeps, with a non-monotone ll(β) curve. `stepping_stone_pt(warmup=N)`
  now runs one β = 1 chain first and starts every stone from its state; the stones relax downhill
  quickly and the curve is monotone.
- **Fixed (k/K)² ladder.** During the first 80% of burn-in the interior betas are re-spaced so that
  Δβ · sd(ll) is equal across stones (equal swap difficulty, equal stone-term variance); the ladder
  is frozen before recording, so the estimator is unchanged. An acceptance-gradient (Vousden) update
  was tried first and cannot move when all acceptances are zero.
- **One core of four.** The K+1 chains now run in persistent worker processes; a swap exchanges β
  labels, not states, so only scalars cross the pipes (identical results, ~3.5× on the runners).
- H1 and H0 are separate jobs (`pt_hyps`), unrelated data run once per seed.

What remains is arithmetic: sd(ll) per stone is 25–70 nats at M=25, so ~1 nat of Δβ · sd per stone —
acceptance in the tens of percent — needs ~50 stones, and ~100 at M=100. Runs 41618398207 (K=24)
and 41618402064 (K=48), 3 warm-started replicates per marginal on one related and one unrelated
pair, measure the noise under the rebuilt estimator before the grid is rerun under v4.

**Result of the noise check (read 2026-11-04; `results/cloud/pt_41618398207.json`, `pt_41618402064.json`).**
Doubling the ladder does not reduce the noise. Per-estimate sd of a log-marginal is 21 nats at
K=24 and 29 at K=48 (related LBF replicates 56/62/2 at K=24, 43/92/14 at K=48; unrelated −19/14/19
and −16/−36/4). Mean swap acceptance rises from 0.15 to 0.40 with K=48, as the arithmetic
predicted, but the *minimum* adjacent-pair acceptance stays at 0.00–0.01 in every run: one pair of
neighbouring rungs near the cold end never exchanges, whatever the spacing. That is the signature of
a first-order transition in β — the cold replica sits in one of the latent basins described under
*Mixing* below and the warm replicas in another — so the estimator's noise is the sampler's basin
problem seen from a different side, not a ladder-spacing problem. The §8.3 grid stays blocked until
the cold-end mixing is fixed; the K=48 "pooled" rows in that file are from a stale merge and should
be ignored (the per-replicate rows are correct).

### §8.1 recovery — v4 (wide slab, collapsed rate moves, reverse proposals)

Model v4 (commits b8a659e, 3a39a13, 32505b5): the rule slab is widened from sd 1 to sd 4√t
(`events.SIGMA2_U = 16`; with the identity logit near 6 an sd-1 rule was inaudible), ρ_c and ω are
updated with the edge events marginalised out, and half the ancestor proposals go through the
reversed transducer. "Detectable" now means |w| > 2.5 *and* the rule visibly fired at least three
times (the old criterion counted rules with opportunities that never fired).

3 chains × 3,000 sweeps / 1,500 burn, n = 49 per cell:

| | v2, M=25 | **v3, M=25** | **v3, M=100** | **v4, M=25** | **v4, M=100** |
| --- | --- | --- | --- | --- | --- |
| ω coverage / median over-estimate | 0.48 / 1.9× | 0.78 / 1.5× | 0.57 / 1.3× | 0.84 / 1.5× | 0.69 / 1.3× |
| λ₀–λ₃, ρ coverage | 0.80–0.92 | 0.84–0.94 | 0.84–0.94 | 0.78–0.88 | 0.76–0.90 |
| t_root coverage | 0.92 | 0.94 | 0.67 | 0.98 | 0.69 |
| root-set coverage / chains R̂ ≤ 1.25 | 0.62 / 0.46 | 0.60 / 0.39 | 0.60 / 0.37 | 0.60 / 0.31 | 0.60 / 0.24 |
| detectable-rule coverage (n rules) | 0 | 0 (2) | 0.01 | 0.38 (15) | 0.47 (101) |

(`results/cloud/recovery_41515150819.json`, `recovery_41515170630.json`, `recovery_41558063972.json`,
`recovery_41558066414.json`.) Rerun at M=25 on the same seeds after the `move_u` birth-density fix
(bba12c5; `recovery_41722894945.json`, n = 48): ω coverage 0.92 at 1.35× over, t_root 1.00, λ/ρ
0.77–0.90, root-set 0.63, chains R̂ ≤ 1.25 0.52, detectable rules 0.38 (15). The u fix moves ω and
between-chain agreement a little; the root forms do not move.
The planted-rule traces behind v4 (`studies/u_planted.py`, `u_diag.py`): with latents clamped at
truth the sampler finds the planted rule with inclusion 1.0 and ω exact; started from the study's
child-copy initialisation, inclusion 1.0 and ω 1.2–1.3×; started from a prior draw it lands in a
wrong basin (inclusion 0.6, ω 24×). So the parameter moves are sound and the residual ω bias and the
unchanged root-set coverage are the ancestral-form layer, which no version has moved.

### Sampler correctness and mixing (2026-11-03)

The root-set coverage of 0.60 has not moved across four model versions, and only 24–46% of the
chains agree with each other (R̂), so the question was whether the sampler is *wrong* or merely
*stuck*. Two lines of work, both in `studies/`:

**Getting it right (Geweke 2004), `studies/geweke.py`.** Marginal-conditional draws (θ from the
prior, data simulated) against successive-conditional draws (one sampler sweep, then the leaf forms
redrawn exactly from their conditional through the FFBS lattice). The test found two real bugs —
`move_u` understated the birth-slot density by log 2 on edges with no mismatch counts (bba12c5), and
the exact side had to draw root-edge rules at t_root rather than the forest's fixed length — and then
produced apparent z-scores of 5–16 on the form-length statistics for every latent-move subset, in
inconsistent directions. Those were an artefact of the error bars: batch-means ESS (b = √n)
overstates the ESS of a chain whose autocorrelation time for form lengths is in the thousands. With
Geyer's initial-positive-sequence ESS and long runs (`results/geweke/`; 30,000 iterations × 2
sweeps for the latent and full tests, 12,000 × 6 for the parameter moves) every statistic is within
noise: max |z| 2.1 over 5 latent statistics (ESS 36–879), 2.4 over 14 in the full test, 2.2 over 14
for the parameter moves — 33 z-scores in all, none beyond 2.5. **The sampler targets the posterior.**
The remaining problem is mixing.

**Mixing.** `studies/basin_swap.py` locates the gap between a chain started from child copies and
one started from the truth (130–340 nats at depth 0.25): swapping the truth latents under the copy
chain's θ recovers +136 nats as a sum of per-concept gains (16/25 concepts, 8–52 nats each), while
swapping θ alone recovers +34, so the basins are coupled through θ *and* through many independent
per-concept latents — no single-site move crosses. Tried and insufficient on their own: exact FFBS
Gibbs over an ancestor form (`move_forms_ffbs`, up to two outgoing edges), the collapsed
event-plus-form move (`move_joint_ffbs`), tempered transitions per concept (`move_tempered`,
`MASSCOMP_TEMPER_P`), prior-side tempering in the PT ladder (`Sampler.bp`, which in fact pulled ω
up 2×), and a subtree regrow move (`move_regrow`, `MASSCOMP_REGROW_P`; accepted ~49% of the time and
correct under Geweke, but it does not leave the basin at depth 1.5 or 0.25). Both starts put t_root
at 0.6–1.8 against a truth of 0.25. Candidates not yet tried: a joint θ-plus-latents proposal,
SMC/annealed importance sampling from the prior, and more PT rungs at the cold end.

### Findings of 2026-11-04 (cloud diagnostics)

**The form-length support was too small.** Ancestral forms were capped at 8 segments
(`MAX_FORM_LEN`, `ffbs.T_MAX`). Of 60 prior-simulated recovery datasets at depth 0.25, 22 contain
true ancestors longer than 8 (2% of all ancestral forms, 4.4% of leaf forms); on seed 2 (λ₃ = 3.6,
insertion probability ≈ 0.3 per slot) 24% of the ancestors and 42% of the leaves exceed 8 and one
leaf is 26 segments. On that dataset even truth-started chains cannot hold the truth
(`results/cloud/diag_41804631324.json`: root-set coverage 0.25–0.33 from every start, ω 5–26× the
truth, mean log-likelihood 585 nats apart between starts). The cap is now 12 (`MASSCOMP_MAX_FORM_LEN`),
covering 99.5% of simulated ancestors, at ~10% cost per sweep. Whether the simulator should produce
such words is a question for the spec (plan §2 says forms are 2–6 segments; the λ₃ prior as coded
puts a few per cent of its mass on runaway insertion).

**Move mix (ESS per CPU-second, `diag_41806079603.json`).** Sixteen cells on seed 7 at depth 0.25
(eight mixes × copy / truth start, 3,000 sweeps). The candidate-set Gibbs move came bottom on
log-likelihood ESS/s in every mix containing it on both starts (0.02–0.12 against 0.15–0.21 without)
while costing a third of the sweep; it is off by default (`MASSCOMP_LATENT_MOVES` still accepts
`gibbs`). The older MH moves (`forms`, `joint`) help |u| and λ mixing (ESS/s 1.56 vs 0.93) for 30%
of the cost and stay. No mix crosses from the copy-start basin to the truth-start one in 3,000
sweeps (ω 3–11× vs 1.3–2.8× the truth), so the start point and the root-only move below are what
is being tested, not the mix.

**Basin controls on four datasets (`diag_41804631324.json`, 1,500 sweeps).** With the regrow move off,
copy and truth starts agree on root coverage on seeds 1, 3 and 7 (0.65–0.91) but differ in the
posterior by 30–150 nats: the copy basin has ω 1.4–5.5× (truth start 0.8–2.5×) and, on seed 1,
t_root 1.2–1.7 against 0.26–0.30 — the second family explained as replaced at the root and the root
edge stretched to make that cheap. The regrow move with events re-decided (inherit / replace /
borrow) halves the ω gap at p = 0.1 and closes it on seed 7 at p = 0.3 (both starts ω 2.4×,
posteriors 18 nats apart) at three times the sweep cost; it passes Geweke (max |z| 1.4 over the
latent statistics, borrow fraction now exercised).

**Root-only regrow closes most of the gap** (`diag_41811261079.json`, cap 12, no Gibbs move, 3,000
sweeps). `move_root_regrow` is an exact Gibbs step on the joined root's form and the events of its
two edges (the regrow machinery with the subtree limited to the root): 10 ms per sweep for 25
concepts, accepted always. With it the posterior gap between copy-started and truth-started chains
is 37 / 66 / 7–40 nats on seeds 1 / 3 / 7 (was 80–190), t_root from the copy start 0.25–0.9 (was
1.2–2.1), and copy-start root coverage 0.96 / 0.75 / 0.70–0.78. On seed 7 with the subtree regrow
added (p = 0.1) all three starts agree within 15 nats with the same ω. It is on by default
(`MASSCOMP_ROOT_REGROW_P=1`). The *leaf-anchored initialisation* (`MASSCOMP_INIT=lattice`, every
ancestor drawn bottom-up from G × its children's transducers) did not help — on seed 1 it found a
worse t_root basin (1.9–2.4) — and stays optional. Seed 2 under the new cap: truth-start root
coverage 0.67 and a 200-nat better posterior than before, copy start still 130 nats behind.

**§8.1 rerun with root regrow (`recovery_41818033147.json`, 48 datasets, 3,000 sweeps).** Chains
now agree: R̂ ≤ 1.25 on 92% of datasets (was 52%). Coverage: λ 0.81–0.88, ρ 0.90, t_root 1.00,
ω 0.83 (median 1.43× over), detectable rules 0.54 (n = 15). Root-set coverage **unchanged at 0.64**,
and per dataset almost identical to the previous run's (seed 424: 0.11 vs 0.16; seed 446: 0.94 vs
0.94): it is a property of the dataset, not of the sampler.

**The root-set metric was the problem.** On seed 424 (λ₀ = 2.6, heavy sound change) the exact
conditional of the joined-root form, computed at the chain's states, puts the truth at
probability e⁻⁷ to e⁻²⁵ — and the most probable forms at the same level: the posterior is diffuse
over hundreds of forms, as it should be when the root edges carry little information. The 90% HPD
set of such a posterior has hundreds of members and contains the truth, but the set assembled from
a few hundred samples holds a sliver of the mass and counts the truth as missed. The study now also
reports a **Rao-Blackwellised HPD membership** (`studies/common.root_hpd_rb`): for each recorded
state the exact conditional of the root form with the adjacent events marginalised
(`node_conditional`, the arithmetic of `move_root_regrow`), averaged over states to give every
sampled form's and the truth's marginal probability; the truth is in the 90% set iff the posterior
mass on forms more probable than it is below 0.9, estimated as the fraction of samples whose form
is more probable than the truth. Seed 424: 0.90 (joined root 1.00, family roots 0.88) against 0.00
/ 0.10 by the sample set; the truth's mass-above quantiles [0.21, 0.29, 0.48, 0.77, 0.86] against
the [0.1, 0.25, 0.5, 0.75, 0.9] of a calibrated posterior. The full study with both metrics is run
41827115991 (below). The family-root conditional is skipped in states where a cross-family edge
borrows from that root (the lattice holds two out-edges), which is rare.

**§8.1 with the Rao-Blackwellised root metric (`recovery_41827115991.json`, 47 datasets — seed 442
crashed on an empty form before the fix, 3,000 sweeps, root regrow on).**

| quantity | value | plan criterion |
| --- | --- | --- |
| root forms in the 90% HPD set, RB estimate | **0.94** (joined root 0.98, family roots 0.93) | ≥ 0.80 |
| same by the sample-set estimate | 0.64 (joined root 0.55) | — |
| truth's mass-above quantiles (10/25/50/75/90%) | 0.00 / 0.00 / 0.27 / 0.62 / 0.83 | 0.1 / 0.25 / 0.5 / 0.75 / 0.9 if calibrated |
| λ₀ … λ₃ coverage | 0.85 / 0.81 / 0.83 / 0.87 | 0.85–0.95 |
| ρ (mean) / t_root coverage | 0.89 / 0.98 | 0.85–0.95 |
| ω coverage, posterior-mean ratio | 0.79, median 1.36× | 0.85–0.95 |
| detectable rules covered | 0.54 (n = 15) | — |
| chains agreeing (R̂ ≤ 1.25) | 0.87 | — |

The root-form criterion passes; the posterior is if anything conservative (the truth sits higher in
the posterior than a calibrated model would put it: 0.94 instead of 0.90, quantiles below nominal).
The continuous parameters sit at or just under the 0.85 floor (s.e. ≈ 0.045 at n = 47): λ₁, λ₂ at
0.81–0.83 and ω at 0.79, with ω still 1.4× over in posterior mean — the one remaining systematic
shortfall, and it is small enough that the next question is whether 3,000 sweeps give the ω
posterior its full width. The `pass` flag in the file is false because of these three.

**§8.3 noise check under the new cap (`pt_41815813221.json`, K=24, 3 replicates, root regrow still
off in this run).** Not improved: per-estimate sd of a log-marginal 40–66 nats (related LBF
replicates −23 / +118 / +87, unrelated +62 / −4 / +73), minimum adjacent-rung swap acceptance 0.00 in
every run. The cold-end stone terms differ by 50–100 nats between replicates of the same marginal:
the stones do not equilibrate across the latent modes in 1,500 sweeps. The same check with the root
regrow move on is run 41823009928.

**Where the ω over-estimation comes from (`diag_41833705645.json`).** On the four §8.1 datasets with
substantial ω that the posterior over-shot (404, 422, 423, 446; depth 1.5, root regrow on), chains
started from the truth put ω at 1.14–1.70× the truth and chains started from child copies at
1.5–2.5×, with the truth-started posterior 44–150 nats higher — all of it on the prior side (internal
edges and events), the leaf likelihood equal. At the study's depth of 1.5 the root edges are mostly
replaced, so the root step cannot reach the basin, which lies inside the families. The remaining
§8.1 shortfall (ω coverage 0.79, λ₁–λ₂ 0.81–0.83) is therefore still mixing. The subtree regrow with
events and borrowing re-decided closes it (`diag_41844468490.json`): at p = 0.15 / 0.3 on seeds 404
and 423 the copy and truth starts agree on ω (1.39 vs 1.21× and 1.69 vs 1.79× at 0.15; 1.27 vs 1.23×
and 1.73 vs 1.75× at 0.3) and the copy-start posterior is within ±50 nats of the truth-start one,
on either side. It is on by default at p = 0.15 (`MASSCOMP_REGROW_P`), ~0.19 s per sweep on the
depth-1.5 datasets.

**§8.1 with the subtree regrow (`recovery_41853059272.json`, 48 datasets).** ω coverage **0.94**
(from 0.79), posterior-mean ratio median 1.19 (from 1.36); the three remaining ω misses are datasets
whose true ω is ~0.005, below what 25 concepts can resolve. ρ 0.92, t_root 1.00, λ₀–λ₃ 0.83 / 0.81
/ 0.85 / 0.85, root forms (RB) 0.95 (joined root 0.98, family roots 0.95), chains agreeing 0.94,
detectable rules 0.69. Everything is at the plan's criteria except λ₀ and λ₁ at 0.81–0.83 against
the 0.85 floor — one to two standard errors (0.043) under it. A further 96 datasets (seeds 448–543,
same settings; `recovery_41874847711.json`) settle it as noise:

| §8.1, pooled 144 datasets (M = 25, depth 1.5, 3,000 sweeps, root step + subtree regrow) | coverage | criterion |
| --- | --- | --- |
| ω (posterior-mean ratio median 1.18×) | 0.875 | 0.85–0.95 |
| λ₀ / λ₁ / λ₂ / λ₃ | 0.868 / 0.889 / 0.889 / 0.889 | 0.85–0.95 |
| ρ (mean) | 0.896 | 0.85–0.95 |
| t_root | 0.986 | 0.85–0.95 |
| root forms in the 90% HPD set (RB) | 0.954 (joined root 0.98) | ≥ 0.80 |
| detectable rules (n = 41) | 0.72 | — |
| chains agreeing (R̂ ≤ 1.25) | 0.944 | — |

Standard error 0.025 at n = 144. The λ misses are symmetric (posterior above the truth in 7 of 19
for λ₀, 9 of 16 for λ₁), so there is no residual bias. **§8.1 passes**, with one reading to make:
t_root is *over*-covered (0.99) because the model identifies it only through the root-edge events
and rules (12.13), so its posterior is wide by construction — a property of the spec, not of the
sampler. The study's `pass` flag is still false on the two 48-dataset files alone; the pooled numbers
above are the result.

**§8.3 noise check with root regrow (`pt_41823009928.json`).** The H1 marginal now replicates: related
−4021 / −4019 / −4025 (sd 2.9), unrelated sd 9.2. The H0 marginal does not: −4087 / −4101 / −4024
(sd 41) and 44 for the unrelated pair, so the LBF still scatters by 40–50 nats and the cold-end
rung pair still never swaps. The join-path estimator below avoids estimating H0 at all.

**Join-path LBF (`marglik.join_path`, `studies/lbfj.py`).** One stepping stone per dataset along the
root join: the inherit / borrow terms of the two root edges are raised to α and the family roots'
innovation prior to 1 − α, so α = 1 is H1 and α = 0 is H0 with the joined root, t_root and the
root-edge rules and events as decoupled dummies (`State.alpha`, `State.join_delta`). Exact against
enumeration on a two-family toy (`tests/test_join_path.py`: 0.61 exact, estimates 0.69 / 0.67 /
0.54; the likelihood path gives 0.59). On the seed-2100 related pair (depth 0.25, K = 16, 600 sweeps
per stone, ladder (k/K)², ~28 min on one core) two replicates give +51.1 and +44.0 against the
likelihood path's −23 / +118 / +87 on the same data. The ladder is now dense near α = 1, where the
power variable grows (the top stone had an effective sample size of 3). Five datasets × three replicates in the
cloud (`lbfj_41833085610.json`, depth 0.25, K = 16, 600 sweeps per stone):

| pair | related LBF (3 replicates) | unrelated LBF (3 replicates) |
| --- | --- | --- |
| 2100 | +57.2, +73.5, +48.1 | −4.0, −5.9, −7.1 |
| 2101 | +3.8, +4.7, +2.2 | −3.2, −1.8, −2.8 |
| 2102 | +30.7, +22.1, +19.8 | −3.1, −7.2, −1.1 |
| 2103 | +9.7, +19.7, +16.7 | −1.6, −0.4, −4.3 |
| 2104 | +28.6, +28.2, +40.5 | +2.9, +1.5, +2.3 |

Replicate sd 1–13 nats on related pairs and 1–3 on unrelated (the likelihood path: 40–70), every
unrelated pair at or below +3, the 99th-percentile null τ = 2.8 nats and 14 of 15 related estimates
above it (power 0.93 at depth 0.25). The first separation the study has shown.

**§8.3 grid on the join path (`lbfj_41847561276.json`: depths 0.25 / 0.5 / 1.0, 20 matched pairs each,
2 replicates, K = 16, 600 sweeps per stone, M = 25).**

| depth | related LBF, mean (per-pair min … max) | unrelated LBF, mean (max) | τ = null 99th pct | power |
| --- | --- | --- | --- | --- |
| 0.25 | +19.5 (1.3 … 65.2) | −3.7 (+4.3) | 4.3 | **0.90** |
| 0.5 | +9.9 (1.6 … 38.0) | −4.2 (−0.3) | −0.3 | **1.00** |
| 1.0 | +7.5 (−3.7 … 29.6) | −4.4 (−0.7) | −0.7 | **0.90** |

Per-pair values are the mean of the two replicates; τ is the 99th percentile of the unrelated
per-pair values and power the fraction of related pairs above it. On single replicates the
estimator's occasional 10–20 nat miss (replicate sd median 2–3, max 13 / 22 / 11 by depth; one
unrelated replicate at +11 whose twin was −2) lowers power to 0.57 / 0.82 / 0.65, so two replicates
per pair, or a longer ladder, are part of the protocol. Against the plan's criteria: power ≥ 0.9 at
every depth, unrelated 99th percentile ≤ 4.3 nats (criterion < 5), unrelated mean negative at every
depth. **§8.3 passes on the join-path estimator**, at the depths the simulator's horizon allows
(12.3), with the caveat that the related signal fades fast: a fifth of the related pairs at depth
1.0 sit below +2 nats.

**The longer ladder (`lbfj_41991819908.json`: K = 24, 600 sweeps per stone, the first ten datasets
of each depth, two replicates; 2.6 h per estimate against 0.6).** Replicate sd falls from a median of
3.3 / 4.1 nats (related, depths 0.25 / 1.0) and 3.7 / 1.7 (unrelated) at K = 16 to 1.0 / 1.3 and
0.5 / 0.6, and the single-replicate misses go with it: no unrelated replicate above +2.4 (K = 16 had
+11.0, +3.6, +2.6 on the same datasets), so **single replicates separate at K = 24: power 0.90 at
depth 0.25 and 1.00 at depth 1.0** (τ = +2.4 / +0.1) where K = 16 single replicates gave 0.57 / 0.65.
Per-dataset means agree with the K = 16 grid to 3–4 nats on average, so the shorter ladder was noisy
rather than biased. One related dataset (2106: +32.7 / +11.0 at K = 24, +26.3 / +7.4 at K = 16) is
bimodal on both ladders and sets the max replicate sd (15); nothing in the ladder fixes a posterior
with two basins. Protocol from here: K = 24 single replicates, or K = 16 in pairs when a replicate sd
is wanted. Related pairs below +2 nats: 2 of 20 at depth 0.25 (datasets 2107 and 2108 whose signal
is small on both ladders), none at depth 1.0.

**Geweke restart battery (`diag_41818729390.json`, `diag_41827548303.json`).** Sixteen cells, 12,000–
20,000 fresh exact draws each, three sweeps per draw: latent moves on four θ draws with ω from the
prior and at 0.5, each parameter move alone, the full sampler on two draws, the subtree regrow at
0.3 — max |z| 2.45 over ~220 statistics (one 3.04 on the t_root move that came back 2.28 on a
rerun). This is the correctness check from now on; the chained design's z-scores are not
interpretable when a θ draw mixes slowly.

**Geweke after the speedups, and a correction to the test.** With the chained successive-conditional
design, a move-subset cell that leaves most forms frozen (root regrow + events only) produced
z = −42 on the internal form length at an "effective sample size" of 20,000: the chain's time
average was one dataset's frozen values, and the error bar was meaningless. The same mechanism (very
slow event mixing under some θ draws, ESS 4–20) produced the 3–6σ event-fraction discrepancies on
seed 2 that appeared on both the old and new code, and the recurring positive z on the rare borrow
fraction. `studies/geweke.py` therefore has a **restart variant** (`MASSCOMP_GEWEKE_RESTART=1`):
every iteration starts from a fresh exact joint draw and records the state after a few sweeps.
Invariance is a one-step property, so this tests the moves directly, with iid error bars, and no
dependence on mixing. On seed 2 with borrowing made common (`MASSCOMP_GEWEKE_OMEGA=0.5`) the full
default move set including root regrow passes (max |z| 1.5 over 3,000 draws, every statistic). A
15-cell restart battery (latent moves on 4 seeds × 2 ω, each parameter move, full mode, regrow 0.3)
is the authoritative check from now on; its results are recorded below when in.

### Real-data groundwork (2026-11-04)

Not a study — groundwork for the tier-1-style probe the spec allows before any claim: known
relationships, attested proto-language, used to find what the prototype gets wrong on real words.

- `masscomp/data.py` reads a Lexibank / CLDF wordlist (`forms.csv` with CLTS segmentation) and maps
  every segment onto the inventory: already-present symbols as they are; length, nasalisation,
  velarisation, gemination and similar marks stripped; IPA variants folded (ɛ→e, ɪ→i, ʊ→u, ɔ→o,
  ɑ ɒ ɐ æ ʌ→a, ə ɯ→ɨ, ɾ ʁ ɹ→r, ʋ β→v, ɣ→ʕ, ɕ→ʃ, …); diphthongs split; multi-word forms and synonyms
  dropped (the prototype has no lexeme sets). Unmapped segments are reported, not guessed.
- `MASSCOMP_INVENTORY=extended` adds 11 segments the plan's 29 lack and Romance / Germanic need
  (ʒ ɲ ʎ ts dz tʃ dʒ ç y ø ɨ; affricates set both the stop and fricative manner bits, the front
  rounded and central vowels take the three free cells of the vowel cube). The plan inventory is
  unchanged by default, so every simulation result above stands.
- NorthEuraLex (`lexibank/northeuralex`, 107 languages × 1,016 concepts, CLTS-segmented) has Latin
  and six Romance languages plus Germanic, Uralic, Basque and Turkish controls. After mapping, 789
  concepts have a single-word form in all 15 of those languages; form length averages 5.1 segments
  with 4% over 8 and none over 12 (the new cap fits; the plan's 2–6 does not). Classical Latin is the
  held-out proto-language, with the usual caveat that Romance continues the spoken accusative
  (flos → flor), so exact matches are not the right score.

**First reconstruction probe (2026-11-04) — withdrawn.** The run reported here on 11-04 (λ₃ dragged
to −1.6 and consonant-skeleton roots with λ free; 99% of root edges replaced with λ fixed) was computed
with broken kernels: numba's on-disk cache had been populated under the plan inventory, and the cached
edit DP and lattice read `EPS`, `NOINS` and `DUMMY` as compile-time constants (29) while the extended
tables put them at 40, so the deletion column, the no-insertion column and the start-of-form row were
all read from a vowel's slot. Found on 11-05 when the Rao-Blackwellised root probabilities came out
above 1 (`results/realdata/withdrawn/`). The kernels now derive the inventory size from their array
arguments (`tests/test_inventory_cache.py` checks the other inventory in a subprocess after this one
has cached); nothing else ran under the extended inventory, so no simulation result is affected, and
the cloud runners compile fresh in every job. The cost arithmetic below still describes the model;
the measured degeneracy does not.

In the model's own cost terms (prototype λ = 4.5 / 3 / 2 / 6, 40 segments): a root segment costs 3.7
nats under the flat innovation prior G; inheriting a segment unchanged costs 0.2; any substitution
4.8–6 and a deletion 6.7; an insertion 3.8 at λ₃ = 6. Romance daughters share consonant skeletons and
differ systematically in their vowels, and the context-free transducer has no cheap way to say "the
vowels changed regularly": a per-edge rule could, but the spike-and-slab prior expects three rules per
edge where a real edge carries dozens. The roadmap items — a phonotactic G, rule budgets that scale,
context-conditioned rules, λ fitted on reference families — are still the things real data need.

### Innovation prior G (2026-11-05)

`events.py` now holds G as a length table times a segment unigram: `G(y) = G_LEN[|y|] ∏ G_SEG[y_i]`.
The plan's flat G (uniform length 2–6, uniform segments, soft floor −40 − 8/segment outside) is the
default and draws from the same RNG stream as before, so every seeded dataset above is unchanged.
`set_G(len_logp, seg_logp)` installs another; `fit_G(forms)` an add-½ smoothed unigram and add-1
length histogram of a wordlist; `random_G(seed)` a Dirichlet draw for consistency tests;
`MASSCOMP_G = flat | random:<seed> | <json from save_G>` selects one per process (diag knob `g`).
The lattice takes the tables as arguments and stays exact: the segment weight is paid at emission
and the length weight at the terminal, so a unigram × length G costs nothing over the flat one
(`tests/test_phonotactic_g.py`: lattice vs brute force and the candidate Gibbs move vs enumeration
under a random non-flat G). A bigram G would need an MH correction on top of the lattice proposal;
not done.

Why the simple form first: the Romance degeneracy argument above is about the *level* of the
innovation cost (3.7 nats per segment makes "replace" dear and "short root + insertions" cheap);
a unigram over the daughters' own segment frequencies brings a typical 5-segment form from −19 to
about −13 nats, which is the correction that matters, and it does so without any claim about which
sequences are words.

**Geweke under a random non-flat G (`diag_41993479282.json`).** Eight restart cells under
`MASSCOMP_G=random:3` and `random:11` (full sampler on two seeds and 12,000 draws each; latent moves
on two seeds × ω from the prior and at 0.5; regrow at 0.3; 20,000 draws): max |z| 2.60 over ~100
statistics, the 2.6 on the 2.6%-rare borrow fraction. The lattice is exact under a unigram × length G.

**Romance probe with the corrected kernels (`diag_41995697698.json`, 2 chains × 1,500 sweeps,
extended inventory).** Three settings, two or three seeds, n = 50 and 100 concepts; the score is the
normalised edit distance (NED) of the sampled joined root to Classical Latin, against the NED of the
daughters themselves (the model has not seen Latin):

| G | λ | n, seed | NED root / daughters | NED root / daughters, cognate subset | root length ÷ Latin | t_root (2 chains) | root edges inherited (ibero / italo) |
|---|---|---|---|---|---|---|---|
| fitted | free | 100, s3 | **0.61** / 0.68 | **0.36** / 0.46 | 0.95 | 0.36 / 0.25 | 0.78 / 0.61 |
| fitted | free | 50, s1 | 0.70 / 0.71 | 0.42 / 0.47 | 0.99 | 0.12 / 0.15 | 0.92 / 0.87 |
| fitted | free | 50, s2 | **0.64** / 0.69 | **0.39** / 0.50 | 0.99 | 0.32 / 0.27 | 0.82 / 0.75 |
| fitted | pinned (sd 0.02) | 50, s1 | 0.70 / 0.71 | 0.38 / 0.47 | 0.76 | 0.26 / 1.92 | 0.81 / 0.75 |
| fitted | pinned | 50, s2 | 0.70 / 0.69 | 0.49 / 0.50 | 0.84 | 4.38 / 4.11 | 0.12 / 0.02 |
| flat | free | 50, s1 | 0.78 / 0.71 | 0.56 / 0.47 | 0.76 | 0.90 / 1.30 | 0.78 / 0.34 |
| flat | pinned | 50, s1 | 0.74 / 0.71 | 0.44 / 0.47 | 0.65 | 0.31 / 0.62 | 0.89 / 0.71 |
| flat | free | 50, s2 | 0.73 / 0.69 | 0.54 / 0.50 | 0.82 | 0.96 / 0.34 | 0.83 / 0.25 |

With G fitted to the daughters' other concepts and λ free, the joined root is closer to Latin than
any daughter is (by 0.05–0.07 NED overall, 0.05–0.11 on the cognate subset), of the right length, and
reached by inheritance on 75–92% of the family-root edges, with the two chains agreeing on t_root
(0.12–0.36) and on λ (4.4 / 5.6–7.2 / −1.3 / 7.2–7.5). Under the flat G the root is *farther* from
Latin than the daughters, 18–35% too short, and the Italo-Romance root edge is replaced two times in
three (the flat G makes "a new short word" cheap). Pinning λ at the plan's prior means is worse on
every column and on seed 2 disconnects the joined root altogether (t_root 4.2, root edges 2–12%
inherited). The fitted G with λ free is therefore the setting for real data from here; the learned
λ₂ < 0 (deletion cheaper than keeping a segment) is the model's way of absorbing the daughters'
irregular losses and is what produces union roots such as `oktombre` (Romanian's analogical *m*
kept because five deletions are cheaper than one insertion); directional costs are the remedy, not
a wider prior. As a first, cheap test of that reading, `MASSCOMP_LAM_MIN="-,-,0,-"` (diag knob
`lam_min=-+-+0+-`) truncates the prior on λ₂ at zero — deletion can be free but not cheaper than
keeping a segment — as a truncated normal (density renormalised; the random-walk move rejects
proposals below the bound; prior draws by rejection, so seeded datasets are unchanged when no bound
is set). The three fitted-G probe cells under it (`diag_42020677180.json`): λ₂ sits on the bound in
every chain (the posterior wants it negative), λ₁ rises to 6.7–8.4 to compensate, and the roots move
a little toward Latin — NED 0.600 / 0.686 / 0.639 against 0.611 / 0.695 / 0.643 free, and on the
cognate subset 0.308 / 0.418 / 0.376 against 0.356 / 0.423 / 0.386 — while getting 5–10% shorter
(length ratio 0.92 / 0.89 / 0.93). Union roots do not disappear: *twig* goes from `ramora` to
`ramor` (Romanian's *-r* kept; `ramo` is the next form), because once deletion costs the identity
bonus rather than less, a per-edge rule that deletes the segment on each daughter edge takes over.
A mild gain, not a cure: the asymmetry has to be in the direction of change (loss cheaper than gain,
lenition cheaper than fortition), not in the level of λ₂.

### Rule budget and left-context rules (2026-11-05)

Two switches, both off by default so every result above stands. `MASSCOMP_U_RULES` (default 3) is
the expected number of rules per edge under the spike-and-slab prior whatever the edge's length;
`MASSCOMP_U_PER_T=κ` makes it κ·t_e instead, scaling with time as the slab variance already does.
`MASSCOMP_U_CONTEXT=1` adds left-context-conditioned rule slots: `(s, o, ctx)` applies s → o only
after input segment ctx (N_SEG = start of form), `("ins", o, ctx)` inserts o only after ctx; the
unconditioned slots stay, so a regular change costs one rule and a conditioned one costs one rule
too. The transducer builds a (ctx, s, o) table when an edge carries conditioned substitution rules
(conditioned insertions live in the existing per-context insertion table and are exact everywhere),
and the exact edit DP, the simulator's sampler and `derive` read it. The alignment lattice keeps
proposing from the context-free tables; the proposal density of every lattice move (`_ffbs_score`)
is now the lattice's own density (`logprob_lattice`) and the target the exact edge terms, so the
moves are Metropolis-Hastings steps that remain exact Gibbs steps whenever no conditioned rule is
present. Mismatch counts for the data-informed rule birth carry the left context. Cost: +11% per
sweep on the depth-0.25 benchmark with the switch on. **Geweke restart battery with the switch on
(`diag_41998075254.json`)**: seven cells (full sampler on two seeds, with κ = 10 under `random:3`,
the rule move alone on two seeds and κ = 10, latent moves with ω from the prior and at 0.5; 12,000–
20,000 draws) — max |z| 1.67–2.60 on six of them; the seventh (full mode, seed 1) has +3.52 on the
borrow fraction (0.026 vs 0.025) with ω at +2.79 and the inherit fraction at −2.53, one correlated
excursion of the rare statistic that the README records on earlier batteries too, while the same
move set with borrowing made common (ω = 0.5, where the fraction is measured precisely) passes at 1.46.
The rerun of that cell with 20,000 draws gives max |z| 2.16 (the borrow fraction again, at +2.2),
and a fourth seed 1.97 (`diag_42017443023.json`): the 3.52 was the rare statistic's tail. The
context-rule sampler passes.
Right context is not built: the lattice resolves a dependency on the preceding segment at emission time, and a dependency
on the following one would need a state the lattice does not carry; a mirrored transducer (right
context = left context of the reversed strings) is the way to add it if the probes call for it.

### Real-data relationship tests and the evidence atlas (2026-11-05)

`studies/realdata_lbf.py` asks the tier-1 question of the prototype on NorthEuraLex: nine fixed
3-leaf families (`FAMILIES`: Ibero- and Italo-Romance, West Germanic, Nordic, Slavic, Finnic, Saami,
Turkic, Mongolic; daughter edges 0.3, binary) joined pairwise at a root of length t_root ~ Gamma(2, 1),
n random concepts with a single-word form in all six languages, G fitted to the six languages'
other concepts, the join-path estimator (K = 16 × 600 sweeps, two replicates). `join_path` now
records a trace: the power variable per concept at every stone (`join_delta_by_concept`), so the
thermodynamic integral decomposes exactly into per-concept contributions (`ti_concept`, summing to
`ti_total`; the stepping-stone total is the reported LBF), and at α = 1 the event fractions per edge,
borrow routes, t_root, ω, λ and the sampled joined-root and family-root forms. The TI total is biased on
a coarse ladder: at the bottom stone the root-edge events are drawn from their prior (a tenth of them
INHERIT from a joined root that is a random G draw), so E[Δ] there is a large negative number that the
trapezoid integrates while the stepping stone's log-mean-exp is dominated by the samples without
such events — the two totals can differ by 20 nats (Ibero–Slavic, full list: +12.2 against −6.7). The
ranking of concepts by TI is what the loanword reading above used, and it holds up; for the number,
`join_path` now also records a **leave-one-concept-out stepping stone** (`loo_concept`): on the same
samples, how much each stone's term falls when one concept's share of Δ is removed, summed over
stones. Marginal contributions, not a decomposition; the explorer shows them when present. `studies/explorer.py`
turns a battery into one self-contained page (the *evidence atlas*: a map with the tested pairs as
arcs weighted by LBF, the null band from the known-unrelated pairs, a per-pair dossier with the
annotated tree, the per-concept attribution bars and sequence logos of the reconstructed forms).

What the development runs (n = 20, K = 6 × 120, one replicate) already show, and what it means for
the design: the concepts carrying the cross-family evidence for Germanic–Nordic are *december*,
*connect*, *cup* and *sharpen*, i.e. Latin loans and Middle Low German loans in every one of the six
languages — a pairwise test on the full 1,016-concept list credits shared contact vocabulary as
inheritance, and would eventually draw a Hungarian–German arc. The remedy the comparative method
uses is the one applied here: `concepts=basic` restricts the draw to the Swadesh-100 ∪ Leipzig-Jakarta
items (135 mapped onto NorthEuraLex; 125–135 available per pair), the vocabulary borrowed least. The
full-list battery (11 pairs: three known related within a family, two Indo-European pairs at
Proto-Indo-European depth, six not known to be related, of which Turkic–Mongolic is the contested
Altaic case and is scored as unrelated; plus two flat-G controls) and the same 11 pairs on basic
vocabulary ran as `diag_42005522205.json` and `diag_42017391267.json` (K = 16 × 600, two
replicates, 50 concepts, fitted G, extended inventory; the same seed draws the same 50 words within
a concept set):

| pair | known | log BF, full list (2 replicates) | log BF, basic vocabulary (2 replicates) |
|---|---|---|---|
| Germanic – Nordic | related | +53.5 (+56.9, +50.0) | **+89.3** (+91.9, +86.8) |
| Ibero – Italo-Romance | related | +39.5 (+41.5, +37.5) | **+70.4** (+75.6, +65.3) |
| Finnic – Saami | related | +33.9 (+33.2, +34.6) | **+45.5** (+33.0, +57.9) |
| Ibero – Germanic | related, PIE depth | +16.8 (+13.1, +20.6) | +7.6 (+10.8, +4.4) |
| Ibero – Slavic | related, PIE depth | +12.2 (+15.3, +9.2) | −17.5 (−32.1, −2.9) |
| Ibero – Finnic | unrelated | −6.6 (+1.4, −14.6) | −10.0 (−5.3, −14.7) |
| Ibero – Mongolic | unrelated | −8.4 (−5.5, −11.4) | −29.4 (−16.0, −42.8) |
| Germanic – Finnic | unrelated | −9.9 (−0.3, −19.5) | −18.5 (−11.7, −25.3) |
| Turkic – Mongolic | contested (Altaic) | −12.3 (−18.9, −5.7) | −7.4 (−11.2, −3.5) |
| Ibero – Turkic | unrelated | −12.5 (−3.3, −21.8) | −8.8 (−3.1, −14.4) |
| Finnic – Turkic | unrelated | −22.0 (−12.7, −31.3) | −6.7 (−8.0, −5.4) |
| Ibero – Finnic, flat G | unrelated | +143.5 (+145.2, +141.7) | — |
| Ibero – Italo-Romance, flat G | related | +113.9 (+139.0, +88.7) | — |

Every pair known to be related within a family is far above every pair known not to be, on both
lists; every unrelated pair is negative on the mean on both lists (worst single replicate +1.4, on
the full list), and Turkic–Mongolic sits among the unrelated pairs. The flat-G controls say why the
phonotactic prior is not optional on real data: under the plan's flat G the *unrelated* Ibero–Finnic
pair scores +143, because a new word is so dear that sharing any root and editing it heavily beats
two independent draws.

The two lists disagree exactly where a linguist would expect, and the per-concept attribution says
why. On basic vocabulary the within-family evidence roughly doubles (basic vocabulary is where the
cognates are: Germanic–Nordic's top words become *blood, swim, wind, bite, wing, we*, from *creep,
price, wait, fingernail* on the full list), while the two Proto-Indo-European-depth pairs lose the
words that carried them on the full list, which are loans in every case: Ibero–Germanic's +16.8
rested on *family* (+10.9; Latin *familia* in German and English), *figure* (+5.7; Latin *figura*),
*grey* (+4.4; Romance *gris* is a Germanic loan), *north* (+3.1; Romance *norte* from Germanic),
*boat* (+1.7), and Ibero–Slavic's +12.2 on *figure* (+11.3) and *master* (+2.8) alone. What remains
on basic vocabulary is the real thing or nothing: Ibero–Germanic keeps +7.6 from *wind* (+4.1;
Latin *ventus* ~ *wind*), *salt* (+1.9), *thou* (+1.1; *tu* ~ *du*), *fish*, *nose*, *star* — Indo-
European cognates, with one loan (*mountain*, English from French) — and Ibero–Slavic keeps only
*thou* (+2.7) and *I* (+0.8) and falls to −17.5. So the prototype, on basic vocabulary, detects one
of the two PIE-depth pairs marginally (+7.6 against a null band whose six members span −6.7 to
−29.4, with replicate sds of 1–13) and misses the other, which is about what the simulator's
horizon (12.3) led us to expect; on the full list it "detects" both with borrowed vocabulary. Basic
vocabulary is the protocol from here, and the attribution trace is the audit that makes a positive
result readable.

**Two more draws of the 50 words, at K = 24 (`diag_42027079408.json`, single replicates).** The
draw of the words matters as much as the estimator at this depth:

| pair | seed 1, K = 16 (2 reps) | seed 2, K = 24 | seed 3, K = 24 | mean ± sd over draws |
|---|---|---|---|---|
| Germanic – Nordic | +89.3 | +113.6 | +95.6 | +99.5 ± 10.3 |
| Ibero – Italo-Romance | +70.4 | +91.6 | +77.1 | +79.7 ± 8.8 |
| Finnic – Saami | +45.5 | +55.9 | +33.1 | +44.8 ± 9.3 |
| Ibero – Germanic (PIE depth) | +7.6 | +3.3 | −5.1 | +1.9 ± 5.3 |
| Ibero – Slavic (PIE depth) | −17.5 | −5.2 | +10.0 | −4.2 ± 11.2 |
| six unrelated pairs | −6.7 … −29.4 | −3.0 … −25.9 | −2.9 … −28.8 | all 17 estimates ≤ −2.9 |

The within-family pairs are far above the band on every draw; the two PIE-depth pairs straddle it,
and the words that carry them when they are positive are cognates each time (*wind, salt, thou,
name, star, nose* for Germanic; *salt, new, seed, white, nose, I, thou* for Slavic — *blanco* ~
*bělъ* is the same root under two formations). So with 50 basic words the prototype's signal at
Proto-Indo-European depth is real and of the same size as the draw-to-draw variation: at the
detection threshold, not past it. The test that settles whether more words settle it — every
available basic-vocabulary item (125–135 per pair) for the three PIE-depth pairs among these
families (Ibero–Germanic, Ibero–Slavic, Germanic–Slavic) and four unrelated controls, two
replicates at K = 24 — is in the cloud.

**Latin as a donor family (`diag_42032981698.json`; K = 16, two replicates, 50 words).** Donor
families are separate trees in the forest, never joined, that every node can borrow from
(`donors=latin`; the concept draw is kept identical to the run without donors). The first battery was
withdrawn: borrowing from a donor on the root edges was available to H1 and not to H0 (whose family
roots were plain G draws), so every pair containing Ibero-Romance inflated — +60 for Ibero–Mongolic,
with half of the Ibero root forms "borrowed from Latin" at α = 1 — and the join path now keeps
donor borrowing at full strength at every α, so both hypotheses share it and only inheritance from
the joined root (and root-level contact between the two tested families) is measured. On the
corrected path, basic vocabulary, the same draw as above:

| pair | without Latin | with Latin as donor |
|---|---|---|
| Germanic – Nordic | +89.3 | +98.7 (+86.8, +110.7) |
| Ibero – Italo-Romance | +70.4 | **+13.7** (+12.9, +14.5) |
| Finnic – Saami | +45.5 | +32.4 (+0.4, +64.5) |
| Ibero – Germanic (PIE depth) | +7.6 | −1.2 (+0.4, −2.8) |
| Ibero – Slavic (PIE depth) | −17.5 | −5.1 (−4.8, −5.3) |
| Ibero – Finnic | −10.0 | −7.5 |
| Ibero – Mongolic | −29.4 | −5.7 |
| Germanic – Finnic | −18.5 | −12.3 |
| Turkic – Mongolic | −7.4 | −8.6 |
| Ibero – Turkic | −8.8 | −3.8 |
| Finnic – Turkic | −6.7 | −25.5 (−52.5, +1.5) |

Two readings. Ibero–Italo-Romance falls from +70 to +14 (and from +40 to +10 on the full list):
with Latin in the forest, "both branches took the word from Latin" explains most of what "both
inherited it from Proto-Romance" explained before, because Latin *is* the ancestor and the model has
no way to prefer descent over contact when the source is the same form — what survives is the
shared innovation (*fiʎo*, not *filius*). That is the adversarial case for donors: a donor that is
the ancestor removes the evidence, as it should under this model's definition of the hypotheses;
the right structure for Latin is in the tree, as an attested older stage (the `romlat` family, in
the cloud). Second, several donor cells have replicate sds of 12–32 nats where the same pairs
without donors had 1–13 (Finnic–Saami +0.4 / +64.5, Finnic–Turkic −52.5 / +1.5), on pairs where
Latin is irrelevant. Per-stone diagnostics on Finnic–Saami (below) show where this comes from, and
it is not specific to donors.

**The ladder on real data (per-stone diagnostics, Finnic–Saami, 50 basic words).** On the "top"
ladder (α_k = 1 − (1 − k/K)², dense at α = 1) the bottom stone is broken on real data: at α = 0 the
joined root is a free G draw and the root-edge events come from their prior, so Δ there has mean
−370 to −660 with sd 240–260 (two root edges × 50 concepts × a random inherit term), and the first
stone's log-mean-exp, over an interval of 0.23 (K = 8), 0.12 (K = 16) or 0.08 (K = 24), is estimated
from one effective sample (ESS 1.0–1.4; term −27 to −57, different on every replicate). The
simulation studies never saw this because M = 25 short forms keep Δ ten times smaller. A
**two-sided ladder** (`spacing=both`, α_k = ½(2k/K)^p mirrored at the top; `power=3`, K = 24)
puts stones at 0.0003, 0.002, 0.008, 0.02, … and keeps d·sd(Δ) of order one at the bottom: every
bottom stone then has ESS 12–25 and the bottom region sums to −12.5 and −13.0 on two seeds — a real
part of the LBF (the Occam cost of H1's root structure, the reason unrelated pairs sit below zero),
now measured instead of guessed. What remains is **hysteresis in the transition** (α ≈ 0.8–0.95),
where the family roots lock onto the joined root concept by concept and the chain, cooled from
α = 1 with 150 sweeps per stone, carries its state down: two seeds give +46 and +86 on the same
data with every bottom stone agreeing and stones 15–18 contributing 34 against 69 (ESS 1.4–7).
That is the sampler's mixing in a phase-change region, not the ladder; `direction=alt` runs the two
replicates cooled and heated so the pair brackets it, and more sweeps per stone there are the cure.
Every real-data number above was estimated on the "top" ladder. Splitting each estimate into its
bottom stone and the rest shows what that cost: the bottom stone ranges from −3 to −87 between
replicates of the same cell (Finnic–Saami with the Latin donor: −86.5 and −7.6, the rest +86.9 and
+72.1), while **the sum of stones 1 and up agrees across replicates and draws** — within a family
+40 to +133 (Ibero–Italo-Romance with Latin as donor +16 to +22), Ibero–Germanic on basic vocabulary
+6 to +22 (full list +32 to +35), Ibero–Slavic +11 to +21 (full list +20), the unrelated pairs +2 to
+12, except Turkic–Mongolic at +7 to +22. The bottom region's true value, measured on the two-sided
ladder, is about −12 for a 50-word pair and depends little on the pair (at α = 0 the dummies follow
their priors whatever the data), so the picture with the bottom region taken as −12 is: within-family
pairs +30 to +120, the two PIE-depth pairs on basic vocabulary −6 to +10, the unrelated pairs −10 to
0, and Turkic–Mongolic −5 to +10 — the contested Altaic pair resembles the Proto-Indo-European-depth
pairs, not the unrelated ones, which is what contact between proto-languages looks like to a model
with no morphology and no notion of regular correspondence beyond its per-edge rules. The
separations reported above stand; the deep pairs are at the threshold, and the protocol from here is
`spacing=both,power=3,K=24,direction=alt,ess_min=8`.

**All basic vocabulary on the two-sided ladder (`diag_42057281462.json`: 127–135 words per pair,
K = 24 cubic two-sided, 600 sweeps per stone with extension below ESS 8, one replicate cooled from
α = 1 and one heated from 0).**

| pair | known | log BF (cooled, heated) | bottom region (α ≤ 0.04) | top region (α ≥ 0.9), cooled / heated | words doing the work |
|---|---|---|---|---|---|
| Ibero – Germanic | PIE depth | **+24.1** (+31.0, +17.1) | −15.9 / −13.5 | +41 / +21 | horn, salt, wind, night, name, thou, nose, fire; mountain, round (loans) |
| Ibero – Slavic | PIE depth | **+19.1** (+29.0, +9.2) | −13.7 / −14.3 | +28 / +15 | new, salt, thou, night, full, seed, white, fly |
| Ibero – Finnic | unrelated | +7.5 (+15.5, −0.6) | −12.5 / −14.4 | +26 / +14 | who (*quién* ~ *ken*), seed, name (*nombre* ~ *nimi*), neck, woman |
| Slavic – Finnic | unrelated | +3.7 (+7.4, +0.1) | −13.7 / −13.1 | +19 / +11 | we, leaf, bark, fly, salt, name |
| Germanic – Slavic | PIE depth | −7.1 (−5.1, −9.1) | −14.2 / −16.8 | +7 / +8 | salt, two, water, nail, not |
| Ibero – Turkic | unrelated | −7.1 (−9.1, −5.1) | −15.2 / −13.3 | +6 / +7 | who, he, red |
| Germanic – Finnic | unrelated | −9.2 (−8.9, −9.4) | −13.1 / −14.3 | +4.5 / +5 | name |

The ladder now behaves: the bottom region is −12.5 to −16.8 on every pair and both directions
(the Occam cost of the join, now measured), and the whole replicate spread sits in the top region,
cooled above heated wherever something locks (hysteresis: +41 against +21, +28 against +15, +26
against +14), nothing where nothing does (Germanic–Finnic 4.5 / 5.1). With 130 words the two
Ibero-Romance pairs at Proto-Indo-European depth come out clearly positive on both directions and
on the cognates a linguist would list — *horn, salt, wind, night, name, nose, thou* against Germanic;
*new, salt, thou, night, full (lleno ~ polnyj), seed* against Slavic — while Germanic–Slavic is
missed: its cognates (*two, water, nail, salt, not*) sit behind Grimm's law, and a feature-distance
model without rules cannot see *zwei* ~ *dva*, *Nacht* ~ *noč*, *Herz* ~ *serdce*. The controls are
the other half of the story: Ibero–Finnic reaches +7.5 (cooled +15.5) and Slavic–Finnic +3.7 on
exactly the look-alikes long-range comparison has always cited — *quién* ~ *ken*, *nombre* ~ *nimi*,
*semilla* ~ *siemen* — so at 130 words the gap between a Proto-Indo-European-depth pair and a
chance-resemblance control is 10–20 nats, with a hysteresis bracket of 10–20 nats on each. Mass
comparison with this machinery finds Indo-European, and it also finds Nostratic. The rule-budget
battery (κ = 30, context rules on; the Grimm's-law test) and the Latin-in-tree battery are in the
cloud on the same protocol.

**The hysteresis is a first-order transition, not a sampling budget (Finnic–Saami, 50 words,
K = 24 two-sided, local runs).** Tripling the sweeps on the transition stones (`boost=3`) widens the
cooled/heated bracket rather than closing it (+70.7 against +23.6, from +45.9 / +33.1): with more
time at each α the cooled chain locks harder and the heated chain stays free longer, i.e. the two
directions sit in two modes — every family root locked onto the joined root, or every one free —
separated by a barrier no single-site move crosses, and the mean of the two is the estimate with
the half-width as its error. Fixing t_root at 1.0 (no t_root move) does not remove it, it moves it:
the transition shifts to α ≈ 0.5–0.8 and the bracket is +53.6 / +15.5, so the slow variable is the
latents' collective state (with λ), not the root length. What would cross the barrier is replica
exchange between the stones of the join path with the transition region dense enough for adjacent
stones' power variables to overlap, or a per-concept join path (one concept joined at a time, each
step a short ladder) that never has a collective transition; both are estimator work for the next
round. For every number here the bracket is reported and the verdicts do not depend on which end is
taken.

**The rule budget must be bounded (`diag_42057298026.json`: κ = 30 with context rules, all basic
vocabulary, two-sided ladder).** With the expected number of rules per edge at κ·t_e and t_root
free, the root edges buy rules by growing: t_root ends at 6–13 on every pair, the budget on a root
edge at 200–400 rules, and H1 fits anything — the cooled replicate of an *unrelated* pair reaches
+36 (Ibero–Finnic) and +13 (Germanic–Finnic), the heated one −40 and −46, brackets of 60–100 nats
on every cell, Germanic–Slavic at +22 / −33. Nothing can be read from it except that the plan's
scaling of the rule prior with edge length (§3.5) is unsafe on an edge whose length is itself a free
parameter. The bounded variants — context rules at the default budget of three, ten rules per edge
without context, and both — are in the cloud; the first cells (`diag_42092082810.json`, all basic
vocabulary, two-sided ladder, cooled / heated):

| pair | 3 rules (above) | 10 rules | 3 rules + context | 10 rules + context |
|---|---|---|---|---|
| Ibero – Germanic | +24.1 (+31.0, +17.1) | **+34.7** (+47.8, +21.7) | +10.1 (+10.1, +10.1) | +12.7 (+12.4, +13.0) |
| Ibero – Slavic | +19.1 (+29.0, +9.2) | **+35.5** (+47.8, +23.2) | +20.7 (+30.9, +10.4) | +18.7 (+17.5, +20.0) |
| Germanic – Slavic | −7.1 (−5.1, −9.1) | −0.0 (+2.8, −2.8) | −4.5 (−6.5, −2.6) | +0.6 (+1.7, −0.6) |
| Ibero – Finnic (control) | +7.5 (+15.5, −0.6) | +2.6 (+8.3, −3.2) | +1.0 (−0.4, +2.3) | +2.8 (−1.5, +7.2) |
| Germanic – Finnic (control) | −9.2 (−8.9, −9.4) | −6.3 (−6.3, −6.2) | −6.2 (−7.0, −5.3) | −9.3 (−9.1, −9.6) |

Ten context-free rules per edge, a bounded budget, is the setting that moves the right pairs: both
Ibero-Romance pairs at PIE depth rise to +35 (brackets +22…+48), Germanic–Slavic from −7 to 0, and
the controls stay where they were (+3 and −6; the Finnic look-alikes fall from +7.5 to +2.6).
Context rules make things worse at every budget (+10 to +19 on the Ibero pairs): the slot space is
40 times larger, so under the same expected number of rules each rule costs 3.7 nats more, and the
changes that matter here — Grimm's law, satemisation, the vowel shifts — are context-free anyway.
Germanic–Slavic ends the night at the Occam threshold under every bounded setting: the words it
finds are the cognates (*water* ~ *voda*, *salt*, *two* ~ *dva*, *stand*, *night*, *not*, *we*,
*sun*), each worth 0.5–3.5 nats, and ten of them together just pay for the join. Twenty and forty
rules per edge (`diag_42138698093.json`) do not lift it honestly: Germanic–Slavic +1.4 (+5.8, −3.0)
at twenty and +16.0 (**+41.3, −9.3**) at forty, Ibero–Germanic +48.5 (+73.9, +23.2) and +36.7
(+70.4, +3.0), while the Ibero–Finnic control goes to +12.3 (+32.6, −8.0) and +6.2 (+22.9, −10.4)
and Germanic–Finnic stays at −9 and −12. Past ten rules the cooled replicate of any pair with a few
look-alikes finds a rule-rich fit the heated one never reaches, the brackets open to 50 nats, and a
control's cooled value (+33) overtakes a deep pair's heated one — the overfitting of κ = 30 in a
milder form. Ten context-free rules per edge is therefore the bounded budget of record: brackets of
at most 25 nats, controls at or below +8, both Ibero-Romance pairs at +35, Germanic–Slavic at 0.

**Latin inside the Romance tree (`diag_42057315383.json`: `romlat` = (((spa, por), cat), lat), all
basic vocabulary, two-sided ladder, two seeds).** Romance+Latin against Germanic +11.2 (+14.6, +7.7)
and +20.4 (+25.5, +15.2); against Slavic +30.9 (+56.7, +5.1); the controls Romance+Latin–Finnic −5.7
and −6.5 and –Turkic −4.3 and −5.1. Against the same pairs without Latin (+24.1 and +19.1) this is
no gain for Germanic and a wide-bracketed gain for Slavic: an attested older stage pulls the family
root toward Latin, but with Latin as a fourth leaf at the same depth as the modern languages the
daughters pay for their distance from it, and the join still has to see *nox* ~ *noč* with a
three-rule transducer. The two κ = 30 cells of this battery (brackets of 80–100 nats) are the
overfitting of the paragraph above and are ignored. (`diag_42043900602.json`, the same cells on the "top" ladder, is
dominated by its bottom stone — Ibero–Turkic −116 and −147 on two draws — and is kept only as the
record of why the ladder changed.)

### Replica exchange, rate–time coupling, directional costs (2026-11-06)

**Replica exchange along the join path (`marglik.join_path_re`, `estimator=re`).** The K + 1 stones
run as coupled chains, one per α, and after every round of sweeps neighbouring chains exchange
states by Metropolis: the target at α is b(x)·exp(α Δ(x)) with Δ the join power variable, so a swap
of the states at α_k and α_{k+1} is accepted with probability min(1, exp(−(α_{k+1} − α_k)(Δ_{k+1} −
Δ_k))). This is the remedy for the hysteresis of a single chain walked along the ladder: a locked
state flows down, a free one up, and each stone samples both modes in their posterior proportion.
Exact on the two-family toy (`tests/test_join_path.py`); on Finnic–Saami (50 basic words, K = 24
two-sided cubic, 150 sweeps per round) two independent runs give +59.8 and +49.6 where single
chains cooled and heated gave +46 / +86 and, with three times the sweeps, +71 / +24. Swap
acceptance runs 0.1–0.4 through the transition (α ≈ 0.85–0.9) and 0.5–1.0 elsewhere; a denser
ladder there (K = 32) is the next refinement. Cost: (K + 1) sweeps per round, one warm-up at α = 1
from which every replica starts, ~600 MB per estimate for the 25 replicas (their transducer memo
tables are cleared every 50 rounds).

**Rate–time coupling (`MASSCOMP_RATE_TIME=1`, `MASSCOMP_T_REF`, default 0.3).** The plan's
transducer changes segments at the same rate on every edge (spec 12.13): only the event hazards
and the rule prior see t_e. With the switch on, every change logit — substitution, deletion,
insertion — gains log(t_e / T_REF), so that for short edges P(change) grows in proportion to the
edge length and for long ones it saturates through the softmax: a continuous-time substitution
process in the log-linear form, with the tables unchanged at t_e = T_REF (the daughters' depth in
the real-data forests, so a fitted λ keeps its meaning). The t_root move rebuilds the root-edge
tables and recomputes their form terms, which is what makes t_root identified by the amount of
change rather than only by the events. Off, nothing changes.

**Directional cost (`MASSCOMP_DIRECTIONAL=1`).** A fifth weight λ₄ ~ N(1, 1) truncated at 0 on the
consonantal strength *gained* by a substitution (`segments.STRENGTH`: voiceless stop 7, voiced stop
6, voiceless fricative 5, voiced fricative 4, nasal 3, liquid 2, glide 1, vowel 0; `FORTITION[s, s′]
= max(0, STRENGTH[s′] − STRENGTH[s])`), so that b → p costs λ₄ more than p → b and a root that is the
union of its daughters' segments stops being the cheapest story. Lenition and loss gain nothing.
The directional weight is drawn after the plan's four, so seeded datasets keep their first four.

**λ₃ truncated at 4.5 by default** (plan 12.12): 6.7% of the prior's mass, the insertion rates that
let simulated words grow without bound; datasets whose λ₃ fell below are redrawn (rejection), every
other seeded dataset is unchanged.

**Geweke under the switches (`diag_42189539865.json`, `diag_42193994219.json`).** Rate–time: full
sampler on three seeds (12,000–20,000 restart draws) max |z| 2.82 / 1.84 / 1.61 and, rerun at
20,000 draws, 2.78 / 1.61; latent moves with borrowing common 0.65. Seed 1 puts a 2.8 on a λ
component each time but on a different one (λ₀ then λ₁, with the sign flipping), which is noise,
not a biased move; the t_root move's rebuild of the root-edge tables under the coupling is correct.
Directional: full sampler 1.34 / 1.96, both switches 2.32, latents 0.59. Both pass.

**The Romance probe under the coupling (`diag_42189558636.json`).** With n = 100 both chains settle
at t_root 0.07–0.08 — the Ibero- and Italo-Romance roots a short step from the joined root, 89–92% of
root edges inherited — and the root sits at NED 0.636 from Latin against 0.611 without the
coupling (cognate subset 0.376 against 0.356): no gain for the reconstruction. With n = 50, three
chains of four fall into the other basin, t_root 6.3–6.9 with every root edge "replaced": under the
coupling a long join edge makes the root-edge transducer nearly uniform, so nothing is inherited
through it and the event prior then prefers the edge longer still. The coupling therefore makes the
two basins of the joined root (12.14) farther apart for a plain chain — what the join-path estimator
sees of that is in the protocol pairs below.

### Environment-conditioned rules (2026-11-06, branch `env-rules`)

**Why.** The replica-exchange runs showed what context-free rules cost: with ten of them per edge a
control pair with look-alikes (Ibero-Romance–Finnic) rose to +19 while Ibero-Romance–Germanic needed a
loose transducer to reach +18. A context-free rule has to be leaky — "k → h on this branch" is charged
for every k that stayed — so a real conditioned change (lenition between vowels, palatalisation before
front vowels, final devoicing) pays for only part of its evidence, while a three-word coincidence pays
in full. The comparative method's power is conditioned regularity; the model carried only the
unconditioned half. The earlier test of left-context rules (above) was not a fair one either: it
spread a single budget over 41× the slots, so every context-free rule also became 3.7 nats dearer and
the whole model got stricter — which is all that result shows.

**What the data say.** Aligning NorthEuraLex Latin to its daughters (feature-weighted edit alignment,
counts of s → o by environment on the Latin side; `scratchpad/envs/scope2.py`): intervocalic lenition is
present and branch-specific — Latin t → ð between vowels 18 times in Spanish (9 kept), k → ɣ 13 (9),
p → β 8 (6); Portuguese 19 / 8, 12 / 10, 10 / 9; Italian 0 / 40, 0 / 24, 2 / 18 — final-s loss is the
strongest conditioned pattern of all (159 words to 10 exceptions in Spanish), and Italian shows k → tʃ
before vowels (24 cases) where Spanish does not. On the 131-word basic list these drop to 1–3 cases
per rule: conditioned rules earn their cost on the fuller lists, which is where loanwords live too —
the trade-off the donor mechanism is for.

**The model.** Ten natural-class environments (`segments.ENV_NAMES`: `#_` initial, `_#` final, `V_`
after a vowel, `_V` before one, `V_V` between vowels, `C_` / `_C` after / before a consonant, `_Vfront`,
`_Vhigh`, `N_` after a nasal), read off the input segments either side of a position; several can hold
at once, so a position's environment is a bitmask (`ENV_MASK[prev, next]`). A conditioned rule
`(s, o, env)` / `("ins", o, env)` adds its weight only where the environment holds. The two kinds of
rule have separate budgets (`MASSCOMP_U_RULES` context-free, `MASSCOMP_U_ENV_RULES` conditioned,
default 0 = the plan's model), so the inclusion cost of a conditioned rule is log(16,400 / budget)
against log(1,640 / budget) for a free one: exactly log 10 ≈ 2.3 nats more, the price of naming the
environment, and no exception penalty. "k → h everywhere" therefore loses to "k → h before front
vowels" as soon as the unconditioned statement has three or four exceptions — the judgement a
historical linguist makes, as arithmetic. The transducer keeps conditioned rules sparse and builds
per-position tables for one input form on demand (`EdgeParams.pos_tables`, memoised per form), which
a position-indexed edit DP (`_dp_pos`) reads; `derive`, `sample` and the brute-force check cover the
new slots; the alignment lattice still proposes from the context-free tables and the moves correct
by Metropolis-Hastings as before; the data-informed rule birth counts a mismatch for every
environment that holds at its position. The single-segment left-context mechanism is replaced.
Tests (`tests/test_env_rules.py`): environments, conditioning and normalisation, DP against brute
force, derive against rebuild, sampling against logprob, the prior's two budgets, and a planted
intervocalic t → d in eight words the sampler recovers with weight > 2 from a rule-free start.

**First look (local, 300 sweeps × 2 chains, Romance, 100 words, ten free + ten conditioned rules).**
The rules with posterior inclusion ≥ 0.5 are ones a Romanist would recognise: Catalan final-r loss
(`r → Ø / _#`), Portuguese `o → u / _#` and `e → ɨ / C_`, Romanian `a → ɨ / C_`, Spanish `Ø → j /
_Vfront` (diphthongisation) and `s → θ`, Italian final-vowel paragoge (`Ø → e / _#`), Portuguese
`n → Ø / _C` and `s → ʃ / V_`. Lenition is not among them: without Latin in the tree the model can
reconstruct the Ibero-Romance ancestor with the lenited consonants, so the test that counts is the
200-word probe with and without conditioned rules (reconstruction distance to Latin, the rules on
the ibero and italo edges) and the Geweke battery under the switch — both dispatched on the branch
(`diag` runs 42221963790, 42221968104). The real-data protocol pairs under conditioned rules follow
once those pass.

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

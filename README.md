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
log-likelihood, ω, λ, t_root, |u| and the root forms' change rate, for comparing move mixes).
Sampler knobs (`regrow_p`, `regrow_events`, `regrow_borrow`, `ffbs_p`, `latent_moves`, `init`, …)
go in the spec and become the `MASSCOMP_*` environment variables. The account allows 20 concurrent
jobs (public repository: minutes are not metered), shared with the `study` workflow.

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
power variable grows (the top stone had an effective sample size of 3). A 5-dataset × 3-replicate
run is in the cloud (study `lbfj`).

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

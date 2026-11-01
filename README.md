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

## Study results (reduced scale, 2026-11-01)

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

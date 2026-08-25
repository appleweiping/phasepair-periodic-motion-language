# PhaseSet frozen experiment and statistics plan

**Plan ID:** `phaseset-multiperson-20260825`  
**Frozen at:** 2026-08-25 20:43:38 +08:00  
**State:** `FROZEN / NOT_EXECUTED / SEALED_TEST_CLOSED`

This is the public-safe execution census. It contains no private membership,
caption payload, endpoint, credential, checkpoint, or unaggregated result.

## 1. Fixed optimization specification

Seeds are `1729`, `2718`, and `31415`.

Base qualification uses AdamW with learning rate `2e-4`, weight decay `0.01`,
30 epochs, 5% linear warmup, cosine decay, gradient clipping at `1.0`, and an
effective global batch of 128 windows. Residual systems freeze the selected
base and use AdamW with learning rate `3e-4`, weight decay `0.01`, 20 epochs,
the same warmup/decay policy, clipping, and effective global batch. Edge-budget
bucketing changes only microbatch size and gradient accumulation.

BF16 may be used only after target-server qualification; otherwise every
comparable system uses FP32. No system receives a precision, batch, data,
caption, or checkpoint-selection advantage. Residual controls remain within
plus or minus 1% of full trainable parameter count, preferentially by sharing
one head and disabling registered inputs.

## 2. Base qualification: nine runs

| Base ID | System | Seeds | Selection input |
|---|---|---:|---|
| B0 | shared actor temporal encoder + masked actor mean | 3 | validation only |
| B1 | shared actor encoder + Set Transformer/PMA | 3 | validation only |
| B2 | four alternating temporal/set-attention layers + PMA | 3 | validation only |

The qualification metric is the mean of bidirectional validation R@1, averaged
over the three seeds and registered validation strata. An exact tie selects
fewer parameters, then lower frozen-runtime latency, then the smaller base ID.
The three seed-specific checkpoints of the winning architecture are frozen.

## 3. Final residual matrix: twenty-four runs plus base rows

| ID | System | Registered intervention |
|---:|---|---|
| 00 | qualified group base | frozen base score; no periodic increment |
| 01 | generic six tokens | equal-capacity learned group tokens without registered signal structure |
| 02 | marginal Morlet power | actor marginal periodic power only |
| 03 | mean/difference DCT | non-phase frequency-domain control |
| 04 | PhasePair pair-bag | all unordered pair tokens averaged; no incidence topology |
| 05 | coverage/missing-only | coverage and missingness inputs without phase incidence content |
| 06 | incidence-shuffled | preserve half-edge multiset, degree, and coverage while deterministically shuffling actor–edge ownership |
| 07 | phase-stripped full | retain energy and coherence; remove phase and lag |
| 08 | PhaseSet full | pair component plus actor–edge incidence topology plus phase |

Systems 01–08 each run for three seeds, producing 24 residual runs. Together
with nine base-qualification runs, the formal training census is 33 attempts.
The final score matrix contains nine systems times three seeds. A deterministic
K-only diagnostic is not a trainable run; the primary hard gallery is strictly
K-matched so it must remain at chance.

## 4. Checkpoint and test discipline

Checkpoints are selected by one fixed validation primary metric only. The test
manifest, holistic caption digest, auxiliary caption digest, and hard-gallery
digest are sealed before training. No test observation may alter a model,
checkpoint, seed, split, prompt, text normalization, candidate set, metric, or
paper claim rule. After all validation decisions and aggregate code are frozen,
one formal sealed-test evaluation is admitted.

Every failed or interrupted run receives a terminal classification and remains
in the census. Retries link to their predecessor and do not erase it. A seed is
never replaced because its score is inconvenient.

## 5. Primary and secondary metrics

The primary metric is

```text
0.5 * (T2M group R@1 + M2T any-positive R@1)
```

Also reported are bidirectional R@1/3/5/10, median rank, K=3, K=4, macro-K,
parameter count, FLOPs, peak allocated memory, throughput, and K scaling.
Full-gallery retrieval is confirmatory. Group-Hard-32 is auxiliary: negatives
match K and duration, then match actor-marginal band power, total motion energy,
and root speed, and are ranked by a frozen text-similarity rule. Hard-negative
construction is model-independent and frozen before test scoring.

The three test participant components receive leave-one-component-out
sensitivity estimates so a single social group cannot silently drive the
headline result.

## 6. Registered statistical family

H1 compares system 08 to 00 and is the sole primary hypothesis. H2–H8 compare
08 to systems 01–07 respectively. Each comparison uses paired capture-level
contributions with 100,000 frozen bootstrap resamples. Windows and captions
from one capture are not independent units. Participant components are used
for the separately reported leave-one-component-out sensitivity analysis, not
as a second bootstrap sampling level. H2–H8 form one Holm-corrected family; H1
is reported separately.

For every comparison, release artifacts record the point difference,
percentile confidence interval, raw tail probability, corrected decision where
applicable, direction, resampling seed, sample/capture/component counts, and
aggregate digest. Missing or non-finite rows invalidate the table rather than
being dropped.

Claim decisions are deterministic:

- `topology-supported` only if 08 stably beats 04 and 06;
- `phase-supported` only if 08 beats 07;
- `structure-supported` only if 08 beats 01;
- `base-improvement-only` if H1 passes but structural controls do not;
- otherwise `null-or-negative`.

## 7. Mandatory qualification suite

Before any real training, the data-free suite must cover:

- K=2 exact-positive-zero topology values and gradients, and bitwise equality
  between new full and new pair-only paths;
- unchanged legacy descriptor, swap algebra, and v0.1.0 golden behavior;
- every permutation for K=3 through K=6, plus randomized K=8/32/128/256;
- padding, isolated-versus-mixed-batch, and chunk 64/128/256/320/full
  invariance;
- gradient equivariance and pair-endpoint symmetry;
- known-delay locality and a same-pair-histogram/different-incidence witness;
- all-band-invalid, one-neighbor, local-missing, exact-zero, and mask cases;
- allocation profiling excluding dense actor-pair tensors;
- tiny end-to-end training, checkpoint, interruption, resume, gallery,
  bootstrap, table, and paper-number rendering;
- fault injection for truncation, duplicate terminals, environment/manifest
  drift, and disk exhaustion;
- staged blob, all-ref, wheel, codeload, and release-asset privacy scans.

## 8. Ordered execution graph

```text
rights + private manifests
  -> target runtime inventory
  -> Linux/CUDA/Morlet/CLIP/determinism/precision qualification
  -> synthetic forward/backward/checkpoint/resume
  -> disposable 64-capture overfit
  -> 9 base qualification attempts
  -> freeze winning base checkpoints and periodic cache
  -> 24 residual attempts
  -> freeze validation decisions
  -> one sealed-test evaluation
  -> 100k bootstrap + Holm
  -> one frozen aggregate
  -> CSV/JSON + LaTeX tables + editable figures + paper + release
```

Absent rights, server access, or a qualified runtime is an explicit `HOLD`, not
permission to substitute a dataset, fabricate a receipt, or claim completion.

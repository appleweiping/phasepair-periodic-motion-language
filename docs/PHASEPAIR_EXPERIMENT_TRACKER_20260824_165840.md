# PhasePair experiment tracker successor v7：Morlet NumPy-reduction KERNEL01

**Family ID:** `phasepair-scientific-contract-successor-v7/20260824_165840`  
**Public artifact ID:** `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
**真实状态:** `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW / AUTHORITY0 / HOLD`

No implementation、private/data/GPU test、run、checkpoint、score、result或 claim exists。

## 0. Frozen predecessor and current family

- predecessor: `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_162208.md`
- bytes/SHA-256: `9016 / f4c00da38fd624c41c27196004ebe12bc4ebdf6e3c98bb5c5e9a0e7d1317632e`
- predecessor status now: `REVISE_PREDECESSOR`
- reason: `SIGNAL_SCOPE_SENTENCE_MISSTATES_CURRENT_MORLET_DELTA`
- 162208/153008/older families/fixed aliases remain frozen/read-only；162208不是 CLEAN/current。

Current exact7：

1. `idea-stage/docs/phasepair_research_contract_20260824_165840.md`
2. `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`
3. `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
4. `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
5. `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
6. `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
7. `artifact://phasepair/PHASEPAIR_MIME_PRIMARY_EVIDENCE_RECEIPT_AND_FAMILY_MANIFEST_20260824_165840.md`

第7件self hash只能 external STOP绑定。

## 1. Family/evidence ledger

| ID | item | exact truth | status |
|---|---|---|---|
| C000 | predecessor disposition | 162208 frozen/read-only | `REVISE_PREDECESSOR` |
| C001 | successor exact7 | authored candidate; fresh byte review pending | `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW` |
| E001 | MIME official PDF | arXiv2607.22702v1; bytes4298511; SHA `3f954596c72607d0b727ab4900e4dc245ee363b8a556839919cfb2d1730150a8`; 14 pages | `EVIDENCE_CAPTURED` |
| E002 | temp/workspace PDF | temp deleted; workspace none | `PASS_RECORDED` |
| E003 | official code/checkpoint | not evidenced public at access | `NOT_PUBLICLY_EVIDENCED` |
| A000 | execution authority | none | `AUTHORITY0` |

`EVIDENCE_CAPTURED`不等于 implementation/scientific CLEAN。

## 2. Optimizer closure tracker

| ID | exact requirement | contract state | runtime state |
|---|---|---|---|
| OPT00 | full set=motion+resolved CLIP text+project w/b+logit | `SPEC_CLOSED` | `HOLD_NO_RESOLUTION` |
| OPT01 | MIME motion 89/155 groups, params35,111,936 | `ARITH_CLOSED` | `HOLD_NO_CODE` |
| OPT02 | early motion 26/43 groups, params13,014,016 | `ARITH_CLOSED` | `HOLD_NO_CODE` |
| OPT03 | late motion 53/89 groups, 142/26,017,280 | `ARITH_CLOSED` | `HOLD_NO_CODE` |
| OPT04 | resolved text exact rows/classes/D | fail closed if absent | `HOLD_RESOLVED_TEXT_INVENTORY` |
| OPT05 | exactly groups decay/no_decay; WD1e-4/0 | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| OPT06 | group order + UTF-8 name order | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| OPT07 | union full/intersection0/duplicate0 | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| OPT08 | vision/pretrained projection excluded | `SPEC_CLOSED` | `HOLD_NO_RESOLUTION` |
| OPT09 | state `2P` moments/`R` step scalars | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| OPT10 | mutants rejected | fixture set closed | `NOT_RUN` |

No numerical full-model parameter total is recorded because `D/Pcd/Pcn/Rcd/Rcn` are unresolved。Text tower不得被静默freeze。

## 3. Base dropout tracker

| ID | exact requirement | contract state | runtime state |
|---|---|---|---|
| BD00 | generator domain `phasepair-base-dropout-v1` | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD01 | H0/SplitMix64/u64/threshold/LSB/scale | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD02 | MIME/early/late sites=49/16/32 | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD03 | inventory bytes/SHA exact | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD04 | one vectorized `[AB,BA]` call/site/step | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD05 | same full H0 tuple replay invariant | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD06 | text active dropout=0 resolution proof | fail closed | `HOLD_NO_RESOLUTION` |
| BD07 | small golden independent oracles | values frozen | `NOT_RUN_IN_CODE` |
| BD08 | batch128 golden independent oracles | values frozen | `NOT_RUN_IN_CODE` |
| BD09 | eval consumption0 | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD10 | row/schedule/application-trace/9 outer receipts | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| BD11 | 9 mutant classes rejected, including world-size/gradient-checkpoint/hidden-recompute | fixture set closed | `NOT_RUN` |

Author-time oracle values are contract data, not production test PASS。Inventory bytes/SHA为 MIME `1632/aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9`、early `487/d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b`、late `1187/8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16`；one-step trace bytes/SHA为 `152/77a59a881b589b399ab6c782c5639942e4080893d6c8106b968eea102917312d`、`86/7c41f6589db05caebe80e114355c9bfd5fa749b0c92e7948a2975ed390143868`、`118/6825fd7926d4c3ddf9ccee706994da69f9a589c52b4f9bbad86d9fffd3470a8e`。Framework RNG/system-name H0/two ABBA calls/wrong threshold/site reorder/eval consumption均必须 rejected。

## 4. Preserved late/InterEdit closures

| item | exact preserved value | runtime |
|---|---|---|
| late tensor/param count | 142 / 26,017,280 | `HOLD_NO_CODE` |
| late towers/PE | two independent4-block；shared frozen PE | `HOLD_NO_CODE` |
| late MAC | `8192T^2+25436160T+524288`, ABBA×2 | `HOLD_NO_CODE` |
| late T1/T2/T300 ABBA | 51,937,280 / 102,858,752 / 16,737,304,576 | `HOLD_NO_CODE` |
| InterEdit mask | `[aL,aL,aM,aM,aH,aH]` | `HOLD_NO_CODE` |
| InterEdit Morlet reads | 0 | `HOLD_NO_CODE` |
| N masks | `000000/000011/001111/111111` | `HOLD_NO_CODE` |
| N299 bins | 20/30/60 | `HOLD_NO_CODE` |

## 5. Base run ledger（9 full +1 disposable）

| systems | seeds | count | role | status |
|---|---|---:|---|---|
| 00 MIME | all3 | 3 | paper-audited base/cache owner | `HOLD` |
| 07 TMR_STYLE_EARLY_FUSION | all3 | 3 | in-house matched control | `HOLD` |
| 08 TMR_STYLE_LATE_FUSION | all3 | 3 | in-house matched control | `HOLD` |
| 00 MIME disposable | 1729 | 1 | first64 overfit only | `HOLD` |

Nine full patterns=`phasepair-run-v2/BASE_TRAIN/<seed>/<00|07|08>`。No receipt/checkpoint/result exists。Full optimizer/dropout gates must PASS before any row leaves HOLD。

## 6. Residual run ledger（18）

| system | seeds | runs | status |
|---|---|---:|---|
| 01 GENERIC | all3 | 3 | `HOLD` |
| 02 WAMO_MARGINAL_WAVELET | all3 | 3 | `HOLD` |
| 03 INTEREDIT_MEAN_DIFFERENCE_DCT | all3 | 3 | `HOLD` |
| 04 NO_RELATION | all3 | 3 | `HOLD` |
| 05 PHASE_STRIPPED | all3 | 3 | `HOLD` |
| 06 PHASEPAIR_FULL | all3 | 3 | `HOLD` |

Run pattern=`phasepair-run-v2/RESIDUAL_HEAD_TRAIN/<seed>/<01..06>`，恰18。Base dropout v1与 head dropout v2 nonalias。

## 7. Exact sealed rows and hypotheses

For each seed outer order `1729,2718,31415`, system inner order `00..06`，共21 rows；00 fixed token N/A、01..06 fixed-token v2。Hypothesis vocabulary逐字为 `H1-H6`，secondary Holm family逐字为 `H2-H6`。No row exists，全部 `HOLD`。

| H | comparison | status |
|---|---|---|
| H1 | FULL-MIME | `NOT_RUN` |
| H2 | FULL-GENERIC | `NOT_RUN` |
| H3 | FULL-WaMo | `NOT_RUN` |
| H4 | FULL-InterEdit | `NOT_RUN` |
| H5 | FULL-no-relation | `NOT_RUN` |
| H6 | FULL-phase-stripped | `NOT_RUN` |

H2-H6五项共同 Holm；full-gallery/3seed/100000 source-cluster bootstrap继承。H6 failure优先 `NO_GO_PHASE_NOT_NEEDED`。

<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:BEGIN -->
## 7.1 Morlet exact-reduction kernel gate tracker

| ID | exact requirement | contract state | runtime state |
|---|---|---|---|
| KERNEL01 | formula + serialization SHA `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7` | `SPEC_CLOSED` | `HOLD_NO_CODE` |
| KERNEL01 | runtime/reduction=`CPython3.14.5 / NumPy2.4.6 / C-contiguous / five exact np.sum / axis0 / explicit dtype,out=None,keepdims=False / initial,where omitted` | `SPEC_CLOSED` | `NOT_RUN` |
| KERNEL01 | unit-energy bound exact `<=2^-52 = 2.220446049250313e-16` | `SPEC_CLOSED` | `NOT_RUN` |
| KERNEL01 | bands1..6 errors `[1,0,1,1,1,0] ULP` | `AUTHOR_ORACLE_RECORDED` | `NOT_RUN_IN_CODE` |
| KERNEL01-LTR-MUT | SHA `d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524`；errors `[2.5,1.5,0.5,1,1,0] ULP`；band2 DC `3.0493334034159067e-16` | `MUST_KILL_BY_DC_BOUND` | `NOT_RUN` |
| KERNEL01-MUT | rounded bound `2.22e-16` rejected by bands1/3/4/5 | `MUST_KILL` | `NOT_RUN` |

Canonical target/error/bound、zero-DC threshold、descriptor、InterEdit Morlet read0均未改变；pairwise或left-to-right prose不得替代 exact NumPy calls。Author oracle不是 implementation PASS、score或 result。
<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:END -->

## 8. STOP truth

本 tracker不会提前写 fresh CLEAN。Successor状态只能 `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW`；fresh review、authority与全部 gates前不得 promotion、实现或运行。

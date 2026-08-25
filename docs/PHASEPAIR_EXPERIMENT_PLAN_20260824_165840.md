# PhasePair experiment plan successor v7：Morlet NumPy-reduction gate

**Family ID:** `phasepair-scientific-contract-successor-v7/20260824_165840`  
**Public artifact ID:** `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
**状态:** `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW / AUTHORITY0 / HOLD / PLAN_ONLY`

## 0. Frozen predecessor and family

本文件继承 frozen predecessor全部未替换 plan steps、sealed rows、statistics与 terminal priority：

- predecessor: `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_162208.md`
- bytes/SHA-256: `9973 / fd7a2cbfb93d0afeb4dc60b907da6df3dd74a8c87196c2d10226d4ad2e901f67`
- disposition: `REVISE_PREDECESSOR`，原因=`SIGNAL_SCOPE_SENTENCE_MISSTATES_CURRENT_MORLET_DELTA`；162208 的 exact NumPy-reduction gate、其余 plan steps与 sealed rows继续按 exact content identity继承；本 successor无 plan semantics变化。

Current exact7：

1. `idea-stage/docs/phasepair_research_contract_20260824_165840.md`
2. `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`
3. `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
4. `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
5. `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
6. `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
7. `artifact://phasepair/PHASEPAIR_MIME_PRIMARY_EVIDENCE_RECEIPT_AND_FAMILY_MANIFEST_20260824_165840.md`

第7件 external STOP hash only。162208/old families/fixed aliases只读。

## 1. Unchanged sealed experiment census

```text
system order=00 MIME,01 GENERIC,02 WAMO_MARGINAL_WAVELET,
             03 INTEREDIT_MEAN_DIFFERENCE_DCT,04 NO_RELATION,
             05 PHASE_STRIPPED,06 PHASEPAIR_FULL
seeds=1729,2718,31415
sealed validation rows=21
residual training runs=18
base full qualification runs=9 (00/07/08 x3)
hypotheses=H1-H6; Holm family=H2-H6
```

No score/run exists。WaMo/InterEdit仍是 separate in-house matched controls；InterEdit mask pure-DCT paired bits且 no Morlet read。Late 142/26,017,280/MAC/ABBA与 MIME primary evidence保持132411 exact values。

## 2. Gate order amendment

执行顺序只能是：

1. fresh exact7 byte review；
2. rights/allowlist/lineage与 actor commitment gates；
3. resolved CLIP text receipt；
4. full optimizer static closure；
5. `phasepair-base-dropout-v1` static closure；
6. existing architecture/MAC/padding/InterEdit goldens；
7. disposable overfit qualification；
8. 9 full base runs与 full-gallery MIME strength gate；
9. freeze eligible MIME base/cache；
10. 18 residual runs、21 sealed rows、H1-H6 inference。

任何前项 HOLD/FAIL禁止后项。特别是 resolved text rows未 materialize时不得以 motion-only optimizer开始训练；dropout goldens未 PASS时不得用 framework RNG临时运行。

## 3. Optimizer implementation plan

三个 base model分别从 closed motion inventory与同一 CLIP resolution receipt生成 full canonical inventory。Exact motion partitions：

| model | decay tensors/numel | no_decay tensors/numel |
|---|---|---|
| MIME | 89 / 35,020,288 | 155 / 91,648 |
| early | 26 / 12,985,856 | 43 / 28,160 |
| late | 53 / 25,959,424 | 89 / 57,856 |

随后加入 CLIP text `(Rcd,Pcd)/(Rcn,Pcn)`、`text.project.weight`到 decay、`text.project.bias`与`text.logit_scale`到 no_decay。Optimizer恰两组、order `decay,no_decay`、组内 UTF-8 names升序；vision/pretrained projection excluded。

Static receipt/tests必须完成：

| ID | exact check | failure |
|---|---|---|
| OPT01 | three motion name partitions/count/numel | `OPTIMIZER_MOTION_LEDGER_FAIL` |
| OPT02 | resolved text rows same for all bases; missing=>HOLD | `RESOLVED_TEXT_INVENTORY_HOLD` |
| OPT03 | full union coverage, intersection0, duplicate0 | `OPTIMIZER_COVERAGE_FAIL` |
| OPT04 | exact group order/UTF-8 name order/WD | `OPTIMIZER_ORDER_FAIL` |
| OPT05 | vision/pretrained projection optimizer count0 | `OPTIMIZER_FROZEN_LEAK` |
| OPT06 | post-step moments=`2P`, steps=`R`, logical bytes=`8P+8R` | `OPTIMIZER_STATE_FAIL` |

Mutants motion-only142、missing project/logit、duplicate/omit、group swap、LN gamma decay、pool.query no_decay、embedding no_decay、frozen projection inclusion全部必须 rejected。Receipt含 full inventory/group/state hashes与 actual optimizer source/runtime/serialization hashes。

## 4. Base dropout implementation plan

唯一 production generator=`phasepair-base-dropout-v1`，详见 MIME successor §4。Static implementation不得复用 head-v2 domain。H0精确字段恰为 seed、epoch_index、global_optimizer_step、site_ordinal、site_name_ascii、tensor_rank、shape_dims；system/model/run/metric/result均不进入。

Production inventories：MIME49、early16、late32，binary inventory bytes/SHA逐字为：

```text
MIME  1632 / aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9
early  487 / d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b
late  1187 / 8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16
```

每 site/step一次 `[2,B,...]` mask生成，pass order `[AB,BA]`；两slice distinct elements。同一 `(seed,epoch,global_step,ordinal,name,shape)` tuple必须 replay相同；architecture-specific ordinals/names独立。Eval consumption=0。Resolved CLIP text active dropout必须为0，否则 HOLD/fresh amendment。

| ID | fixture/check | expected |
|---|---|---|
| BD01 | MIME seed1729/e0/g0/o0 `[2,2,4,3,3]` | H0 `8cea86bd9c255cb42379fa23d7f5fa776f45437e2a1142351a4e847fc8b81048`; drops18; packed `e28f5de37639f08d12606eb44efaeeb1f36b2b0a8408cfd481a2ad40712b0e28`; expanded `69a38dc3bbfd1fa19a0a9a34077f7ad22f92d0d5b671d3a4e0daca862feb1646` |
| BD02 | same site `[2,128,4,3,3]` | H0 `94bf4ff117cbf6d73c33b2803005429296f0f218fd61fad04a1dea4379b71e87`; drops943; packed `2d379e24e65e14a334ecbe8cd1e7747d63eae29786acca5b108aadaad5fb038e`; expanded `5cf16c06bd8c127660137fad206cdcbc0d0e72a2f1302b81326931c76c6bb776` |
| BD03 | early/late small+B128 | architecture-specific exact H0/packed/expanded goldens |
| BD04 | AB/BA call trace | one call/site; leading2; no expand/two calls |
| BD05 | eval trace | zero H0/mask/counter reads |
| BD06 | schedule/trace/outer receipts | every row digest + exact application trace + 9 outer pointers |

Mutants framework RNG、system-name H0、ABBA two-call/d0=1、wrong threshold、wrong packing、site reorder、eval consumption、hidden text dropout全部必须 rejected。One-step application trace bytes/SHA：MIME `152/77a59a881b589b399ab6c782c5639942e4080893d6c8106b968eea102917312d`、early `86/7c41f6589db05caebe80e114355c9bfd5fa749b0c92e7948a2975ed390143868`、late `118/6825fd7926d4c3ddf9ccee706994da69f9a589c52b4f9bbad86d9fffd3470a8e`。

## 5. Existing late/InterEdit gates remain

Late audits still require exact 142 names、26,017,280 motion params、two nonsharing towers、frozen PE、true AB/BA、MAC formula/goldens、padding/pool/swap receipts。Optimizer test必须在142 motion inventory基础上继续加入 full text trainable rows；不能把 motion audit误当 optimizer union。

InterEdit tests still require N=1/3/10/299 masks `000000/000011/001111/111111`、N299 groups20/30/60、S/D same-band mask、swap bitwise、zero-energy active、Morlet Bomb read0。WaMo可读取自己的 Morlet marginal mask；system03不可。

## 6. Nine base runs and qualification

Runs恰：

```text
phasepair-run-v2/BASE_TRAIN/<1729|2718|31415>/<00|07|08>
```

同 seed三model共用 batch/caption/crop manifests、resolved text initial bytes、optimizer schema/kwargs/global-step mapping、30 epochs、full gallery/evaluator。各model motion architecture不同但 optimizer都含同 text rows；dropout generator算法相同而 site inventories独立。

MIME gate：mean严格超过 early mean与late mean；至少2/3 seeds超过该seed stronger comparator；minimum delta>=-0.005。任何 weak comparator、subset、Hard-32、64-smoke、single seed、frozen text、unclosed optimizer/dropout都 `MIME_ADAPTATION_NOT_QUALIFIED`。

## 7. Residual/sealed inference unchanged

Base qualified后才可冻结每seed MIME base/cache，随后 systems01-06 each3 seeds=18 runs。Residual head继续自己的 `phasepair-head-dropout-v2`；base generator不得替代。Validation rows按 seed outer/system inner恰21；H1 FULL-MIME，H2 generic，H3 WaMo，H4 InterEdit，H5 no-relation，H6 phase-stripped；Holm只对H2-H6五p。Full-gallery、3seed fixed nested atoms、100000 source-cluster bootstrap与 terminal priorities全继承。

<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:BEGIN -->
## 7.1 Morlet exact-reduction `KERNEL01` plan delta

Gate-order第6项现在同时包含独立 `KERNEL01`，且必须在第7项 disposable overfit前 PASS。Oracle必须在绑定 CPython `3.14.5`/NumPy `2.4.6`上，以 signal spec列出的五次 C-contiguous exact `np.sum`重建 canonical公式，并验证 SHA `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`、unit-energy bound `<=2^-52 = 2.220446049250313e-16`及 errors `[1,0,1,1,1,0] ULP`。`initial/where`必须省略；不得声称或实现 left-to-right/pairwise替代。

两类 mutants都必须杀死：(1) explicit left-to-right reduction应给 SHA `d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524`与 errors `[2.5,1.5,0.5,1,1,0] ULP`，并由 band2 zero-DC `3.0493334034159067e-16 > 2.22e-16`拒绝，failure=`MORLET_LEFT_TO_RIGHT_REDUCTION_MUTANT_SURVIVED`；(2) decimal-bound `2.22e-16`仍由 canonical bands1/3/4/5拒绝，failure=`MORLET_UNIT_ENERGY_BOUND_MUTANT_SURVIVED`。Formula、target SHA、threshold、InterEdit Morlet read0与所有 run/statistics rows不变。当前仅 author-time oracle，production状态 `HOLD_NO_CODE / NOT_RUN`。
<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:END -->

## 8. Truthful stop

当前仅 public MIME evidence captured与 contract authoring；resolved CLIP、code、static tests、private/data/GPU runs、scores/results均未发生。Tracker只能 `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW`/HOLD/NOT_RUN，不得写 fresh CLEAN或 promotion。

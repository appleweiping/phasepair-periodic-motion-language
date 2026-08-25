# PhasePair refinement successor v7：Morlet NumPy-reduction amendment

**Family ID:** `phasepair-scientific-contract-successor-v7/20260824_165840`  
**Public artifact ID:** `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
**状态:** `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW / AUTHORITY0 / HOLD / NO_RESULT`

## 0. Frozen predecessor and scope

本文件从 frozen predecessor继续，仅修 signal §1 一句 scope summary：153008已修 exact unit-energy bound，162208的唯一科学 delta才是 canonical NumPy-reduction runtime/prose与独立 `KERNEL01` left-to-right mutant；162208其余科学正文逐byte继承：

- predecessor: `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_162208.md`
- bytes/SHA-256: `11151 / f2ada32d8d84a47491caff98d0bd2c9af0fc5dbf7adf5d41a7bf4dc01155a5cb`
- disposition: `REVISE_PREDECESSOR`，原因=`SIGNAL_SCOPE_SENTENCE_MISSTATES_CURRENT_MORLET_DELTA`；162208 reduction contract保持有效但不得当 current/CLEAN。

153008已继承并保持：late exact architecture/142/26,017,280/MAC；InterEdit pure-DCT common masks/no Morlet read；actor unconditional uniqueness；separate WaMo/InterEdit；MIME public evidence；exact `2^-52` Morlet energy bound；7 systems×3=21、18 residual、9 base、H1-H6/Holm H2-H6；narrow novelty boundary。

Current exact7：

1. `idea-stage/docs/phasepair_research_contract_20260824_165840.md`
2. `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`
3. `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
4. `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
5. `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
6. `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
7. `artifact://phasepair/PHASEPAIR_MIME_PRIMARY_EVIDENCE_RECEIPT_AND_FAMILY_MANIFEST_20260824_165840.md`

第7件无 self-hash；162208/153008/144842/134842/132411/fixed aliases只读。

## 1. QA blocker A：optimizer只覆盖 motion/单组 — CLOSED IN SPEC, RESOLUTION/IMPLEMENTATION HOLD

### Finding

132411 late spec同时说 canonical motion inventory=142 与 optimizer只有一个 group/names逐字同142。该写法遗漏实际训练必须 fine-tune 的 resolved CLIP text-transformer rows、project weight/bias与 logit scale，也把所有 bias/LN错误地置于 WD1e-4；可能产生较弱、与 MIME/early不匹配的 comparator。

### Amendment

三个base的 full trainable inventory现唯一为：各自 complete motion inventory + same resolved CLIP text-transformer trainable rows + `text.project.weight/bias` + `text.logit_scale`。Vision与 pretrained `text_projection` frozen/excluded。若 resolved rows未落，所有base HOLD；绝不改成 frozen text。

Optimizer恰两groups、固定 order `decay(wd1e-4),no_decay(wd0)`，组内 canonical names按 UTF-8 bytes升序。Weights/embeddings/queries decay；bias、LN gamma/beta、logit no_decay。Union=full trainable、intersection empty、每name恰一次。

Motion mechanical ledger：

```text
MIME  decay 89/35,020,288; no_decay155/91,648; total244/35,111,936
early decay 26/12,985,856; no_decay 43/28,160; total 69/13,014,016
late  decay 53/25,959,424; no_decay 89/57,856; total142/26,017,280
```

Late `pool.query`为 decay。加入 resolved text `(Rcd,Pcd)/(Rcn,Pcn)` 后，project weight加 decay `(1,512D)`，project bias+logit加 no_decay `(2,513)`。完整 parameter/state totals只能由 resolution receipt重算；当前没有这些数值，truthful status HOLD。

Receipt绑定 full inventory、ordered group lists、coverage/intersection/duplicates、per-group count/numel/WD、constructor/source/runtime、step0与first-step state。Mutants覆盖 motion-only142、missing project/logit、duplicate/omit、wrong group/order/WD、frozen projection leak。

## 2. QA blocker B：base dropout引用不存在 — CLOSED IN SPEC, IMPLEMENTATION HOLD

### Finding

132411只称使用 predecessor counter generator，却没有任何 predecessor实际冻结 base-training mask bytes。System/model RNG、AB/BA调用次数、site顺序与 threshold皆可变；三base的 strength comparison不可复核。

### Amendment

新增独立 domain `phasepair-base-dropout-v1`。H0恰含 seed u64、epoch index u32、global optimizer step u32、site ordinal u16、semantic site length/name、tensor rank/shape，不含 system/model/run/pass/device/distributed-rank/metric/result。用 SplitMix64 constants、`floor(2^64/10)` integer drop threshold、LSB packing、keep float32 raw `0x3f8e38e4`。

Production inventories经 canonical architecture traversal/ordinal expansion闭合，明确禁止 UTF-8/lexical sort替代：MIME49、early16、late32，hash分别：

```text
aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9
d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b
8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16
```

每 site/global step一次 vectorized `[AB,BA,...]` mask；两slice distinct elements，禁止独立两calls或expand同mask。同一完整 H0 tuple逐mask bitwise replay；architecture-specific ordinal/name独立。Eval consumption0。Text resolution必须证明 active dropout=0，否则 HOLD并另开 amendment。

MIME/early/late small与B128 goldens已由独立 oracles一致计算，production仍需重新 PASS。每 architecture×seed恰一 inner schedule与9-key outer receipt，共9；system ID只在 outer。Mutants：framework RNG、H0加system name、ABBA two calls/expand、wrong threshold、wrong packing、site reorder、eval consumption、hidden text RNG。

## 3. Preserved QA closures

- Late exact input projection/LN/frozen PE、two independent4-block towers、QKV/out/FFN/dropout/padding、queries/fusion、ABBA、initializer inventory、142/26,017,280与 MAC均保留；仅 optimizer从 motion audit扩成 full trainable audit，base dropout generator改为本 family。
- InterEdit slots `[S-low,D-low,S-mid,D-mid,S-high,D-high]` 与 `[aL,aL,aM,aM,aH,aH]`保留，Morlet read0，N masks与20/30/60 counts保留。
- Actor commitments对 `2*N_final`无条件唯一；asset065237 equal arrays exception不授予 eligibility。
- MIME paper-faithful mapping + MIME/early/late×3/full gallery strength gate保留；不允许 frozen-text或 weak/dropout-variable comparator。
- MIME/early/late均只称 in-house adaptation/control，不是 official reproduction。

## 4. Cross-document exact table

| invariant | successor value |
|---|---|
| timestamp/family | `20260824_165840`, exact7 |
| predecessor | 162208 `REVISE_PREDECESSOR` |
| successor status | `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW` |
| optimizer groups | exactly2: decay then no_decay |
| optimizer union | motion + resolved CLIP text + project w/b + logit |
| frozen exclusion | vision + pretrained text_projection |
| unresolved text | HOLD, never freeze text |
| motion group ledgers | MIME89/155; early26/43; late53/89 |
| base dropout | `phasepair-base-dropout-v1` |
| base sites | MIME49 / early16 / late32 |
| ABBA dropout | one vectorized leading2 call/site/step |
| replay law | same full H0 tuple bitwise same；architecture sites独立 |
| late | 142 / 26,017,280 / ABBA MAC exact |
| InterEdit | paired DCT masks; Morlet reads0 |
| systems/seeds/scores | 7 / 3 / 21 |
| residual/base runs | 18 / 9 |
| hypotheses/Holm | H1-H6 / H2-H6 |
| evidence | public MIME captured; code/checkpoint not evidenced |
| authority/results | 0 / none |

## 5. Novelty/falsification unchanged

这些 amendments只消除 baseline training自由度，不扩大 novelty。唯一可检验黄色仍是 qualified MIME/separate controls之上的 dyadic Morlet complex relation residual增量。即使未来通过，也不得写 broad first、causal timing、official MIME/WaMo/InterEdit reproduction。

## 6. Literal/count regression oracle

下列是 authoring-time static oracle，不是 runtime/scientific PASS：

| ID | exact assertion | expected |
|---|---|---|
| LIT01 | research keep-column header与两行 bitstring | header literal=`first32 keep`；每行长度恰32 |
| LIT02 | B128 fixture site literal与 independent H0 recompute | `blocks.00.self_a.attn`；H0=`94bf4ff117cbf6d73c33b2803005429296f0f218fd61fad04a1dea4379b71e87` |
| LIT03 | plan H0 semantic field set | 恰7项：`seed,epoch_index,global_optimizer_step,site_ordinal,site_name_ascii,tensor_rank,shape_dims` |
| LIT04 | outer receipt keys ending `_sha256` | 恰5项：`model_config,batch_manifest_set,site_inventory,schedule_receipt,application_trace` |
| LIT05 | base-dropout eligibility test range | literal `BD01-BD06`，count=6 |
| LIT06 | must-kill mutant enum | count=9；第9类明确含 world-size、gradient-checkpoint、hidden-recompute |

任一 literal/count drift、以 `same`作为 H0 site name、把 dropout inventory lexical-sort、或漏掉任一 mutant，均 `LITERAL_COUNT_REGRESSION_FAIL`。

<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:BEGIN -->
## 6.1 Inherited QA blocker C：rounded Morlet unit-energy bound — REMAINS CLOSED IN SPEC

### Finding

绑定 `151744` 的公式与 pinned SHA `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`重算得到 bands1..6 unit-energy errors `[2^-52,0,2^-52,2^-52,2^-52,0]`。旧 literal `<=2.22e-16`小于 `2^-52=2.220446049250313e-16`，因此与同一 pinned oracle冲突并错误拒绝 bands1/3/4/5。

### Inherited amendment

153008已把 unit-energy bound改为 exact binary64 `<=2^-52 = 2.220446049250313e-16`（hex `0x1.0000000000000p-52`）。本 successor不改该条款；`KERNEL01_DECIMAL_ROUNDED_BOUND_MUTANT`继续必须被 bands1/3/4/5杀死。

## 6.2 QA blocker D：left-to-right prose contradicts pinned bytes — CLOSED IN SPEC, IMPLEMENTATION HOLD

### Finding

153008 research:92、signal:81/90/93把 zero-DC correction、energy normalization与 `KERNEL01` reductions称为逐项 left-to-right，但它同时绑定的 CPython `3.14.5` / NumPy `2.4.6` target SHA `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`和 errors `[1,0,1,1,1,0] ULP`来自 C-contiguous `np.sum`。显式 scalar left-to-right产生不同 SHA `d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524`与 errors `[2.5,1.5,0.5,1,1,0] ULP`，并使 band2 zero-DC=`3.0493334034159067e-16 > 2.22e-16`。因此153008是 `REVISE_PREDECESSOR`，不得视为 current/CLEAN。

### Amendment and regression oracle

Signal authority现逐字冻结五次一维 C-contiguous `np.sum`：`sum(c)` complex128、`sum(g)` float64、`sum(abs(psi_raw)^2)` float64、`sum(psi)` complex128、`sum(abs(psi)^2)` float64；每次 `axis=0,out=None,keepdims=False`、显式 `dtype`且 `initial/where`省略，使用绑定 NumPy运行时默认 reduction order。禁止把它称 pairwise或left-to-right。Canonical target/error/bounds不变；`KERNEL01_LEFT_TO_RIGHT_REDUCTION_MUTANT`必须同时命中上述 mutant SHA/error/DC fixture并因 digest与band2 DC被杀。该 author-time closure仍是 `HOLD_NO_CODE / NO_RESULT`。
<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:END -->

## 7. STOP action

下一步仅允许 exact7 static UTF8/LF/path/hash/reference、reverse-normalized diff、optimizer/dropout/literal-count oracle两遍独立验证，然后请求 fresh QA。不得触碰 private/server/data/GPU/Git/fixed aliases；不得宣称 CLEAN或 fabricated result。

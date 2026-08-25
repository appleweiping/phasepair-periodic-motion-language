# 研究合同 successor v7：Morlet NumPy reduction 窄修复

**Family ID:** `phasepair-scientific-contract-successor-v7/20260824_165840`  
**本文件精确路径:** `idea-stage/docs/phasepair_research_contract_20260824_165840.md`  
**状态:** `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW / AUTHORITY0 / HOLD / NO_DATA / NO_GPU / NO_RESULT / NO_CLAIM`

## 0. Frozen predecessor 与 exact7

本文件是 frozen predecessor 的 delta successor；未被本 family 明确替换的科学、信号、late-fusion、InterEdit、MIME evidence 与统计条款按 exact content identity 继承：

- predecessor path: `idea-stage/docs/phasepair_research_contract_20260824_162208.md`
- predecessor bytes/SHA-256: `10579 / d2a67de489d11a565e6a616f9d9303de5472922f92b69110a4d8f2bba4080fe8`
- predecessor disposition: `REVISE_PREDECESSOR`，原因=`SIGNAL_SCOPE_SENTENCE_MISSTATES_CURRENT_MORLET_DELTA`；162208 的 canonical NumPy-reduction contract、exact `2^-52` bound、公式、target digest、inventories、goldens与 arithmetic继续按 exact content identity继承；本 successor只修 signal scope summary，不改科学语义。

Current family 精确且仅含：

1. `idea-stage/docs/phasepair_research_contract_20260824_165840.md`
2. `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`
3. `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
4. `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
5. `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
6. `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
7. `artifact://phasepair/PHASEPAIR_MIME_PRIMARY_EVIDENCE_RECEIPT_AND_FAMILY_MANIFEST_20260824_165840.md`

第7件只内嵌前六件 bytes/SHA-256；其自身 final hash 由 STOP 外部 census 绑定，不作自引用。`162208`、`153008`、`144842`、`134842`、`132411`、`130023`、`151744`、`144034` 与 fixed aliases 均为 frozen history，不是 current。

## 1. Scientific boundary 与 invariant census（unchanged）

唯一黄色、尚未成立的命题仍是：相对 paper-audited、full-gallery-strength-qualified in-house MIME adaptation，dyadic Morlet cross-spectrum/coherence-like/phase-field residual 是否提供可重复的小幅双向 retrieval 增量。Broad first/causal/official-reproduction claims 全红；WaMo-style marginal-wavelet 与 InterEdit-style mean/difference-DCT 是两个分离的 in-house matched controls。

PhasePair-local actor commitment 对全部 `2*N_final` HMAC digests 无条件全局唯一。asset065237 的 equal-array exception 不授予 PhasePair eligibility；distinct lineage+ordinal source rows 必须有 distinct digest，任何同 digest 均 `PHASEPAIR_ACTOR_COMMITMENT_DUPLICATE_FATAL`。

```text
seeds=[1729,2718,31415]
systems=[00 MIME,01 GENERIC,02 WAMO_MARGINAL_WAVELET,
         03 INTEREDIT_MEAN_DIFFERENCE_DCT,04 NO_RELATION,
         05 PHASE_STRIPPED,06 PHASEPAIR_FULL]
score rows=7*3=21; residual runs=6*3=18; base full runs=3*3=9
hypotheses=H1-H6; secondary Holm family=H2-H6
```

System03 pure-DCT paired mask仍为 `[aL,aL,aM,aM,aH,aH]`、Morlet reads=0；N=1/3/10/299 goldens仍为 `000000/000011/001111/111111`。Late comparator仍为 142 motion tensors、26,017,280 motion params、frozen PE 与 true AB/BA rerun；`MAC_ABBA(T)=2*(8192*T^2+25,436,160*T+524,288)`。

## 2. Amendment A：full-trainable optimizer 唯一闭合

`00 MIME`、`07 TMR_STYLE_EARLY_FUSION`、`08 TMR_STYLE_LATE_FUSION` 每个模型的 full trainable inventory 都必须是：

```text
closed model-specific motion inventory
+ identical resolved CLIP text-transformer trainable rows
+ text.project.weight
+ text.project.bias
+ text.logit_scale
```

Vision rows与 pretrained CLIP `text_projection` 均 frozen、optimizer 不可见。132411 的“optimizer 只有一组/仅 motion names”逐字作废；不允许为凑 142 names 而冻结 text。Resolved CLIP text inventory 尚未形成时，三种 base training 一律 `HOLD_RESOLVED_TEXT_INVENTORY`。

唯一 AdamW 恰两组，group0=`decay`、WD `1e-4`；group1=`no_decay`、WD `0`。每组内部 parameter canonical names 按 UTF-8 bytes 严格升序。所有 matrix/embedding/query/other weight 进入 decay；全部 bias、LayerNorm gamma/beta 与 `text.logit_scale` 进入 no_decay。两组 union 必须等于 full trainable inventory，intersection 为空，每名恰一次。

令 resolved CLIP text-transformer 的 decay/no_decay tensor counts 为 `Rcd/Rcn`，numel 为 `Pcd/Pcn`，hidden width 为 `D`。完整账本为：

| model | decay tensors / numel | no_decay tensors / numel | total tensors / numel |
|---|---|---|---|
| MIME | `90+Rcd / 35,020,288+Pcd+512D` | `157+Rcn / 91,648+Pcn+513` | `247+Rcd+Rcn / 35,111,936+Pcd+Pcn+512D+513` |
| early | `27+Rcd / 12,985,856+Pcd+512D` | `45+Rcn / 28,160+Pcn+513` | `72+Rcd+Rcn / 13,014,016+Pcd+Pcn+512D+513` |
| late | `54+Rcd / 25,959,424+Pcd+512D` | `91+Rcn / 57,856+Pcn+513` | `145+Rcd+Rcn / 26,017,280+Pcd+Pcn+512D+513` |

`text.project.weight[512,D]` 是 decay；`text.project.bias[512]` 与 scalar `text.logit_scale` 是 no_decay。Late `pool.query` 虽为1-D仍必须 decay。数值 `D/Pcd/Pcn/Rcd/Rcn` 只能来自 closed resolution receipt，不能估计。

## 3. Amendment B：`phasepair-base-dropout-v1`

MIME/early/late 的 motion dropout 统一使用 canonical SHA256/SplitMix64 mask generator；禁止 framework RNG。H0 只含 domain、seed、epoch index、global optimizer step、site ordinal/name与 runtime shape；不含 system/model/run/pass/device/distributed-rank、metric或 result。所有 uint64 运算逐步 mod `2^64`；drop threshold 恰 `floor(2^64/10)=1844674407370955161=0x1999999999999999`；keep bits 按 flat C-order LSB-first packing，keep 展开为 float32 `10/9` raw `0x3f8e38e4`，drop 为 `0x00000000`。

每个 semantic site 每 global optimizer step 只调用一次，tensor leading dimension 恰2且顺序 `[AB,BA]`。两 pass 是同一 scheduled tensor 的不同 elements，不是相同 mask，也绝非两次独立调用。Eval 不生成 H0、不推进 step、不消费 mask。

Production motion site inventories按 architecture traversal 固定 ordinal后恰：MIME 49、early 16、late 32；binary inventory bytes/SHA分别为 `1632/aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9`、`487/d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b`、`1187/8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16`。同一完整 H0 tuple重放必须逐字一致；三种 architecture 的 canonical names/ordinals不同，故各自独立但共用同一算法。CLIP resolution必须证明 active text-dropout site count=0；若发现任何 `0<p<1` text site，必须 HOLD并 fresh amendment，不得转用 framework RNG。

独立 Python/.NET oracle 已对生成器定义给出 contract goldens：

| seed/step/site/shape | H0 | first32 keep | drops | packed SHA-256 | expanded SHA-256 |
|---|---|---|---:|---|---|
| `1729/e0/g0/o0/blocks.00.self_a.attn/[2,2,4,3,3]` | `8cea86bd9c255cb42379fa23d7f5fa776f45437e2a1142351a4e847fc8b81048` | `11111101110111110111111111101111` | 18 | `e28f5de37639f08d12606eb44efaeeb1f36b2b0a8408cfd481a2ad40712b0e28` | `69a38dc3bbfd1fa19a0a9a34077f7ad22f92d0d5b671d3a4e0daca862feb1646` |
| `1729/e0/g0/o0/blocks.00.self_a.attn/[2,128,4,3,3]` | `94bf4ff117cbf6d73c33b2803005429296f0f218fd61fad04a1dea4379b71e87` | `11111101111111101111111111111111` | 943 | `2d379e24e65e14a334ecbe8cd1e7747d63eae29786acca5b108aadaad5fb038e` | `5cf16c06bd8c127660137fad206cdcbc0d0e72a2f1302b81326931c76c6bb776` |

每 architecture×seed恰一 inner schedule receipt与一9-key outer pointer，共9份 outer receipts；system ID只能出现在 outer pointer，绝不可流入 H0/schedule。Application-trace one-step hashes为 MIME `77a59a881b589b399ab6c782c5639942e4080893d6c8106b968eea102917312d`、early `7c41f6589db05caebe80e114355c9bfd5fa749b0c92e7948a2975ed390143868`、late `6825fd7926d4c3ddf9ccee706994da69f9a589c52b4f9bbad86d9fffd3470a8e`。强制 mutants：framework RNG、H0 注入 system name、AB/BA 两次调用、错误阈值、错误 packing、site reorder、eval consumption与 hidden text dropout；任一 accepted 即 qualification FAIL。

<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:BEGIN -->
## 3.1 Narrow amendment D：Morlet reduction runtime binding

绑定 signal 的六个 canonical Morlet kernels、构造公式、序列化顺序、SHA-256 `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`与153008已修正的 unit-energy acceptance `<=2^-52 = 2.220446049250313e-16`均不变。唯一新科学语义修复是：zero-DC correction、energy normalization、zero-DC diagnostic与 `KERNEL01` unit-energy diagnostic的 reductions不再称 left-to-right，而逐操作绑定 CPython `3.14.5` / NumPy `2.4.6` 对 C-contiguous binary64/complex128一维数组执行的 exact `np.sum`；每次固定 `axis=0`、显式 `dtype`、`out=None`、`keepdims=False`，并省略 `initial`与`where`，内部 reduction order使用该绑定运行时默认实现，不声称 pairwise或逐项顺序。

该 canonical runtime重算仍给 target SHA与 bands1..6 unit errors `[1,0,1,1,1,0] ULP`，全部通过 exact `2^-52` bound且 zero-DC仍通过原 `<=2.22e-16`。Must-kill `KERNEL01_LEFT_TO_RIGHT_REDUCTION_MUTANT`改用显式 tap-order scalar accumulators，得到 SHA `d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524`、unit errors `[2.5,1.5,0.5,1,1,0] ULP`，且 band2 zero-DC=`3.0493334034159067e-16 > 2.22e-16`；digest或任一数值未杀死该 mutant即 fail。153008 的 decimal-bound mutant也继续必须被 bands1/3/4/5杀死。该 authoring oracle不构成 implementation PASS或 result。
<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:END -->

## 4. Evidence、qualification 与 authority

Official MIME evidence仍绑定 arXiv `2607.22702v1` PDF：4,298,511 bytes、SHA-256 `3f954596c72607d0b727ab4900e4dc245ee363b8a556839919cfb2d1730150a8`、14 pages、temp/workspace PDF 均不存在；access时 official code/checkpoint未被 primary source证明公开。

Eligibility仍是 paper-faithful mapping audit加 MIME/early/late 各3 seeds、same data/text/objective/optimizer/30 epochs/full validation gallery；MIME mean严格高于 early/late各自 mean，至少2/3 seeds高于该seed较强 comparator，minimum delta `>=-0.005`。Optimizer/dropout任一 HOLD/FAIL 都禁止 base qualification。

本 family 只 author contract；没有 implementation、private data、GPU、test、checkpoint、score、result或 claim。Fresh byte-level review未给 CLEAN前只能保持 `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW`，不得 promotion fixed aliases。

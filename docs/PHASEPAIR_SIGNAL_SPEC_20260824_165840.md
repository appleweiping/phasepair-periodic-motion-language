# PhasePair signal delta-spec v7：Morlet exact NumPy reduction gate

**Family ID:** `phasepair-scientific-contract-successor-v7/20260824_165840`  
**本文件精确路径:** `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`  
**状态:** `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW / AUTHORITY0 / HOLD / SPEC_ONLY`

## 0. Frozen predecessor 与 current exact7

本文件按 exact content identity 继承 frozen signal predecessor 的 common adapter、actor HMAC、Morlet/cross fields、six heads、pure-DCT InterEdit、score/loss、goldens与 receipts：

- predecessor: `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_162208.md`
- bytes/SHA-256: `9600 / f4343ae331bf34263e06d77587a3864e0b994fad056e436fa3ecff086526f09a`
- disposition: `REVISE_PREDECESSOR`，原因=`SIGNAL_SCOPE_SENTENCE_MISSTATES_CURRENT_MORLET_DELTA`；162208 的 kernel公式、canonical NumPy-reduction runtime/prose、left-to-right mutant、序列化、target digest与 exact energy bound仍有效；仅下述 scope summary被本 successor替换。

Current exact7 paths：

1. `idea-stage/docs/phasepair_research_contract_20260824_165840.md`
2. `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`
3. `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
4. `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
5. `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
6. `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
7. `artifact://phasepair/PHASEPAIR_MIME_PRIMARY_EVIDENCE_RECEIPT_AND_FAMILY_MANIFEST_20260824_165840.md`

第7件不自哈希。162208/153008/144842/134842/132411/fixed aliases只读且不是 current。

## 1. Signal invariants remain exact

- PhasePair-local `2*N_final` actor HMAC commitments无条件唯一；bitwise-equal arrays无例外，source lineage+ordinal distinct rows若 digest相同立即 fatal。
- exact order=`00 MIME,01 GENERIC,02 WAMO_MARGINAL_WAVELET,03 INTEREDIT_MEAN_DIFFERENCE_DCT,04 NO_RELATION,05 PHASE_STRIPPED,06 PHASEPAIR_FULL`。
- seeds `[1729,2718,31415]`；21 sealed score rows；18 residual runs；9 base qualification runs；H1-H6，Holm H2-H6。
- WaMo 与 InterEdit 独立；joint marginal+S/D forbidden。
- Late comparator已继承 exact 142 motion tensors / `26,017,280` motion parameters与 AB/BA MAC；144842 amendment只补 full optimizer与base dropout，153008已修 exact `2^-52` Morlet unit-energy bound；本 successor唯一科学 delta 是下述 canonical CPython `3.14.5` / NumPy `2.4.6`、C-contiguous exact `np.sum` reduction runtime/prose与 left-to-right mutant gate。

InterEdit 唯一 inputs仍是 common crop后 finite valid `N*5` activity与 DCT group sets：

```text
J_low={m:0.5<=m*30/(2N)<1.5}
J_mid={m:1.5<=m*30/(2N)<3.0}
J_high={m:3.0<=m*30/(2N)<=6.0}
m_interedit=[aL,aL,aM,aM,aH,aH]
a_g=1 iff finite input and |J_g|>0
```

禁止读取 Morlet length/energy/cross availability；zero DCT energy不关 mask。N=1/3/10/299 masks=`000000/000011/001111/111111`，N299 bins=20/30/60；swap后 S不变、D取负且 energy/token/mask bitwise invariant。

## 2. Base optimizer is an upstream signal eligibility gate

三个 full bases 的 trainable union 都必须包括 respective complete motion inventory、同一 resolved CLIP text-transformer trainable inventory、`text.project.weight/bias` 与 `text.logit_scale`；vision/pretrained CLIP `text_projection` frozen。唯一 optimizer恰两组：`decay(wd=1e-4)` 后 `no_decay(wd=0)`，组内 UTF-8 names升序，union=all trainable、intersection empty。

Full optimizer closure的 canonical authority仅为 `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md` §2-§4。Resolved text inventory未 materialize即 `HOLD_RESOLVED_TEXT_INVENTORY`；不得仅优化 motion或把 text tower静默冻结后生成任何 signal cache。

## 3. Base dropout is an upstream cache eligibility gate

Base training所有 p=.1 motion dropout由 `phasepair-base-dropout-v1` 生成。H0 fields恰为 domain NUL、seed u64be、epoch index u32be、global optimizer step u32be、site ordinal u16be、semantic site ASCII length/name、tensor rank u8与 shape dims u32be；不含 system/model/head/run/pass/device/distributed-rank/metric/result。SplitMix64、integer threshold、LSB packing与 float32 scale由 MIME successor spec逐byte冻结。

每 site/step只生成一个 leading `[AB,BA]` 的 tensor；AB、BA masks是一个 scheduled tensor中的不同 elements，禁止两个 calls。Eval与所有 fixed-token signal production不消费 base-dropout counter。Base cache receipt必须绑定 optimizer full-inventory hash、two-group hashes、dropout site-inventory hash、schedule aggregate与 frozen checkpoint hash；任一不符不得供 systems01-06读取。

Production site census：MIME49、early16、late32；binary inventory hashes依次 `aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9`、`d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b`、`8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16`。同一完整 H0 tuple必须重放相同 mask；architecture-specific ordinal/name令不同sites独立。CLIP text active stochastic sites必须经 resolution证明为0；否则 entire base stage HOLD等待新合同。

## 4. Residual-head separation

本 amendment只替换 base-training dropout。Frozen residual head仍使用 `phasepair-head-dropout-v2`、单一 `[2,B,6,256]` site与既有 goldens；两个 generator domains绝不互换。Base dropout artifact、head dropout artifact、fixed-token cache与各head checkpoint必须 nonalias且各自 receipt。

系统01-06只有在 MIME base paper mapping、full optimizer、base dropout、3-seed/full-gallery strength gate全部 PASS 后才能训练。任何 weak/frozen-text/framework-RNG base均 `BASE_CACHE_INELIGIBLE`。

<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:BEGIN -->
## 4.1 Canonical Morlet NumPy-reduction amendment and `KERNEL01`

六频带顺序、参数与 kernel公式保持 predecessor/bound `151744` 原值：`fs=30`，`f=[0.75,1.125,1.6875,2.53125,3.796875,5.6953125]`，`L=[120,80,54,36,24,16]`，`sigma=[20,13.333333333333334,8.8888888888888893,5.9259259259259256,3.9506172839506171,2.6337448559670782]`。对每 band、`n=0..L_k-1`：

```text
u_n = n - (L_k-1)/2
g_k[n] = exp(-u_n^2 / (2 sigma_k^2))
c_k[n] = g_k[n] * exp(-i 2 pi f_k u_n / fs)
beta_k = sum_n c_k[n] / sum_n g_k[n]
psi_raw_k[n] = c_k[n] - beta_k g_k[n]
psi_k[n] = psi_raw_k[n] / sqrt(sum_n |psi_raw_k[n]|^2)
```

Kernel arrays与 reduction runtime唯一绑定为 CPython `3.14.5`、NumPy `2.4.6`。每band的 `g_k`是 shape `[L_k]`、C-contiguous `np.float64`；`c_k`、`psi_raw_k`、`psi_k`是同shape、C-contiguous `np.complex128`。Magnitude-squared arrays只能按 `q_raw_k=np.square(np.abs(psi_raw_k),dtype=np.float64)` 与 `q_unit_k=np.square(np.abs(psi_k),dtype=np.float64)` 产生，均为 C-contiguous `np.float64`；两个 ufunc的 `out`均省略。禁止 Python built-in `sum`、`math.fsum`、显式 scalar loop、`np.add.reduce`替代或 strided/transposed view。

下列五个 reductions是 byte authority；`initial`与`where`两个 keyword **不传入**（不是传 `None`），除已列参数外没有 order/reshape/flatten 参数：

```python
sum_c    = np.sum(c_k,      axis=0, dtype=np.complex128, out=None, keepdims=False)
sum_g    = np.sum(g_k,      axis=0, dtype=np.float64,    out=None, keepdims=False)
energy_0 = np.sum(q_raw_k,  axis=0, dtype=np.float64,    out=None, keepdims=False)
dc_sum   = np.sum(psi_k,    axis=0, dtype=np.complex128, out=None, keepdims=False)
energy_1 = np.sum(q_unit_k, axis=0, dtype=np.float64,    out=None, keepdims=False)
```

`beta_k=sum_c/sum_g`，`psi_k=psi_raw_k/np.sqrt(energy_0)`，zero-DC diagnostic=`np.abs(dc_sum)`，unit error=`np.abs(energy_1-np.float64(1.0))`。NumPy内部 reduction order使用上述绑定运行时对该一维 C-contiguous input的默认实现；合同不把它模糊命名为 pairwise，也不把它重解释成 left-to-right。序列化保持每 band先 `uint16_be(L_k)`，再按 tap依次 `float64_be(real),float64_be(imag)`；六band连接后 SHA-256必须等于 `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`。Zero-DC acceptance仍为 `<=2.22e-16`。

153008已冻结且本 successor保持的 unit-energy acceptance为：

```text
e_k = abs(sum_n |psi_k[n]|^2 - 1)
e_k <= 2^-52
2^-52 = 2.220446049250313e-16
binary64 hex = 0x1.0000000000000p-52
expected e_1..e_6 = [2^-52,0,2^-52,2^-52,2^-52,0]
```

`KERNEL01` 为独立、data-free CPU oracle：重建上述六kernel，逐byte验证 target digest，逐band验证 `[1,0,1,1,1,0] ULP`与 exact bound，并验证 zero-DC。Must-kill `KERNEL01_LEFT_TO_RIGHT_REDUCTION_MUTANT`把上述五次 `np.sum`换为 `n=0..L_k-1`的显式 scalar accumulator（complex项每步cast `np.complex128`；float magnitude项逐tap先执行 `np.float64(abs(complex(value))**2)`，accumulator每步cast `np.float64`）；它的 kernel SHA必须为 `d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524`，unit errors必须为 `[2.5,1.5,0.5,1,1,0] ULP`（这里 `1 ULP`固定指 `2^-52`），且 band2 `abs(complex(dc_sum))=3.0493334034159067e-16`，必须因 `>2.22e-16`被杀。若该 mutant任一输出被 canonical接受，failure=`MORLET_LEFT_TO_RIGHT_REDUCTION_MUTANT_SURVIVED`。

既有 `KERNEL01_DECIMAL_ROUNDED_BOUND_MUTANT`也保留：它只把 energy bound换回 binary64 literal `2.22e-16`（hex `0x1.ffe5ab7e8ad5ep-53`），必须由 canonical bands1/3/4/5杀死；否则 `MORLET_UNIT_ENERGY_BOUND_MUTANT_SURVIVED`。Formula、target digest、zero-DC threshold与任何 descriptor语义均不得随本修复改变。
<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:END -->

## 5. Authority hold

本文件未执行任何 producer/golden/test。当前仍 `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW`；无 private/data/GPU/result/claim/promotion authority。

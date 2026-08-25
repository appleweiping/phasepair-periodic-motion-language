# MIME adaptation delta-spec v7：inherited training closure + Morlet reduction fix

**Family ID:** `phasepair-scientific-contract-successor-v7/20260824_165840`  
**Public artifact ID:** `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
**状态:** `CANDIDATE_STOPPED_AWAITING_FRESH_REVIEW / AUTHORITY0 / HOLD / NO_IMPLEMENTATION / NO_RESULT`

## 0. Frozen predecessor、replacement scope、exact7

本文件按 exact content identity继承 frozen predecessor全部未明确替换条款，尤其 paper mapping、closed MIME/early/late architectures、InterEdit pure-DCT、full optimizer、base-dropout、qualification、counts与 evidence：

- predecessor: `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_162208.md`
- bytes/SHA-256: `21568 / 387a1964d9667ce1bb5f11f3da9af6116c18dcc21d477b7a9332f98c7960f8bd`
- disposition: `REVISE_PREDECESSOR`，原因=`SIGNAL_SCOPE_SENTENCE_MISSTATES_CURRENT_MORLET_DELTA`；162208 的 MIME architecture、optimizer/dropout、evidence、exact `2^-52`条款与 canonical NumPy-reduction `KERNEL01`均未被否定；本 successor仅重绑 current-family metadata。

153008 已继承 outer receipt SHA-field count与 eligibility BD range并修正 exact energy bound；本文件逐字继承这些 MIME semantics，唯一新 eligibility差异是绑定 signal 的 exact NumPy-reduction `KERNEL01`。

Early/MIME canonical motion inventories与 common text path继续按 frozen `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260823_151744.md`（51558 bytes；SHA-256 `9fca7d4dc307864d3e5dba75c14e1ff6569a7e805ae75e5f4273e2efead3b394`）content identity继承；该历史路径不是 current family member。

Current exact7：

1. `idea-stage/docs/phasepair_research_contract_20260824_165840.md`
2. `idea-stage/docs/PHASEPAIR_SIGNAL_SPEC_20260824_165840.md`
3. `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`
4. `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md`
5. `artifact://phasepair/PHASEPAIR_EXPERIMENT_TRACKER_20260824_165840.md`
6. `artifact://phasepair/PHASEPAIR_ROUND1_REFINEMENT_20260824_165840.md`
7. `artifact://phasepair/PHASEPAIR_MIME_PRIMARY_EVIDENCE_RECEIPT_AND_FAMILY_MANIFEST_20260824_165840.md`

第7件只内联前六件 hashes，不自哈希。162208/older families/fixed aliases严格只读。

## 1. Unchanged strength systems

Full qualification仍恰9 runs：`00 MIME`、`07 TMR_STYLE_EARLY_FUSION`、`08 TMR_STYLE_LATE_FUSION` × seeds `[1729,2718,31415]`，使用同 final rows/captions/crop、同 resolved text initialization、same objective/optimizer epochs/batches/full gallery/evaluator。

Late motion contract仍恰142 canonical tensors：two independent actor `262->512+LN+frozen PE`、4×pre-LN D512/H4/FFN2048 towers、independent pools、biased1024->512 fusion、true AB/BA rerun；motion parameters=`26,017,280`。Motion partition arithmetic：

```text
late motion decay:    53 tensors / 25,959,424 numel
late motion no_decay: 89 tensors /     57,856 numel
total:               142 tensors / 26,017,280 numel
```

`pool.query`属于 decay，即使其为1-D。Forward MAC保持：

```text
MAC_pass(T)=8192*T^2+25,436,160*T+524,288
MAC_ABBA(T)=2*MAC_pass(T)
T=1/2/300 goldens=51,937,280 / 102,858,752 / 16,737,304,576
```

## 2. Canonical full-trainable inventory

### 2.1 Namespaces and exact union

每个 model 的 full trainable set恰为下列 disjoint union：

1. respective closed motion canonical inventory：MIME `mime.*`；early `tmr.*`；late 132411 `actor_a.*`,`actor_b.*`,`fusion.*`；
2. resolution receipt中所有且仅有 `component=CLIP_TEXT_TRANSFORMER` 与 `requires_grad=true` rows，其 canonical name恰为 `text.clip.` 加 receipt冻结的 resolved state key；
3. `text.project.weight[512,D_clip_hidden]`；
4. `text.project.bias[512]`；
5. `text.logit_scale[1]`。

Vision rows必须 absent于trainable set；若 runtime不得不实例化则全部 frozen且单列。Pretrained CLIP text projection canonical row `text.pretrained_projection.weight` 强制 frozen、`requires_grad=false`、optimizer group=`NOT_APPLICABLE`，只可供 anchor/Hard-32 read-only path。它与本项目 `text.project.weight` 不是同一 tensor。

Resolution receipt必须逐 row 固定：

```text
canonical_name_utf8,resolved_state_key_utf8,component,shape_u64_array,
numel_u64,semantic_class,requires_grad,source_checkpoint_sha256,
optimizer_group
```

Rows按 canonical name UTF-8 bytes升序；mapping一对一、无前后缀猜测。`semantic_class`唯一枚举为 `MATRIX_WEIGHT/EMBEDDING_WEIGHT/QUERY_OR_OTHER_WEIGHT/BIAS/LAYERNORM_GAMMA/LAYERNORM_BETA/LOGIT_SCALE/FROZEN`。若 inventory、D hidden、mapping、frozen projection row、runtime/source digest任一未落，状态=`HOLD_RESOLVED_TEXT_INVENTORY`，不得把 text transformer冻住来绕过 HOLD。

### 2.2 Exact two-group optimizer

唯一 optimizer为一个 AdamW instance且**恰两 parameter groups**：

```text
group ordinal 0: name="decay", weight_decay=1e-4
group ordinal 1: name="no_decay", weight_decay=0
```

两组共同 kwargs仍为 `lr=1e-4,betas=(0.9,0.999),eps=1e-8,amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False`。Group顺序不能交换；每组内 canonical names严格按 UTF-8 bytes升序传入。不得使用 module registration order、dimension heuristic、framework default group或第三组。

Classification只读 `semantic_class`：所有 matrix/embedding/query/other weight进入 decay；全部 bias、LayerNorm gamma/beta、`text.logit_scale`进入 no_decay；FROZEN不得进入任一组。Exact laws：

```text
set(decay_names) union set(no_decay_names) == set(full_trainable_names)
set(decay_names) intersection set(no_decay_names) == empty
len(concat(groups)) == len(set(concat(groups))) == full_trainable_row_count
```

每个 name必须出现恰一次。特别禁止：只传 motion inventory、把 `pool.query`放 no_decay、把 token/position embedding放 no_decay、把 LN gamma放 decay、把 frozen pretrained projection或vision加入 optimizer。

### 2.3 Closed arithmetic for all three bases

三种 motion inventory的独立 mechanical partitions：

| model | motion decay tensors/numel | motion no_decay tensors/numel | motion total tensors/numel |
|---|---|---|---|
| MIME | `89 / 35,020,288` | `155 / 91,648` | `244 / 35,111,936` |
| early | `26 / 12,985,856` | `43 / 28,160` | `69 / 13,014,016` |
| late | `53 / 25,959,424` | `89 / 57,856` | `142 / 26,017,280` |

令 resolved trainable CLIP text rows的 partition为 `(Rcd,Pcd)` 与 `(Rcn,Pcn)`，且 `D=D_clip_hidden`。加入 `text.project.weight` 的1 tensor/`512D` decay，以及 project bias+logit scale的2 tensors/513 no_decay，得到：

| model | full decay | full no_decay | full total |
|---|---|---|---|
| MIME | `(90+Rcd,35,020,288+Pcd+512D)` | `(157+Rcn,91,648+Pcn+513)` | `(247+Rcd+Rcn,35,111,936+Pcd+Pcn+512D+513)` |
| early | `(27+Rcd,12,985,856+Pcd+512D)` | `(45+Rcn,28,160+Pcn+513)` | `(72+Rcd+Rcn,13,014,016+Pcd+Pcn+512D+513)` |
| late | `(54+Rcd,25,959,424+Pcd+512D)` | `(91+Rcn,57,856+Pcn+513)` | `(145+Rcd+Rcn,26,017,280+Pcd+Pcn+512D+513)` |

No numerical full total may be emitted before the resolved receipt supplies `D/Rcd/Rcn/Pcd/Pcn` and proves `Pcd+Pcn=P_resolved_CLIP_text_transformer_trainable`。

## 3. Optimizer state and receipts

Canonical optimizer receipt包含：full inventory canonical JSON bytes/SHA；ordered decay/no_decay name-list bytes/SHA；union/intersection/duplicate counts；每组 tensor count/numel/WD；constructor/source/runtime digests；step0 actual state serialization SHA；first successful step后 state SHA。

Logical post-first-step state按 group0 names后group1 names序列化；每parameter row恰含 `step_u64be`、C-order little-endian float32 `exp_avg`、`exp_avg_sq`，无 max moment（amsgrad false）。令 full `(R,P)`，则 canonical moment elements=`2P`，step scalars=`R`，logical optimizer-state bytes=`8P+8R`。Step0 backend lazy state可为空，但其实际 bytes/hash必须冻结；first-step后每 trainable row必须恰一 state row，frozen rows为0。实际 framework serializer另记录，不得把 canonical logical bytes宣称为 pickle bytes。

Mechanical tests：

- `OPT01_MOTION_PARTITION`：上表三行逐 name重算；
- `OPT02_RESOLVED_TEXT_JOIN`：absence=>HOLD；presence=>rows一对一、same across models；
- `OPT03_TWO_GROUP_COVERAGE`：union/full、intersection0、duplicates0、unclassified0；
- `OPT04_ORDER_AND_WD`：groups `decay,no_decay` 与 UTF-8 order exact；
- `OPT05_FROZEN_EXCLUSION`：vision/pretrained projection optimizer rows0；
- `OPT06_STATE_SCHEMA`：`2P` moments、`R` steps与 per-name shapes；
- mandatory rejected mutants：motion-only142、missing project/logit、duplicate row、group swap、LN gamma decay、pool.query no_decay、embedding no_decay、frozen projection included。

## 4. `phasepair-base-dropout-v1` byte contract

### 4.1 H0、SplitMix64 and application

所有 production motion dropout恰 p=`0.1`。H0 preimage唯一为：

```text
ASCII("phasepair-base-dropout-v1") || 0x00 ||
uint64_be(seed) || uint32_be(epoch_index) ||
uint32_be(global_optimizer_step) || uint16_be(site_ordinal) ||
uint16_be(len(site_name_ascii)) || site_name_ascii ||
uint8(tensor_rank) || concat_i uint32_be(shape_i)
```

`H0=SHA256(preimage)`、`base=uint64_be(H0[0:8])`。H0绝不含 system/model/run/pass/device/distributed-rank/result/metric；outer receipt的 system ID不得进入 inner schedule。`epoch_index=0..29`；global step跨epochs单调从0开始，且与 same-seed common batch manifest逐步一致。`0<=step<2^32`、ordinal/name/rank/dim均须编码无溢出、numel `<2^64`。

C-order flat `j=0..numel-1`，所有运算为 unsigned uint64、logical shift且每步 mod `2^64`：

```text
z = base + 0x9e3779b97f4a7c15*(j+1)
z = (z xor (z>>30))*0xbf58476d1ce4e5b9
z = (z xor (z>>27))*0x94d049bb133111eb
z = z xor (z>>31)
drop = 1[z < 1844674407370955161]
keep = 1-drop
```

Threshold恰 `floor(2^64/10)=0x1999999999999999`；`z=threshold-1` drop、`z=threshold` keep。禁止 float compare/ceil。Keep按 j 写 `packed[floor(j/8)] bit(j mod8)`，LSB-first，末byte unused high bits0。Expanded little-endian float32：drop bits `0x00000000`；keep=`float32_RNE(10/9)` bits `0x3f8e38e4`、LE bytes `e4388e3f`。不得再次 inverted-scale。

每site/global step恰一次 logical call，runtime shape首维 `P=2`，index0=`AB`、index1=`BA`。两slice用连续不同 j elements；禁止 d0=1两calls、共享/expand同一slice。B=`1..128`、T=`1..300`；dense padding elements也消费bits，之后再按 architecture padding-zero。World size固定1、gradient checkpointing off、fused/framework dropout off；attention/FFN显式乘外部 mask。

### 4.2 Binary site inventories

Logical axis codes恰：`P=0x01,B=0x02,H=0x03,T=0x04,D512=0x05,F2048=0x06`。Shape templates：attention=`[P,B,H,T,T]`；projection/FFN-output/fusion=`[P,B,T,D512]`；FFN-inner=`[P,B,T,F2048]`。Binary inventory唯一为：

```text
ASCII("phasepair-base-dropout-site-inventory-v1") || 0x00 || uint16_be(site_count) || rows
row = uint16_be(ordinal) || uint16_be(name_len) || name_ascii ||
      uint8(rank) || concat axis_code_bytes
```

Ordinal不是 lexical sort，而是下列 architecture traversal：

- MIME：`bb=00..03` outer，`ordinal=12*bb+q`，q依次为 `self_a.attn,self_a.proj,self_b.attn,self_b.proj,cross_a_from_b.attn,cross_a_from_b.proj,cross_b_from_a.attn,cross_b_from_a.proj,ffn_a.inner,ffn_a.ffn,ffn_b.inner,ffn_b.ffn`；names加 prefix `blocks.<bb>.`。Ordinal48=`fusion.inner`。
- early：`ordinal=4*bb+q`，q names依次 `blocks.<bb>.self.attn,self.proj,ffn.inner,ffn.ffn`。
- late：actor outer `actor_a,actor_b`，then bb，then q `self.attn,self.proj,ffn.inner,ffn.ffn`；name=`actor_<a|b>.blocks.<bb>.<qname>`；ordinal=`16*actor_index+4*bb+q`。

每个 `.attn`用 PBHTT；`.proj`与`.ffn`用 PBTD；`.inner`用 PBTF；`fusion.inner`用 PBTD。Exact inventory identities：

| architecture/system | sites | bytes | SHA-256 |
|---|---:|---:|---|
| MIME/00 | 49 | 1632 | `aa348eaeb6045a66bfb4efe435eac7d77d6c09e0a30d89fb2a36d199c06d66a9` |
| early/07 | 16 | 487 | `d2ea7874af61dcac27092a23ae3688718b0b876b9d7f630a208d3f0421e1537b` |
| late/08 | 32 | 1187 | `8da922063b717cff35abdb11402d4c4607498649992b57b98cc61996bdb6ad16` |

MIME ordinal1/3 swap mutant inventory SHA=`793983f7bbfdb23b9c59390a2c419613ab1e9247544c018a2154bb7887cb3ca8`，必须 rejected。同一完整 `(seed,epoch,step,ordinal,name,shape)` tuple永远重放相同mask；三architecture的 names/ordinals不同，故 architecture-specific sites独立而算法相同。

Pinned CLIP text resolution必须用 train-mode source+forward trace枚举 Dropout operators/probability；唯一 accepted active `0<p<1` text site count=0，p=0 rows标 `ZERO_PROBABILITY_NONCONSUMING`。若非0，三base全部 HOLD等待 fresh amendment；不得使用 framework RNG。Accepted production site counts因此恰49/16/32。

### 4.3 Row、schedule、trace and nine outer receipts

每 runtime row digest：

```text
SHA256(ASCII("phasepair-base-dropout-row-v1") || 0x00 ||
       uint32_be(epoch_index) || uint32_be(global_step) ||
       uint16_be(site_ordinal) || H0_raw32 || uint8(rank) ||
       concat uint32_be(shape_i) || uint64_be(numel) ||
       uint64_be(drop_count) || packed_sha_raw32 || expanded_sha_raw32)
```

Rows按 global-step outer、ordinal inner。Schedule SHA唯一为：

```text
SHA256(ASCII("phasepair-base-dropout-schedule-v1") || 0x00 ||
       uint64_be(seed) || uint32_be(30) || uint32_be(step_count) ||
       batch_manifest_set_sha256_raw32 || site_inventory_sha256_raw32 ||
       uint64_be(row_count) || concat row_digest_raw32)
```

`row_count=step_count*site_count`。`packed_blob_set_sha256=SHA256(ASCII("phasepair-base-dropout-packed-blob-set-v1")||0x00||uint64_be(row_count)||concat packed_sha_raw32)`，相同 row order。

Application trace exact bytes：

```text
ASCII("phasepair-base-dropout-application-trace-v1") || 0x00 ||
uint32_be(step_count) || uint16_be(site_count) ||
concat_step(uint32_be(global_step) || concat_ordinal uint16_be(ordinal))
```

One-step trace：MIME `152 bytes / 77a59a881b589b399ab6c782c5639942e4080893d6c8106b968eea102917312d`；early `86 / 7c41f6589db05caebe80e114355c9bfd5fa749b0c92e7948a2975ed390143868`；late `118 / 6825fd7926d4c3ddf9ccee706994da69f9a589c52b4f9bbad86d9fffd3470a8e`。MIME ordinal1/3 swap trace SHA=`eb3f1e8246907f43e164dfc6edca88619465f71fffae452e13d3c35ec344d08f`，必须 rejected。

每 architecture×seed恰一个 inner canonical-JSON receipt，共9。Closed keys恰：`schema,generator,seed,epoch_count,global_optimizer_step_count,batch_manifest_set_sha256,site_inventory_sha256,site_count,row_count,schedule_sha256,packed_blob_set_sha256,application_trace_sha256,dropout_probability_num,dropout_probability_den,ab_ba_pass_count,resolved_text_nonzero_dropout_site_count,framework_rng_call_count,eval_counter_call_count,status`。Literals：schema=`phasepair-base-dropout-schedule-receipt-v1`；generator=`SHA256_FIRST8_BE_SPLITMIX64_KEEP_LSB_FLOAT32_RNE_V1`；epoch_count30；p_num/den=1/10；ABBA=2；text/framework/eval counts=0；status=`PASS`。任何 runtime SHA只在真实执行后填写，当前不得捏造。

每inner另由9-key outer pointer引用；keys恰 `schema,system_id,seed,model_config_sha256,batch_manifest_set_sha256,site_inventory_sha256,schedule_receipt_sha256,application_trace_sha256,status`。Schema=`phasepair-base-dropout-outer-receipt-v1`，system IDs恰 `00/07/08`，seeds恰 `1729/2718/31415`，Cartesian order system outer then seed；五SHA小写64hex、status PASS。System ID只允许在outer，不得流入 H0/inner schedule。JSON采用151744 canonical J；outer digest另绑定 domain `phasepair-base-dropout-outer-receipt-v1`。

Eval使用 BombSchedule：不计算H0、不读mask、不推进counter，trace0。Anchor build也eval/dropout-off。

### 4.4 Independent goldens

独立 oracles必须全匹配：

| fixture | H0/base | first32 keep | drops/numel | packed SHA | expanded SHA |
|---|---|---|---:|---|---|
| MIME seed1729/e0/g0/o0 `blocks.00.self_a.attn` `[2,2,4,3,3]` | `8cea86bd9c255cb42379fa23d7f5fa776f45437e2a1142351a4e847fc8b81048 / 8cea86bd9c255cb4` | `11111101110111110111111111101111` | 18/144 | `e28f5de37639f08d12606eb44efaeeb1f36b2b0a8408cfd481a2ad40712b0e28` | `69a38dc3bbfd1fa19a0a9a34077f7ad22f92d0d5b671d3a4e0daca862feb1646` |
| MIME same B128 `[2,128,4,3,3]` | `94bf4ff117cbf6d73c33b2803005429296f0f218fd61fad04a1dea4379b71e87 / 94bf4ff117cbf6d7` | `11111101111111101111111111111111` | 943/9216 | `2d379e24e65e14a334ecbe8cd1e7747d63eae29786acca5b108aadaad5fb038e` | `5cf16c06bd8c127660137fad206cdcbc0d0e72a2f1302b81326931c76c6bb776` |
| early small `blocks.00.self.attn` | `3051926bad90bacf8f188cae87d00587fb0b102b851fb7a0e66ca47de4c1d217 / 3051926bad90bacf` | `11111111011111111111111101100100` | 17/144 | `97a5f8e7d96cf008a93eb14a35d1fd2c816966503eea7273b7ff0ad504e6c40f` | `3a99d4693584652b3cb16d901bf624e069b232af518189464ede175679673f34` |
| early B128 | `a43368d722cfd72d1cb713b9bb790170c1ccfa5fefef127a541c9e9fbc9b8e3b / a43368d722cfd72d` | `10100111111111111101110111011101` | 943/9216 | `a1cd6e3e630e396c915a32b546346cf3eaf4f056fefaa226b967f52164838265` | `67301d4cead8bf316fea2724589e2d71aa534cc64e2e613e94bdf08d30ac602d` |
| late small `actor_a.blocks.00.self.attn` | `eb5ff8b764dd63901ac7bd5978eca12080dd303700c22d5e6ffbcc95d3ea806e / eb5ff8b764dd6390` | `11111101011111111111111111101111` | 20/144 | `fc456a9bcca5c28c8a2da16c1b9d24f968fdf7edd11ffcc184085ec7eef67054` | `bdf5690c01e1b0ebec14e4fae722b0c5f257a6d41c7434b6c41d8274ced8de7f` |
| late B128 | `b1187d67b5b173580bcd251677f2dbe9c90aeab0b8968cdd218b279c10b84c76 / b1187d67b5b17358` | `10111111011011111111100111110111` | 901/9216 | `c292544341af0ecc3e358922b9e4e725386dc43e86a3058f61348dcdd1a7b02b` | `acbb7276723d661a3845e83199025b0d59ba2050f71161c2102ac52bd78b1d88` |
| MIME endian/order seed2718/e1/g44/o9 `blocks.00.ffn_a.ffn` `[2,7,5,512]` | `cb77be17ec4b6f6f4cd8812507270bb586656d888c03f91f60c32e16a2d01833 / cb77be17ec4b6f6f` | `11111111101001111111111111111100` | 3594/35840 | `b98cdfc19bc9d1f049948bbae1db748723eeddf618f77f75ab44a235ceed5b71` | `b7b82ac341c663931cf5fa87ac9418cba45fbabe7b6977367834d68d790a25b8` |

MIME small packed bytes恰18、hex=`bffbfef7bf75fff7fffefffd7dfff7fdf3bf`，expanded576 bytes。其 H0 preimage恰88 bytes；B128 preimage仍88 bytes。

### 4.5 Mandatory tests and must-kill mutants

`BD01..BD06`覆盖三architecture small+B128/nonzero goldens、inventories、row/schedule/trace、one-call ABBA、9 outer receipts、eval0。下列 mutants必须 rejected：

1. framework RNG/F.dropout(p>0) Bomb；RNG state必须 unchanged；
2. H0加 system argument/key；相同 inner tuple跨 outer labels H0不变；
3. AB/BA d0=1独立calls或expand同mask；known bad H0=`531c6fd9686da3c39ca2b0a2a878ebc257adc65c8d3b505fab4ca11cea7706e0`，packed SHA=`960f48be2fedd6a9a487c6c6c332e894561c0249510591abdbd0579bd0175d42`，expanded SHA=`d7633de8a874cf0819c6d354ac3596963d7429052ccc672538d8e049ec7e68dc`；
4. threshold ceil/float/wrong boundary；
5. MSB packing/wrong scale/double scale；
6. site reorder/ordinal drift，含上述 bad inventory/trace hashes；
7. eval BombSchedule access；
8. resolved text active dropout或 fused hidden RNG；
9. world_size>1、gradient checkpointing或 hidden recomputation改变call trace。

任一 accepted mutant=`BASE_DROPOUT_CONTRACT_FAIL`。

## 5. Qualification and claims remain gated

MIME evidence仍为 arXiv `2607.22702v1` PDF bytes4298511/SHA `3f954596c72607d0b727ab4900e4dc245ee363b8a556839919cfb2d1730150a8`/14 pages，official code/checkpoint at access未 evidenced。

Eligibility要求 paper-faithful mapping、full optimizer OPT01-OPT06、base dropout BD01-BD06+mutants、MIME/early/late ×3 seeds/full gallery。MIME mean严格超过 early与late mean，至少2/3 seeds超过该seed较强 comparator，minimum delta>=-0.005。Subset、Hard-32、64-smoke、single seed、frozen text、motion-only optimizer或 framework dropout不能替代。

System03 InterEdit仍仅 pure DCT `[aL,aL,aM,aM,aH,aH]`且 Morlet reads0。Central systems仍7×3=21；residual runs18；base runs9；H1-H6/Holm H2-H6。

<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:BEGIN -->
### 5.1 Morlet exact-reduction `KERNEL01` eligibility delta

MIME/early/late architecture、matched-control fairness、optimizer与 dropout均不变。凡后续 PhasePair signal/cache qualification读取 canonical Morlet bank，必须先通过 signal successor定义的独立 `KERNEL01`：CPython `3.14.5`/NumPy `2.4.6`、C-contiguous binary64/complex128与五次 exact `np.sum(axis=0,dtype=...,out=None,keepdims=False)`（`initial/where`省略）是唯一 reduction authority；公式和 target SHA `4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`、exact energy bound `<=2^-52`及 errors `[1,0,1,1,1,0] ULP`不变。显式 left-to-right mutant必须给 `d6ead906d84238cf64fb6009e8009ac11b16b0002f30b4294a0a373ed7fd5524` / `[2.5,1.5,0.5,1,1,0] ULP`，并由 band2 zero-DC `3.0493334034159067e-16 > 2.22e-16`杀死；旧 decimal-bound mutant仍由 bands1/3/4/5杀死。未通过即 `HOLD_MORLET_KERNEL01`，不得开始 base qualification或 residual run；该 gate不把 Morlet输入加入 InterEdit。
<!-- MORLET-NP-SUM-AMENDMENT-20260824_165840:END -->

## 6. STOP

当前无 resolved CLIP inventory、implementation、test/run/result；相应状态均 HOLD。本文档 closure不是 runtime PASS或 scientific CLEAN。Fresh exact7 review前不得 private/data/GPU/Git/fixed-alias action或 claim。

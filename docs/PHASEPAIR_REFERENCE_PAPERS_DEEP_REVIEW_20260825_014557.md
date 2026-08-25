# PhasePair five-reference deep review and visual-language gate

**Review ID:** `phasepair-five-reference-deep-review-v1/20260825_014557`  
**Project:** PhasePair periodic two-person skeleton-motion/language retrieval  
**Bound objective:** `PHASEPAIR_END_TO_END_EXECUTION_OBJECTIVE_20260825_002858.md`, objective v3, SHA-256 `4e8977ec114de8657f607dc3be0fb2027485cc6d22b3a323e88114bb12c29f29`  
**Review status:** local five-PDF review complete; broader current-literature search remains a separate pre-submission gate  
**Authority:** research review only; no training result, production authority, novelty acceptance, or submission claim  

## 1. Purpose and decision boundary

This report converts the five required local reference PDFs into operational constraints for PhasePair. It is not a generic related-work summary. It records:

1. what each paper actually solves;
2. its input, representation, learning objective, data, evaluation, and reported evidence;
3. the strongest reproducibility or causal-inference limitations visible in the supplied paper;
4. the exact overlap and non-overlap with PhasePair;
5. figure-design rules that may be learned without copying any copyrighted composition;
6. concrete experiments, controls, statistics, and failure analyses that PhasePair must retain;
7. citation metadata verified against an official proceedings or primary paper page where available.

The five PDFs alone do **not** establish novelty. PhasePair's final novelty gate must also cover the closest dyadic-motion, motion-language retrieval, frequency/wavelet, mean/difference-DCT, phase-aligned retrieval, repetition-language, and 2026-era benchmark literature. Any later nearest neighbor is added to the comparison ledger without changing the preregistered experiment family after results are seen.

## 2. Frozen local input ledger

| ID | Local PDF | Pages | Bytes | SHA-256 |
|---|---|---:|---:|---|
| R1 | `reference-artifact://phasepair/2503.17690v2.pdf` | 11 | 3,283,577 | `073e8bd9edda27926612026578318cda2c28141ea4a0473ada55dce8705119f2` |
| R2 | `reference-artifact://phasepair/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.pdf` | 11 | 4,484,508 | `18cae7971aebb66c6d3ecac163c1962b9da842923fb426c8af5df3385b739e97` |
| R3 | `reference-artifact://phasepair/Liao_LangPose_Language-Aligned_Motion_for_Robust_3D_Human_Pose_Estimation_WACV_2026_paper.pdf` | 11 | 3,213,424 | `9d54395f4dea71223074b5b56ac95530ff93aefa11fc897b30a1fc60809e19c7` |
| R4 | `reference-artifact://phasepair/NeurIPS-2025-hmvlmhuman-motion-vision-language-model-via-moe-lora-Paper-Conference.pdf` | 29 | 5,530,658 | `74be15eef3a5dbc0567ff21de8492e876d94860f6e21c3988f5db0c101ece758` |
| R5 | `reference-artifact://phasepair/Rongali_Pose2Lang3D_Distilling_3D_Reasoning_from_2D_Skeletons_via_Language_Supervision_CVPRW_2026_paper.pdf` | 10 | 675,267 | `286923f6dd06479768cd1dc1483faf62f99335e38e107ea98ed4b38c7a1612ea` |

Total textual coverage is 72 of 72 pages. No page was silently treated as an alias for another source.

## 3. Reading and visual-QA coverage

### 3.1 Textual review

All five PDFs were extracted and read from page 1 through EOF. Main paper, references, checklist, appendix, supplementary qualitative pages, metric definitions, and implementation details were included when present. Conclusions below distinguish paper-reported facts from reviewer inferences.

### 3.2 Rendered visual review

The following pages were rendered at 170 dpi and inspected as images:

- CountLLM: pages 1, 3, and 5;
- MotionMaster: pages 1, 3, and 5;
- LangPose: pages 3 and 4;
- HMVLM: pages 2, 4, and 5;
- Pose2Lang3D: pages 1 and 3.

The 13 render outputs were stored temporarily under `tmp/pdfs/recommend-review`. Poppler reported missing display-font substitutions for `Symbol` or `ArialUnicode` on some pages. The diagrams remained readable, but PhasePair must not rely on those fonts and must test its own exported figures with embedded, redistributable fonts.

## 4. R1 — CountLLM

### 4.1 Primary metadata

- Ziyu Yao, Xuxin Cheng, Zhiqi Huang, and Lei Li.
- *CountLLM: Towards Generalizable Repetitive Action Counting via Large Language Model*.
- CVPR 2025, pages 19143–19153.
- Local PDF carries arXiv identifier `2503.17690v2`, dated 29 June 2025.
- Official CVF record: `https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html`.

### 4.2 Problem and method

CountLLM addresses repetitive-action counting from RGB video. Its conceptual change is to replace a narrow density/regression head with an LLM-conditioned pipeline that consumes video features plus a structured periodicity prompt and emits a numerical count in a constrained answer format.

The method uses:

- a pretrained video encoder;
- a 12-layer periodicity transformer with 64 queries;
- a projection layer from video-side tokens to the LLM space;
- Vicuna-7B as the language model;
- LoRA with rank 16 on attention components of the video/LLM stack;
- a structured prompt carrying periodicity description, question, a decimal answer convention, and flags for incomplete boundary cycles;
- progressive training rather than one-shot task fitting.

The three reported stages are semantic alignment on WebVid-10M, periodic alignment on repetitive-action data, and instruction tuning. The paper reports 16 frames for the first stage, 32 for the later instruction stage, and eight A100 GPUs.

### 4.3 Evidence

The evaluation uses RepCount, UCFRep, and Countix with OBO and normalized MAE. It includes:

- in-domain comparisons against regression and counting systems;
- RepCount-to-UCFRep/Countix cross-dataset transfer;
- qualitative out-of-distribution examples;
- ablations for the structured prompt, progressive stages, periodicity transformer, and LoRA.

This is a useful evidence pattern because the main claim is not supported by a single in-domain accuracy table. The paper explicitly tests whether its periodic-language interface survives distribution shift.

### 4.4 Scientific and reproducibility limits

- The task is scalar counting, not motion-language retrieval or interaction understanding.
- Input is RGB video rather than two-person skeleton sequences.
- Periodicity is externally described by prompt structure; the method does not expose a dyadic `AB`/`BA` relation graph, symmetric/directional slots, or a phase-sensitive two-person representation.
- The paper does not report a multi-seed protocol, confidence intervals, paired tests, or a corrected hypothesis family.
- Some baselines use reported values and others are reimplemented, so equivalence of optimization and capacity is not fully closed.
- Large-model and data-scale effects are partially entangled with the periodic-token design.

### 4.5 Exact PhasePair boundary

CountLLM is a direct warning against claiming novelty merely from “language plus periodic motion.” It already shows that explicit periodic language prompts and learned periodic tokens can improve generalization. PhasePair must therefore claim the narrower contribution: a two-person skeleton retrieval representation that jointly preserves ordered `AB`/`BA`, symmetric and directional interaction signals, multi-frequency/phase relations, six fixed relation slots, and source-cluster paired inference.

Falsifiable distinction: if a CountLLM-style periodic prompt attached to a matched skeleton encoder equals PhasePair-full on both retrieval directions under the frozen three-seed protocol, the proposed PhasePair mechanism has not earned an independent empirical claim.

### 4.6 Visual-language review

Strengths to adopt:

- page 1 establishes existing pipeline versus proposed pipeline in seconds;
- page 3 uses one left-to-right method flow with consistent module colors;
- page 5 reuses those colors across the three training stages;
- the prompt is placed close to the component it conditions.

Elements to reject:

- handwritten labels, emoji-like fire/snowflake markers, or playful clip-art semantics;
- dense descriptive text inside the main motivation figure;
- visual dependence on tiny prompt strings at final paper size.

For PhasePair, the analogous introduction figure should show the same two-person skeleton input flowing through a generic encoder failure case and a PhasePair relation-aware success case, with no detailed optimizer or dropout machinery.

## 5. R2 — MotionMaster

### 5.1 Primary metadata

- Nan Jiang, Yunhao Li, Lexi Pang, Zimo He, Siyuan Huang, and Yixin Zhu.
- *MotionMaster: Generalizable Text-Driven Motion Generation and Editing*.
- CVPR 2026, pages 30629–30639.
- Official CVF record: `https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.html`.

### 5.2 Problem and method

MotionMaster unifies text-to-motion generation and text-guided editing. Its main contribution is the combination of a large, systematically augmented motion corpus, a finite-scalar-quantization tokenizer, and a pretrained multimodal LLM whose language vocabulary is partially repurposed for motion tokens.

The paper describes:

- MotionGB, expanded from roughly 400 hours of verified motion capture to about 10,000 hours through temporal concatenation, body-part concatenation, and 24 parametric edit families;
- multi-level language descriptions generated with Gemini and manually validated data processes;
- localized SMPL-X joint features with global trajectory information;
- a four-layer convolutional FSQ tokenizer with temporal compression;
- coarse-to-fine inverse kinematics using VPoser and a rotation recovery step;
- Qwen2.5-VL with motion tokens mapped onto low-use text-token slots;
- separate rotary-position counters for motion and text;
- semantic-density weighting based on T5 features to prevent frequent semantics from dominating.

### 5.3 Evidence

The paper evaluates generation and editing with a 90/10 MotionGB split and a 400-motion test-lite subset. It compares to systems such as T2M-GPT, MoMask, MMM, MotionMillion, MotionLab, and MotionFix. Reported evidence includes:

- local/global rotation, position, and velocity errors;
- retrieval precision and diversity for generation;
- Gemini semantic and physical scores, with a reported human-agreement correlation;
- qualitative compositional generation and body-part editing;
- ablations over data scale, model size, semantic balancing, and joint versus single-task training.

### 5.4 Scientific and reproducibility limits

- Generation/editing is not retrieval; success can be judged by different failure modes and metrics.
- The large augmented corpus is inseparable from the model contribution unless data-matched controls are strict.
- A learned or LLM-based judge needs calibration, raw prompts, sampling configuration, and preferably human cross-checks.
- The supplied paper contains unresolved supplementary cross-reference placeholders in several places and does not fully specify optimizer, duration, hardware, or every training setting in the main body.
- No three-seed uncertainty or paired statistical test is reported in the supplied main paper.

### 5.5 Exact PhasePair boundary

MotionMaster demonstrates that a strong language prior, large data scale, and motion tokenization can produce impressive compositional behavior. It does not model periodic two-person relations, retrieval symmetry/direction, or paired source-cluster inference. PhasePair must not attribute a gain to periodic relations if the same gain could come from data volume, token budget, backbone scale, or language-model capacity.

Operational consequence: every PhasePair residual head must use the same frozen motion/text backbone, captions, split, optimizer budget, seed set, selection rule, and evaluator. WaMo-style and InterEdit-style controls remain separate. A joint “frequency baseline” would not isolate the mechanism.

### 5.6 Visual-language review

Strengths to adopt:

- page 1 uses a consistent qualitative grid and separates generation from editing by color and geometry;
- page 3 presents data construction as three bounded columns with a restrained pastel palette;
- page 5 uses an overview/detail split: tokenizer mechanics on the left, system behavior on the right;
- pose sequences are real, aligned, and captioned instead of represented by generic icons.

Elements to reject:

- too many examples in the introduction figure when a single causal contrast would suffice;
- captions or callouts whose meaning depends on color alone;
- copying any exact body sequence or panel composition.

For PhasePair, use one authentic two-person skeleton sequence and one concise text query in the motivation panel, then reserve complete signal decomposition for the method figure.

## 6. R3 — LangPose

### 6.1 Primary metadata

- Longyun Liao and Rong Zheng.
- *LangPose: Language-Aligned Motion for Robust 3D Human Pose Estimation*.
- WACV 2026, pages 8073–8083.
- Official CVF record: `https://openaccess.thecvf.com/content/WACV2026/html/Liao_LangPose_Language-Aligned_Motion_for_Robust_3D_Human_Pose_Estimation_WACV_2026_paper.html`.

### 6.2 Problem and method

LangPose uses language semantics as a robustness prior for 2D-to-3D pose lifting, especially under occlusion and high-dynamic motion. It has a two-stage design:

1. pretrain a pose encoder and a text encoder with motion-text alignment plus reconstruction from corrupted 2D poses;
2. fine-tune the pose branch for downstream 3D pose estimation without requiring action labels.

The pose branch uses five MotionBERT layers and the text branch uses three BERT layers. Fine-grained BABEL/AMASS action descriptions supply language supervision. Corruptions include joint/frame masking, body-part masking, temporal-window masking, and noise. The paper combines a modified contrastive objective with 3D position and velocity reconstruction losses.

### 6.3 Evidence

The evaluation covers Human3.6M and MPI-INF-3DHP, with both ground-truth and detected 2D inputs where applicable. Metrics include MPJPE, P-MPJPE/H-MPJPE, PCK, and AUC. The paper reports:

- standard subject splits on Human3.6M;
- single-H100 training, batch 16, sequence length 243, and 300 epochs;
- comparisons to pose-lifting baselines;
- ablations for semantic alignment and masking choices;
- t-SNE/qualitative analyses under semantic grouping and corruption.

### 6.4 Scientific and reproducibility limits

- The target is 3D coordinate accuracy, not motion-text retrieval.
- Language labels function as an auxiliary prior and are not queried at test time in the same way as PhasePair's text retrieval branch.
- The main paper does not fully pin learning rate, seed policy, all optimizer fields, multi-run uncertainty, or significance tests.
- Action-label coverage and domain mismatch are acknowledged limitations; fine-grained labels can encode dataset artifacts.
- A t-SNE visualization is descriptive, not causal evidence of a superior aligned space.

### 6.5 Exact PhasePair boundary

LangPose closes off a broad “language improves skeleton motion” claim. PhasePair's distinct claim must be about retrieval-relevant periodic interaction structure, not about the mere use of text alignment or masked skeleton pretraining.

Falsifiable distinction: a matched generic relation residual and a matched no-relation residual must establish whether PhasePair's ordered periodic relations add information beyond language-aligned motion embeddings. The comparison is H2 and H5, not a qualitative embedding plot.

### 6.6 Visual-language review

Strengths to adopt:

- page 3 uses two stable branch colors and minimal labels;
- the pose/text alignment link is visually secondary to the main branches;
- page 4 shows body-part masks directly on a skeleton and temporal masks directly on a strip, reducing explanatory prose;
- the diagrams are space-efficient and maintain a clear overview/detail hierarchy.

Elements to reject:

- labels that become illegible after two-column reduction;
- color-only distinctions without text or shape redundancy;
- a method overview that omits the inference-time path.

PhasePair should borrow the direct “show the mask on the data” principle: display the six relation slots and validity masks on a small AB/BA signal strip rather than describing them in a paragraph inside the figure.

## 7. R4 — HMVLM

### 7.1 Primary metadata

- Lei Hu, Yongjing Ye, and Shihong Xia.
- Published title is spelled *HMVLM: Human Motion-Vision-Lanuage Model via MoE LoRA* in the official PDF.
- NeurIPS 2025, 39th Conference on Neural Information Processing Systems.
- Official proceedings PDF: `https://proceedings.neurips.cc/paper_files/paper/2025/file/8cb564df771e9eacbfe9d72bd46a24a9-Paper-Conference.pdf`.

### 7.2 Problem and method

HMVLM integrates text-to-motion, pose estimation, and motion-video understanding into a foundation language model while attempting to preserve general dialogue knowledge. It combines two ideas:

- a five-expert LoRA mixture, including a non-trainable zero expert selected for motion-unrelated prompts;
- body-part-specific spatial tokenization with separate codebooks, followed by temporal compression for motion sequences.

The gating network is a two-layer MLP with hidden dimension 512 over 512-dimensional CLIP text features. The primary model is Vicuna-7B-v1.5, with rank-8 LoRA experts applied to all linear modules. The zero-expert supervision encourages the pretrained path for general dialogue. Pose and motion vocabularies are appended to the text vocabulary for autoregressive instruction tuning.

The tokenizer uses five body parts, 512-dimensional part embeddings, codebook size 512 per body part, and temporal compression ratio four. Appendix D reports AdamW with `[0.9, 0.99]`, tokenizer learning rate `2e-4`, commitment coefficient `0.02`, and instruction-tuning learning rate `3e-3` with cosine annealing. Joint instruction tuning uses one A800 80 GB GPU for about 120 hours, batch size 32, and micro-batch size 2.

### 7.3 Evidence

The evidence spans three tasks:

- knowledge preservation on MT-Bench before and after text-to-motion tuning, using GPT-4 as judge;
- text-to-motion on HumanML3D and KIT-ML with R-precision, FID, multimodal distance, and diversity;
- pose estimation on Human3.6M and 3DPW with MPJPE and PA-MPJPE;
- qualitative motion-video understanding and semantic-cycle examples;
- expert-weight analysis, body-part-tokenizer ablations, zero-expert loss ablation, and expert-count efficiency analysis.

The paper reports uncertainty terms for standard text-to-motion metrics. It also distinguishes single-task versus multi-task performance, making the cost of unification visible rather than hiding it.

### 7.4 Scientific and reproducibility limits

- The tasks are generation, pose estimation, and video understanding, not two-person retrieval.
- MT-Bench relies on an external model judge; judge version, prompt exactness, and repeated-evaluation variance matter.
- Absolute comparison across different foundation models is not causal; the authors correctly focus on relative degradation, but several baseline collapse values remain striking and require exact reproduction before reuse.
- Multi-task HMVLM trails its single-task version on several motion metrics, so “unified” must not be equated with “best.”
- Reported metric error bars are not accompanied by a preregistered paired test or source-cluster analysis.
- The checklist asserts reproducibility and statistical significance, but the paper still lacks a complete seed/attempt ledger, checkpoint identities, raw judge outputs, and a closed provenance chain in the supplied PDF.
- The semantic-cycle test visibly exposes hallucination when sampled frames miss a key pose; this is a useful failure mode, not proof of cycle invariance.

### 7.5 Exact PhasePair boundary

HMVLM already covers motion-language integration, part-specific tokenization, multi-task training, and catastrophic-forgetting mitigation. PhasePair cannot claim any of those categories broadly. Its proposed contribution is a lightweight, retrieval-specific periodic interaction representation under a fixed text/motion backbone, not a universal human-motion VLM.

Falsifiable distinction: if matched residual heads using generic relation pooling or no relation structure match PhasePair-full across both retrieval directions and all three seeds, the six-slot periodic mechanism has failed. If a larger foundation model wins only by scale, that does not validate the PhasePair mechanism.

### 7.6 Visual-language review

Strengths to adopt:

- page 2 uses qualitative task panels to establish breadth;
- page 4 keeps instruction/gating and modality projection visually separate;
- page 5 uses an overview/detail split for tokenizer and downstream tasks;
- the zero expert is represented as a named computational path, not an unexplained decoration.

Elements to reject:

- mixing too many icons, modalities, expert paths, and textual callouts in one panel;
- connectors that cross or require tracing a long route through the page;
- inconsistent shape language for data, learned modules, and losses;
- claiming invariance or preserved knowledge from a qualitative loop alone.

For PhasePair, the method figure should have one dominant horizontal inference path. Training-only losses and matched heads should sit below it on short orthogonal connectors. No connector should pass through a node or text label.

## 8. R5 — Pose2Lang3D

### 8.1 Primary metadata

- Sai Bhargav Rongali and Kenji Okuma.
- *Pose2Lang3D: Distilling 3D Reasoning from 2D Skeletons via Language Supervision*.
- CVPR Workshops 2026, pages 5399–5408.
- Official CVF record: `https://openaccess.thecvf.com/content/CVPR2026W/ABAW/html/Rongali_Pose2Lang3D_Distilling_3D_Reasoning_from_2D_Skeletons_via_Language_Supervision_CVPRW_2026_paper.html`.

### 8.2 Problem and method

Pose2Lang3D distills 3D pose reasoning into a 2D-skeleton inference model for language tasks. Training uses 2D skeleton images, 3D skeleton images, and language descriptions; inference uses 2D pose only.

The paper combines:

- a PoseFormer-style 2D encoder;
- an ST-GCN 3D encoder;
- a CLIP text encoder;
- U-Net-based noise processing and projection heads;
- uncertainty-modulated 2D alignment and geometry-enriched 3D alignment;
- global and local distillation;
- cross-attention into a MiniGPT/Vicuna language interface.

Its dataset is described as about 80,000 Human3.6M-derived pose-language samples with low-, medium-, and high-level questions generated using LLaVA.

### 8.3 Evidence

The paper trains on Human3.6M-derived data and evaluates cross-dataset behavior on MPII and MS COCO. Metrics include VQA accuracy/F1, BLEU-4, SPICE, runtime, and model size. It reports comparisons against pose-language/VLM baselines and ablations for:

- 3D pose contrast and 2D pose contrast;
- U-Net contribution;
- global versus local distillation;
- loss terms;
- layer/projection choices.

### 8.4 Scientific and reproducibility limits

- The task is static/image-based pose reasoning and captioning, not temporal retrieval.
- The main paper does not close optimizer, learning-rate, epoch, seed, repeated-run, or statistical-inference details.
- Automatically generated language supervision requires a contamination, quality, and leakage audit.
- Cross-dataset evaluation is valuable but does not replace a held-out source-group split when augmented samples share an originating pose.
- Reported percentage improvements over 2D and 3D systems depend on baseline reproduction and metric conventions.
- Rare and extreme poses remain an acknowledged limitation; temporal interaction is outside the model.

### 8.5 Exact PhasePair boundary

Pose2Lang3D establishes that 3D-informed language supervision can enrich 2D skeleton reasoning. It does not represent a pair of people over time, periodic phase relations, symmetric/directional slots, or full-gallery motion-language retrieval.

Falsifiable distinction: PhasePair must remain useful when query text differentiates interaction phase or direction between visually similar two-person sequences. A generic 2D/3D language alignment that achieves the same retrieval gains under the matched protocol would weaken the proposed novelty.

### 8.6 Visual-language review

Strengths to adopt:

- page 1 combines a problem comparison with qualitative question-answer examples;
- the proposal is recognizable through stable path colors;
- page 3 exposes both alignment and distillation rather than hiding them in prose.

Elements to reject:

- handwritten/cloud-like containers;
- a long U-shaped connection spanning the whole figure;
- connector crossings and small labels at branch intersections;
- mixing motivation and complete architecture until neither is readable.

For PhasePair, keep motivation and architecture separate. The architecture may use a two-row layout, but connectors must be short, orthogonal, and directional; the text branch should not loop around the motion graph.

## 9. Closest-work comparison matrix for the five supplied PDFs

Legend: `yes` means explicit in the paper; `partial` means adjacent but not equivalent; `no` means absent from the reviewed method.

| Axis | CountLLM | MotionMaster | LangPose | HMVLM | Pose2Lang3D | PhasePair required claim |
|---|---|---|---|---|---|---|
| Primary input | RGB video | 3D motion + text | 2D pose sequence | text/image/video/motion | 2D/3D pose images + text | two-person skeleton sequence + text |
| Main task | repetition count | generation/editing | 3D pose lifting | multi-task generation/estimation/understanding | VQA/captioning | bidirectional full-gallery retrieval |
| Explicit language | yes | yes | training-time semantic labels | yes | yes | yes, three captions/source |
| Explicit periodic reasoning | yes | no | no | no | no | yes |
| Two-person interaction | no | no | no | no | no | yes |
| Ordered `AB`/`BA` | no | no | no | no | no | yes |
| Symmetric + directional relation | no | no | no | no | no | yes |
| Multi-frequency/phase representation | partial periodic transformer | no | no | no | no | six fixed relation slots and masks |
| Wavelet matched control | no | no | no | no | no | system 03 |
| Mean/difference-DCT matched control | no | no | no | no | no | system 04 |
| Full-gallery multi-positive retrieval | no | no | no | no | no | exact, both directions |
| Source-cluster paired inference | no | no | no | no | no | 100,000 replicates |
| Corrected hypothesis family | no | no | no | no | no | Holm over H2–H6 |
| Sealed one-time test | not reported | not reported | not reported | not reported | not reported | mandatory |

The five-paper matrix supports only a **provisional** narrow boundary. It does not eliminate overlap with other dyadic/frequency/phase retrieval work.

## 10. Provisional novelty statement

The strongest defensible statement before current-literature refresh and real results is:

> PhasePair studies whether an explicit ordered two-person periodic-relation representation—combining symmetric and directional interaction signals across six fixed multi-frequency/phase slots—improves bidirectional skeleton-motion/language retrieval beyond capacity- and training-matched generic, wavelet-marginal, mean/difference-DCT, no-relation, and phase-stripped controls, under source-cluster paired inference.

This statement is intentionally conditional. It becomes a paper contribution only if:

1. the exact frozen implementation and controls are executed;
2. results support at least the relevant preregistered hypotheses without selective reporting;
3. a current primary-source search finds no earlier work with the same combination and task;
4. the paper reports negative or null findings honestly.

Claims prohibited at this stage:

- “first motion-language model with periodic reasoning”;
- “first language-aligned skeleton model”;
- “first frequency-aware motion representation”;
- “state of the art”;
- “generalizable” without predefined cross-domain or held-out-group evidence;
- “phase-aware” when the phase-stripped control is not separately executed;
- “statistically significant” before the frozen paired bootstrap and Holm procedure.

## 11. Experiment obligations derived from the five papers

### 11.1 Data and split integrity

MotionMaster shows how data construction can dominate a result; Pose2Lang3D shows the risk of many derived samples sharing one source. PhasePair must therefore:

- split and bootstrap by source cluster rather than caption or augmented row;
- complete exact and near-duplicate quarantine before training;
- bind every motion, actor, pair, caption, and composite item to immutable lineage;
- report counts before and after every exclusion;
- never let three captions from one source cross a split boundary;
- keep the sealed test inaccessible until method and analysis freeze.

### 11.2 Capacity- and optimization-matched controls

The seven frozen systems remain:

- 00 MIME base;
- 01 generic relation residual;
- 02 legacy TMR-style residual;
- 03 WaMo-style marginal-wavelet residual;
- 04 InterEdit-style mean/difference-DCT residual;
- 05 no-relation residual;
- 06 PhasePair-full.

Each has seeds 1729, 2718, and 31415. The total validation score rows are 21. Base attempts are nine; residual attempts are eighteen if and only if the frozen qualification branch permits them. A merged WaMo/InterEdit head is invalid because its failure does not establish that either standalone comparator is weaker.

### 11.3 Evaluation

Borrow the cross-domain discipline of CountLLM and the single-task/multi-task transparency of HMVLM, but use task-appropriate metrics:

- text-to-motion and motion-to-text R@1/3/5/10;
- median rank in both directions;
- exact multi-positive mask for three captions per source;
- deterministic commitment-derived tie order;
- per-source and per-cluster top-1 atoms;
- every seed reported, not only the best checkpoint;
- error cases stratified by period, phase, direction, interaction type, length, and caption ambiguity.

### 11.4 Statistics

The five papers do not provide the source-cluster paired inference required here. PhasePair keeps:

- H1 for PhasePair-full versus the MIME base comparison;
- H2 versus generic relation residual;
- H3 versus WaMo-style marginal-wavelet residual;
- H4 versus InterEdit-style mean/difference-DCT residual;
- H5 versus no-relation residual;
- H6 versus phase-stripped residual;
- exact 100,000-replicate source-cluster paired bootstrap;
- Holm correction over H2–H6;
- effect estimates, confidence intervals, raw p-values, adjusted decisions, and applicability gates;
- explicit negative/null/non-applicable outcomes.

### 11.5 Failure analyses

At minimum, inspect:

- visually similar motions with different interaction phase;
- symmetric content where AB/BA should not change the semantic match;
- directional content where AB/BA must change it;
- incomplete cycles and boundary truncation;
- long/short period imbalance;
- caption disagreement among the three positives;
- single-person-dominant sequences that may make pair modeling unnecessary;
- camera/global-motion artifacts;
- retrieval ties and near-duplicate contamination;
- cases where wavelet or DCT controls beat PhasePair-full.

## 12. Figure and PowerPoint design contract

### 12.1 Stable semantic palette

Use a restrained palette with redundant labels:

- motion/skeleton path: blue;
- text path: green;
- PhasePair periodic-relation path: violet;
- training-only objective/supervision: amber;
- frozen or reference-only component: neutral gray;
- error/HOLD/invalid relation: red used sparingly.

Colors must remain distinguishable in grayscale through shape, border, or label differences.

### 12.2 Introduction/motivation figure

Required reading order:

1. same two-person skeleton sequence and query;
2. generic motion encoding loses interaction phase/direction;
3. PhasePair exposes ordered multi-frequency relations;
4. retrieval result changes for the correct semantic reason.

Keep it understandable within seconds. No optimizer, dropout, bootstrap, or seven-system matrix belongs in this figure.

### 12.3 Method overview

Required main path:

`two-person skeleton -> motion encoder -> periodic decomposition -> six relation slots/tokens -> semantic periodic aggregation -> text alignment -> bidirectional retrieval`.

Rules:

- one dominant left-to-right flow;
- short orthogonal connectors;
- no connector crosses a node or label;
- AB and BA remain visible;
- symmetric and directional signals are not collapsed into an unexplained box;
- training-only paths sit below the inference path;
- masks appear on the signal/tokens, not only in a legend;
- the single post-aggregation `0.2` residual coefficient is shown at its actual location, not assigned to each slot;
- a zoomed detail panel may show relation construction, but it must not duplicate the entire overview.

### 12.4 Protocol and results figures

- show three fixed seeds and seven systems explicitly;
- distinguish nine base attempts from eighteen residual attempts;
- place the qualification branch before residual execution;
- show full-gallery evaluation, source-cluster atoms, 100,000 bootstrap replicates, Holm H2–H6, then one sealed-test opening;
- use uncertainty intervals and individual seed points in result plots;
- do not hide negative controls or null effects;
- generate every number from frozen result artifacts rather than retyping it.

### 12.5 Visual anti-patterns

Reject:

- emoji, fire/snowflake icons, decorative clouds, or handwritten fonts;
- stock clip art standing in for a real skeleton sequence;
- U-shaped wires around the entire architecture;
- rainbow palettes with no semantic reuse;
- tiny prose paragraphs embedded in nodes;
- gradients, shadows, or glossy effects that reduce print clarity;
- arrows with ambiguous direction;
- legends that carry information absent from the figure itself;
- qualitative cherry-picks without source IDs and selection rules;
- diagrams that imply a result before evidence exists.

## 13. Citation ledger and confidence

| ID | Verified primary metadata | Confidence | Remaining action |
|---|---|---|---|
| R1 | CVF CVPR 2025 HTML/PDF; four authors; pp. 19143–19153 | high | store final BibTeX/DOI in paper ledger |
| R2 | CVF CVPR 2026 HTML/PDF; six authors; pp. 30629–30639 | high | store final BibTeX/DOI when proceedings DOI is issued/verified |
| R3 | CVF WACV 2026 HTML/PDF; two authors; pp. 8073–8083 | high | store final BibTeX/DOI when proceedings DOI is issued/verified |
| R4 | NeurIPS 2025 official proceedings PDF; three authors | high | preserve the published `Lanuage` spelling in exact title metadata; verify canonical proceedings HTML/BibTeX |
| R5 | CVF CVPR Workshops 2026 HTML/PDF; two authors; pp. 5399–5408 | high | store workshop BibTeX and DOI if assigned |

No local PDF was copied into a public artifact. No figure was reproduced. The final paper should cite official proceedings records and keep a source hash ledger for the exact bytes actually reviewed.

## 14. Broader current-literature HOLD

The required five-paper review is complete, but the closest-work gate remains open until authoritative primary sources are reviewed for at least:

- two-person/dyadic motion-language retrieval;
- interaction-aware skeleton encoding;
- wavelet or multi-frequency motion-language models;
- mean/difference and DCT interaction encoders;
- periodic/repetition language models beyond CountLLM;
- phase-aligned or phase-manifold retrieval;
- recent source-grouped motion-retrieval benchmarks;
- any ICASSP/CVPR/NeurIPS/ICCV/ECCV/ACL-family work published after the local freeze.

Previously identified candidate names are search seeds, not accepted citations: MIME, InterEdit, WaMo, SGAR, Period-LLM, OVRCounter, PAMS, PHASOR, and MRBench. Each must be verified against a primary paper or official proceedings page before inclusion.

## 15. Handoff decisions

1. Keep the current narrow PhasePair hypothesis; do not broaden it to generic periodic-language or skeleton-language learning.
2. Do not change the seven-system matrix after seeing validation results.
3. Complete B0 near-duplicate and lineage gates before training.
4. Keep all three captions of a source inside one split and one bootstrap cluster.
5. Preserve separate WaMo and InterEdit controls.
6. Use full-gallery bidirectional evaluation and deterministic tie order.
7. Report all three seeds and all applicable H1–H6 decisions.
8. Treat null or negative evidence as a valid final outcome.
9. Build figures from the PhasePair graph, not by tracing any reference figure.
10. Refresh the broader primary-source novelty search immediately before paper freeze.

## 16. Review completion statement

Within the local five-PDF scope, this review is complete: 72/72 pages were textually reviewed, 13 method/experiment/visual pages were rendered and inspected, input bytes are pinned, official primary metadata was checked for all five papers, and the resulting scientific and visual constraints have been mapped to the PhasePair execution objective.

This completion does not upgrade PhasePair beyond `AUTHORITY0 / HOLD / NO_RESULT`. It closes only the required five-reference local review milestone.

# PhaseSet literature review: evidence, boundaries, and protocol consequences

**Review state:** `PROVISIONAL / AUTHORITY0 / NO_RESULT / PRIMARY-SOURCES-ONLY`
**Reviewed:** 2026-08-25
**Scope:** literature evidence for *PhaseSet: Permutation-Invariant Periodic
Relation Tokens for Multi-Person Motion–Language Retrieval*.

This is a literature and design-boundary record, not an empirical result, a novelty certificate, or a submission-readiness decision. Under the project ARIS rule, semantic judgements about work in the same family remain provisional until independently audited. PhaseSet has no authorized training or sealed-test result; words such as “improves,” “outperforms,” and “generalizes” below describe only what a cited paper reports about itself, never a PhaseSet outcome.

## 1. Review method and local PDF census

The five PDFs in the read-only `recommend paper` directory were independently re-extracted one page at a time and read in full. The audit covered **72/72 pages**. File identity is recorded so that later reviews can detect a replaced PDF.

| ID | Supplied PDF | Pages read | Bytes | SHA-256 |
|---|---|---:|---:|---|
| R1 | `2503.17690v2.pdf` | 1–11 (11/11) | 3,283,577 | `073E8BD9EDDA27926612026578318CDA2C28141EA4A0473ADA55DCE8705119F2` |
| R2 | `Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.pdf` | 1–11 (11/11) | 4,484,508 | `18CAE7971AEBB66C6D3ECAC163C1962B9DA842923FB426C8AF5DF3385B739E97` |
| R3 | `Liao_LangPose_Language-Aligned_Motion_for_Robust_3D_Human_Pose_Estimation_WACV_2026_paper.pdf` | 1–11 (11/11) | 3,213,424 | `9D54395F4DEA71223074B5B56AC95530FF93AEFA11FC897B30A1FC60809E19C7` |
| R4 | `NeurIPS-2025-hmvlm-human-motion-vision-lanuage-model-via-moe-lora-Paper-Conference.pdf` | 1–29 (29/29) | 5,530,658 | `74BE15EEF3A5DBC0567FF21DE8492E876D94860F6E21C3988F5DB0C101ECE758` |
| R5 | `Rongali_Pose2Lang3D_Distilling_3D_Reasoning_from_2D_Skeletons_via_Language_Supervision_CVPRW_2026_paper.pdf` | 1–10 (10/10) | 675,267 | `286923F6DD06479768CD1DC1483FAF62F99335E38E107EA98ED4B38C7A1612EA` |

The existing PhasePair deep-review log was used as a cross-check, not as a substitute for this pass. External facts below use a paper publisher page, arXiv record, DOI landing page, project page, dataset record, or author-owned repository. Secondary survey and paper-note pages were excluded from evidence.

## 2. The registered PhaseSet target (not a result)

PhaseSet is a **planned** bidirectional group-motion/text retrieval study. The frozen contract specifies a shared actor encoder, six fixed Morlet bands, all unordered actor pairs streamed in canonical blocks, symmetric pair features, incident half-edge moments, six permutation-invariant group tokens, and a frozen CLIP text tower with a shared band projection. It is designed for dynamic `K>=2`, with topology forced to exact `+0` (and zero gradient) for `K=2`. Runtime cost is `O(K^2)` in streamed edges; an over-limit input fails closed rather than sampling pairs.

The registered primary source is a project-qualified subset of Embody 3D: **572 eligible native `K>=3` captures**, split **400/96/76** at the participant-component level. The census contains 27 `K=3` and 545 `K=4` captures; all 27 `K=3` captures are in test. Thus `K=3` is a predeclared **unseen-cardinality test slice** and the paper must never imply that training observed `K=3`. These are registered census/split facts, not model results and not Embody 3D’s headline dataset statistics.

The execution ledger contains nine base-qualification runs and 24 residual-system runs (eight systems × three seeds), for **33 training attempts**. H1 (full vs qualified base) is the sole primary hypothesis; H2–H8 compare full against seven structural controls and form one Holm family. After validation freeze, the protocol permits one sealed-test execution and 100,000 paired capture-cluster bootstrap draws. No cell may be populated before that chain closes.

## 3. Complete review of the five supplied papers

### R1. CountLLM

**Citation.** Ziyu Yao, Xuxin Cheng, Zhiqi Huang, and Lei Li, “CountLLM: Towards Generalizable Repetitive Action Counting via Large Language Model,” CVPR 2025, pp. 19143–19153. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html)

- **Task/data.** RGB-video repetitive-action counting on RepCount, UCFRep, and Countix, including cross-dataset transfer and out-of-vocabulary action evaluation. The output is a scalar count with boundary-cycle indicators, not a motion/text retrieval ranking.
- **Method.** A pretrained video encoder feeds a 12-layer periodicity transformer with 64 learned queries, then a projector and Vicuna-7B. LoRA rank 16 is applied to attention. The structured answer format explicitly represents a count and incomplete cycles at the beginning/end.
- **Experiment specification.** The paper describes progressive WebVid-10M semantic alignment, repetitive-action periodic alignment, then instruction tuning; it reports OBO and normalized MAE, cross-dataset tests, ablations of the periodicity prompt/transformer/stages/LoRA, and eight-A100 training. It does not provide PhaseSet-style three-seed paired inference.
- **Useful for PhaseSet.** Make periodic semantics and boundary conditions explicit; separate semantic alignment from periodic alignment; retain a cross-domain stress test.
- **Not supportable from this source.** Any claim of being the first periodic-language model, first language-conditioned periodic system, or first frequency-aware motion model.
- **PhaseSet difference.** PhaseSet consumes multi-person 3D skeleton groups and ranks whole captures against descriptions. It studies actor–relation incidence and physical phase-derived features; CountLLM consumes RGB and predicts counts.

### R2. MotionMaster

**Citation.** Nan Jiang, Yunhao Li, Lexi Pang, Zimo He, Siyuan Huang, and Yixin Zhu, “MotionMaster: Generalizable Text-Driven Motion Generation and Editing,” CVPR 2026, pp. 30629–30639. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.html)

- **Task/data.** Single-person text-to-motion generation and instruction-based editing. MotionGB is constructed by scaling roughly 400 hours of verified motion to a reported 10,000 hours through temporal/body-part composition and 24 edit families, with machine-generated multilevel descriptions.
- **Method.** A four-layer convolutional FSQ motion tokenizer, localized 85D motion features plus global losses, coarse-to-fine VPoser/IK recovery, Qwen2.5-VL shared tokens, separate RoPE counters, and semantic-density sampling.
- **Experiment specification.** A 90/10 MotionGB split and a 400-example test-lite set; local/global position, rotation, and velocity errors; R-precision/diversity; automated semantic/physical scoring; and ablations over data scale, model scale, balancing, and task mixture. The supplied main paper leaves unresolved supplementary `??` references and does not expose the complete training ledger needed for direct reproduction.
- **Useful for PhaseSet.** Audit data-lineage and capacity confounds; distinguish native data from constructed compositions; report local and global failure modes separately.
- **Not supportable from this source.** PhaseSet generation/editing ability, scale-based generalization, or human-quality motion synthesis.
- **PhaseSet difference.** PhaseSet is no-generation retrieval on native multi-person captures. It does not concatenate motions to enlarge the primary dataset and does not use text to supervise physical phase fields.

### R3. LangPose

**Citation.** Longyun Liao and Rong Zheng, “LangPose: Language-Aligned Motion for Robust 3D Human Pose Estimation,” WACV 2026, pp. 8073–8083. [CVF paper page](https://openaccess.thecvf.com/content/WACV2026/html/Liao_LangPose_Language-Aligned_Motion_for_Robust_3D_Human_Pose_Estimation_WACV_2026_paper.html)

- **Task/data.** 2D-to-3D single-person pose lifting. Language is used in pretraining on BABEL/AMASS-derived skeletons, then removed for target-dataset fine-tuning/evaluation on Human3.6M and MPI-INF-3DHP.
- **Method.** Five MotionBERT layers for pose and three BERT layers for text, with body-part/time-window/joint/frame corruption, language alignment, corrupted-pose reconstruction, and position/velocity losses.
- **Experiment specification.** One H100, batch 16, 243-frame clips, 300 epochs; MPJPE, P-MPJPE, H-MPJPE, PCK, and AUC; component and corruption ablations. Multi-seed uncertainty or paired significance is not specified.
- **Useful for PhaseSet.** Preserve masks through corruption; distinguish language used as a training prior from language used as a retrieval query; predeclare robustness probes.
- **Not supportable from this source.** Novelty for language-aligned skeleton learning, group interaction modeling, or retrieval.
- **PhaseSet difference.** PhaseSet aligns group motion and holistic text at inference and never predicts 3D pose from 2D observations.

### R4. HMVLM

**Citation.** Lei Hu, Yongjing Ye, and Shihong Xia, “HMVLM: Human Motion-Vision-Lanuage Model via MoE LoRA,” NeurIPS 2025. The misspelling “Lanuage” is reproduced from the official title rather than silently corrected. [Official proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/8cb564df771e9eacbfe9d72bd46a24a9-Paper-Conference.pdf)

- **Task/data.** A unified single-person model for text-to-motion, 3D pose estimation, motion/video understanding, and general language retention. Data include HumanML3D, KIT-ML, Human3.6M, 3DPW, MoVid, and LMSYS-Chat-1M.
- **Method.** Vicuna-7B-v1.5 with five LoRA experts (including a fixed zero expert), a CLIP-conditioned router, five body-part token streams, extended vocabulary, and next-token training.
- **Experiment specification.** A VQ tokenizer and joint training are described for one A800 80GB GPU; evaluation covers MT-Bench, text-to-motion retrieval/FID/diversity, pose MPJPE, video questions, and ablations. Several tables expose error bars without a complete seed/sampling definition, and multi-task variants trail single-task variants on some metrics.
- **Useful for PhaseSet.** Treat task interference and knowledge retention as empirical questions; expose tokenizer and routing choices; avoid interpreting a unified interface as uniform task gains.
- **Not supportable from this source.** First unified motion-language model, novelty of body-part tokens, or a claim that PhaseSet handles generation, pose estimation, or video QA.
- **PhaseSet difference.** PhaseSet is a compact discriminative group-retrieval residual with a frozen text tower, not an autoregressive generalist model.

### R5. Pose2Lang3D

**Citation.** Sai Bhargav Rongali and Kenji Okuma, “Pose2Lang3D: Distilling 3D Reasoning from 2D Skeletons via Language Supervision,” CVPR Workshops 2026, pp. 5399–5408. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026W/ABAW/html/Rongali_Pose2Lang3D_Distilling_3D_Reasoning_from_2D_Skeletons_via_Language_Supervision_CVPRW_2026_paper.html)

- **Task/data.** Static/image pose VQA and captioning. Training uses 2D pose, 3D pose, and language; inference uses 2D pose. The paper builds an approximately 80K Human3.6M-derived pose/question resource and probes MPII/COCO transfer.
- **Method.** MMPose/PoseFormer inputs, ST-GCN, CLIP text, a U-Net noise branch, global/local alignment transformers, and distillation into a MiniGPT/Vicuna-style language model.
- **Experiment specification.** VQA accuracy/F1, BLEU-4/SPICE, runtime/model size, cross-dataset tests, and ablations over 2D/3D alignment, U-Net, global/local paths, losses, layers, and projectors. Full optimizer, epoch, seed, and statistical details are not present in the supplied paper.
- **Useful for PhaseSet.** Separate privileged training information from inference inputs; audit machine-generated language supervision; include efficiency and cross-domain behavior.
- **Not supportable from this source.** Novelty for 3D-informed language supervision or evidence for temporal, periodic, multi-person retrieval.
- **PhaseSet difference.** PhaseSet uses temporal 3D skeleton groups and human holistic descriptions for primary retrieval; it neither reconstructs hidden 3D pose nor answers free-form VQA.

## 4. Direct multi-person and periodic-motion neighbors

### Embody 3D

**Primary sources:** [paper](https://arxiv.org/abs/2510.16258), [official repository and access instructions](https://github.com/facebookresearch/embody-3d)

- **Task/data.** *Embody 3D: A Large-scale Multimodal Motion and Behavior Dataset* reports roughly 500 individual hours, 439 participants, and more than 54 million frames across single- and multi-person activities, with body/hands/shape, participant audio, and text metadata.
- **Method/experiment relevance.** This is a dataset paper and release toolkit, not a retrieval baseline. The repository records 30 fps data, missing-data masks, and annotation availability that varies by collection.
- **Useful.** Native simultaneous groups, participant-aware splitting, synchronized modalities, and mask-preserving preprocessing.
- **Do not claim.** The project’s 572-capture census is not an official Embody 3D headline number. PhaseSet cannot redistribute controlled assets or imply broader dataset coverage than its eligibility manifest.
- **PhaseSet difference.** PhaseSet defines a retrieval benchmark and a frozen 400/96/76 derivative split over an eligible subset; Embody 3D supplies the underlying records.

### SocialGen and SocialX

**Primary sources:** [paper](https://arxiv.org/abs/2503.22906), [project page](https://socialgenx.github.io/)

- **Task/data.** Heng Yu, Juze Zhang, Changan Chen, Tiange Xiang, Yusu Fang, Juan Carlos Niebles, and Ehsan Adeli, *SocialGen: Modeling Multi-Human Social Interaction with Language Models*. SocialX consolidates multiple motion sources and synthetic combinations into instruction data for 25 generation/understanding tasks.
- **Method.** An XH3D representation stores per-person motion plus root position/yaw relative to a randomly selected reference actor; motion and spatial tokenizers feed a T5-base autoregressive model.
- **Experiment specification.** The paper describes 20 fps processing, codebooks of size 512, GPT-4o-mini caption construction, and multi-GPU tokenizer/pretraining/instruction-tuning stages.
- **Useful.** Variable-person batching, broad task taxonomy, explicit spatial tokens, and actor-count stress testing.
- **Do not claim.** First arbitrary-group motion-language system, first social-motion instruction dataset, or evidence that synthetic combinations validate native group topology.
- **PhaseSet difference.** PhaseSet requires permutation-invariant outputs with no indexed reference actor, uses native simultaneous primary captures, and evaluates full-gallery bidirectional retrieval rather than autoregressive task execution.

### Multi-Motion

**Primary sources:** [ECCV paper page](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8171_ECCV_2024_paper.php), [project](https://shanmy.github.io/Multi-Motion/), [official code](https://github.com/Shanmy/Multi-Motion)

- **Task/data.** Mengyi Shan, Lu Dong, Yutao Han, Yuan Yao, Tao Liu, Ifeoma Nwogu, Guo-Jun Qi, and Mitchell K. Hill, *Towards Open Domain Text-Driven Synthesis of Multi-Person Motions*, ECCV 2024. LAION-Pose and WebVid-Motion are built from estimated poses and web text.
- **Method.** A diffusion architecture alternates pose layers across subjects and motion layers across time, enabling variable subject/frame counts.
- **Experiment specification.** Evaluation concerns generated multi-person motion and open-domain text conditioning, not full-gallery retrieval.
- **Useful.** Separate within-frame social structure from within-person temporal structure; test variable cardinality.
- **Do not claim.** First open-domain multi-person text-to-motion work, and do not treat estimated web pose/captions as clean evidence for native mocap phase.
- **PhaseSet difference.** PhaseSet is retrieval on synchronized 3D captures and uses physical periodic relations; it does not synthesize motion.

### LLaMMo and LLaVerse

**Primary sources:** [AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/37556), [official PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37556/41518)

- **Task/data.** Lei Li, Sen Jia, and Jenq-Neng Hwang, “Multiple Human Motion Understanding,” AAAI 2026, 40(8):6297–6305, DOI `10.1609/aaai.v40i8.37556`. LLaVerse contains reported manual annotations for more than 200K sequences and 600K motion–text pairs from social and professional-sport settings; LLaMI-Bench evaluates understanding.
- **Method.** LLaMMo couples a human encoder and social-temporal M3Former to an instruction-tuned language model and accepts multiple keypoint/video representations.
- **Experiment specification.** Captioning and question answering across social and sport scenarios; it is not a full-gallery capture retrieval protocol.
- **Useful.** Explicit multi-human semantic evaluation, social/temporal factorization, and domain-diverse diagnostics.
- **Do not claim.** First multi-human motion understanding, first multi-human motion instruction tuning, or first large multi-human motion-language corpus.
- **PhaseSet difference.** PhaseSet asks a narrower retrieval question about periodic relation incidence and uses predeclared structural controls instead of open-ended language generation.

### Multi-TPC

**Primary sources:** [Scientific Data paper](https://doi.org/10.1038/s41597-026-06819-x), [official repository](https://github.com/MCMartinLee/Multi-TPC), [data DOI](https://doi.org/10.5281/zenodo.17935560)

- **Task/data.** Meng-Chen Lee and Zhigang Deng, “Multi-TPC: A Multimodal Dataset for Three-Party Conversations with Speech, Motion, and Gaze,” *Scientific Data* 13:429 (2026). It contains 5.3 hours, 24 sessions, 21 participants in seven fixed triads, and synchronized VICON motion, audio, and eye gaze.
- **Method/experiment relevance.** Dataset acquisition/validation at a downsampled 60 fps; no PhaseSet-style holistic group-text retrieval benchmark.
- **Useful.** A natural-conversation `K=3` cross-domain probe with synchronized nonverbal modalities.
- **Do not claim.** It is not a broad arbitrary-cardinality benchmark; fixed triads, physical layout, duration, and demographic scope limit inference.
- **PhaseSet difference.** PhaseSet’s primary K=3 slice is zero-shot within Embody 3D; Multi-TPC is a separately licensed cross-domain probe, never a replacement primary test.

### M3Act

**Primary sources:** [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chang_Learning_from_Synthetic_Human_Group_Activities_CVPR_2024_paper.pdf), [arXiv](https://arxiv.org/abs/2306.16772), [project page](https://cjerry1243.github.io/M3Act/)

- **Task/data.** Che-Jui Chang et al., “Learning from Synthetic Human Group Activities,” CVPR 2024, pp. 21922–21932. M3ActRGB targets multi-person tracking and group activity recognition; M3Act3D reports 87.6 hours of synthetic 3D group activity, average group size 6.7 and maximum 27.
- **Method.** Unity-based controllable generation supports group layouts/activities and downstream discriminative or generative studies.
- **Experiment specification.** Multi-person tracking, group activity recognition, and controllable group-motion generation; no natural-language full-gallery retrieval.
- **Useful.** Large-K engineering, memory/resource qualification, permutation tests, and synthetic diagnostics.
- **Do not claim.** Synthetic results demonstrate native human social behavior, realistic physical phase, or primary-domain generalization.
- **PhaseSet difference.** M3Act3D is an engineering/probe surface only; PhaseSet’s confirmatory data remain native captures.

### MIME

**Primary source:** [arXiv paper](https://arxiv.org/abs/2607.22702)

- **Task/data.** Addison Zucek, Prerit Gupta, Kamila Kuatova, and Aniket Bera, *MIME: Multimodal Interactive Motion Encoder* (arXiv:2607.22702, 2026). It studies two-person interactive text–motion retrieval on Inter-X and downstream conditioning on InterHuman.
- **Method.** Separate actor streams, bidirectional co-attention, explicit frame-level relational features, and curriculum contrastive learning.
- **Experiment specification.** Retrieval galleries of 500, 1,000, and full 2,279 samples, plus use as a frozen prior in interactive generation. As reviewed, no public code/checkpoint was available; the paper says release is planned upon acceptance.
- **Useful.** Dyadic interaction-aware retrieval is the closest task neighbor; retain full-gallery evaluation and avoid flattening actor streams too early.
- **Do not claim.** First interaction-aware retrieval, official MIME reproduction, or direct reproducibility while public code/checkpoints remain absent.
- **PhaseSet difference.** MIME is fixed `K=2`; PhaseSet’s central object is native `K>=3` incidence topology. PhaseSet’s bases are in-house ActorMean/SetPMA/SocialTemporal systems, not a MIME reproduction.

### WaMo

**Primary sources:** [current arXiv v2](https://arxiv.org/abs/2508.03343), [official repository](https://github.com/3DAgentWorld/WaMo/), [ACM DOI](https://doi.org/10.1145/3767308.3835809)

- **Task/data.** Junlong Ren, Gangjian Zhang, Honghao Fu, Pengcheng Wu, and Hao Wang, “WaMo: Wavelet-Enhanced Multi-Frequency Trajectory Analysis for Fine-Grained Text-Motion Retrieval,” ACM MM 2026. HumanML3D and KIT-ML are single-person motion/text retrieval datasets.
- **Method.** Learnable stationary wavelet decomposition, intra/inter-frequency attention, learnable inverse reconstruction, and shuffled-sequence order prediction; DistilBERT supplies text features.
- **Experiment specification.** Bidirectional R@1/2/3/5/10, MedR, Rsum; Adam at `1e-4`, 256D latent, three decomposition levels, temperature 0.07; ablations compare no transform, FFT, DCT, convolution, DWT, and SWT. The official repository states that code/models/documentation are still under construction.
- **Useful.** A mandatory marginal-frequency control and careful transform-family ablation; distinguish local/multiscale frequency content from actor-to-actor phase topology.
- **Do not claim.** First wavelet motion retrieval, or that PhaseSet reproduces WaMo without released code. WaMo’s paper-reported gains are not PhaseSet evidence.
- **PhaseSet difference.** WaMo models marginal single-person joint trajectories with learned SWT. PhaseSet fixes six Morlet bands, binds inter-actor directed relations, and tests group incidence under native `K>=3`.

### InterEdit

**Primary sources:** [current arXiv v2](https://arxiv.org/abs/2603.13082), [official repository](https://github.com/YNG916/InterEdit)

- **Task/data.** Yebin Yang et al., “InterEdit: Navigating Text-Guided 3D Dyadic Human Motion Editing,” accepted to ECCV 2026 according to arXiv v2. InterEdit3D contains 5,161 `(source, target, edit instruction)` pairs derived from InterHuman and split 80/10/10 with interaction-level separation.
- **Method.** Synchronized classifier-free conditional diffusion, semantic-aware plan tokens, and interaction-aware frequency tokens trained against three DCT energy bands. The official repository includes code/configuration and points to data/checkpoints.
- **Experiment specification.** Five transformer blocks, 16 heads, 512D motion embeddings, frozen CLIP ViT-L/14, 16 plan tokens, 1,000 diffusion steps, 50-step DDIM inference; generation FID and generated-to-source/target retrieval with reported 95% intervals, ablations, and human preference.
- **Useful.** A DCT/energy control, source-preservation diagnostics, interaction-disjoint splitting, and a warning that frequency alignment may act as an auxiliary regularizer.
- **Do not claim.** First multi-person editing, editing performance, or generation fidelity for PhaseSet. The repository README still uses an older “multi-human” title; the current v2 title above is authoritative.
- **PhaseSet difference.** InterEdit is dyadic editing/generation and receives target-derived auxiliary supervision during training. PhaseSet is retrieval, supports dynamic K, and forbids language/targets from generating or supervising physical phase, lag, power, or coherence fields.

## 5. Cross-paper comparison and design consequences

| Work | People | Primary task | Frequency/relations | Language role | Direct PhaseSet evidence? |
|---|---:|---|---|---|---|
| CountLLM | video scene | action counting | learned periodic transformer | instruction/output format | No—task/output differ |
| WaMo | 1 | text–motion retrieval | learned marginal SWT | retrieval query | Control only |
| MIME | 2 | interactive retrieval | learned frame relations | retrieval query | Closest dyadic task, not group topology |
| InterEdit | 2 | motion editing | DCT interaction-energy tokens | edit instruction | Control only |
| SocialGen | variable | generation/understanding | social/spatial tokens | autoregressive instruction | Neighbor, not invariant retrieval |
| Multi-Motion | variable | text-to-motion generation | pose/motion factorization | generation condition | Neighbor, not retrieval |
| LLaMMo | multiple | captioning/QA | social-temporal encoder | generated answer | Neighbor, not full-gallery retrieval |
| Multi-TPC | exactly 3 | dataset | synchronized behavior | conversation/audio, not holistic retrieval text | Cross-domain probe only |
| M3Act3D | variable | synthetic group tasks | synthetic group dynamics | no primary retrieval text | Engineering/probe only |
| PhaseSet | native 3–4 primary | planned bidirectional retrieval | fixed bands + pair and incidence topology | frozen semantic band tokens | **NO_RESULT** |

The review imposes the following manuscript constraints:

1. Do not use “first” for multi-person motion–language, group generation/understanding, periodic-language modeling, wavelet retrieval, or dyadic interactive retrieval.
2. State the narrow hypothesis: frequency-specific actor–relation incidence structure may add retrieval utility over a qualified group encoder and matched controls.
3. Treat WaMo-style marginal frequency, InterEdit-style DCT energy, PhasePair/pair-bag, generic tokens, coverage-only, incidence shuffle, and phase stripping as distinct controls.
4. Label Embody 3D’s official scale separately from the registered 572-capture eligibility census.
5. Keep Multi-TPC, M3Act3D, AIOZ-GDANCE, and dyadic corpora outside the primary confirmatory claim.
6. Never describe an in-house adaptation as an official reproduction when public code/checkpoints or exact preprocessing are unavailable.
7. Keep all PhaseSet result cells `HOLD` until the one-shot sealed test and frozen 100,000-draw analysis are authorized and complete.

## 6. Claim boundary

Even if every registered gate later passes, the strongest permitted conclusion is conditional: on the frozen Embody-derived gallery, participant-component split, preprocessing, base-selection rule, eight-control family, and three registered seeds, the full PhaseSet system changed the registered bidirectional retrieval endpoint relative to named controls. It would not establish causal coordination, universal group understanding, intent/identity/ability inference, arbitrary-domain generalization, or generation/editing capability. At this review state there is no empirical conclusion at all.

# PhasePair current closest-work and novelty-boundary refresh

**Refresh ID:** `phasepair-current-closest-work-refresh-v1/20260825_015706`  
**Cutoff:** 2026-08-25, Asia/Shanghai  
**Bound local review:** `PHASEPAIR_REFERENCE_PAPERS_DEEP_REVIEW_20260825_014557.md`, SHA-256 `326f4d8a622b09fd4a751b350a89d0f7d188a8df4d9d9385916d7d655a3ee0c0`  
**Bound execution objective:** `PHASEPAIR_END_TO_END_EXECUTION_OBJECTIVE_20260825_002858.md`, SHA-256 `4e8977ec114de8657f607dc3be0fb2027485cc6d22b3a323e88114bb12c29f29`  
**Status:** current primary-source refresh complete; novelty remains provisional until experiments and a final pre-submission refresh  
**Authority:** literature evidence only; `AUTHORITY0 / HOLD / NO_RESULT`

## 1. Scope and source policy

This refresh addresses the nearest work not exhausted by the five required local PDFs. It uses primary paper pages or official proceedings records. Search aggregators may locate a candidate but do not support a claim. The accepted evidence surfaces in this report are:

- arXiv abstract records for current preprints;
- CVF Open Access pages for CVPR/WACV/CVPR Findings papers;
- NeurIPS official proceedings pages;
- ICLR official proceedings/OpenReview where applicable.

The refresh does not treat a title, project page, code repository, or model-generated summary as proof of a method detail. Details below are limited to what the cited primary record states or to a separately bound local paper review.

## 2. Primary-source candidate ledger

| ID | Work | Primary source | Why it pressures PhasePair |
|---|---|---|---|
| N1 | MIME: Multimodal Interactive Motion Encoder, Zucek et al., 2026 | `https://arxiv.org/abs/2607.22702` | dedicated two-person interactive motion-language retrieval encoder; direct closest base |
| N2 | WaMo: Wavelet-Enhanced Multi-Frequency Trajectory Analysis for Fine-Grained Text-Motion Retrieval, Ren et al., 2025 | `https://arxiv.org/abs/2508.03343` | wavelet and multi-frequency text-motion retrieval are already explicit |
| N3 | InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing, Yang et al., 2026 | `https://arxiv.org/abs/2603.13082` | two-person mean/difference DCT frequency tokens and six band controls already exist |
| N4 | SGAR: Structural Generative Augmentation for 3D Human Motion Retrieval, Zhang et al., NeurIPS 2025 | `https://proceedings.neurips.cc/paper_files/paper/2025/hash/99e6344e5db70177724367b3e55a88c2-Abstract-Conference.html` | part-level retrieval and directional relation alignment are already explicit |
| N5 | CountLLM, Yao et al., CVPR 2025 | `https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html` | periodic language prompts/tokens and progressive periodic training are not new |
| N6 | Period-LLM, Zhang et al., CVPR 2025 | `https://arxiv.org/abs/2505.24476` | cross-modal periodic reasoning and easy-to-hard periodic training are not new |
| N7 | OVR/OVRCounter, Dwibedi et al., 2024 | `https://arxiv.org/abs/2407.17085` | free-form text-conditioned open-vocabulary repetition counting and localization exist at scale |
| N8 | Count What Repeats / PAMS, Gao et al., CVPR Findings 2026 | `https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html` | self-supervised skeleton periodic/phase alignment across temporal scales is explicit |
| N9 | PHASOR, Kim et al., 2026 | `https://arxiv.org/abs/2606.01851` | FFT-parametric phase manifolds and retrieval-aligned action embeddings exist outside language/dyadic retrieval |
| N10 | MRBench, Liu et al., 2026 | `https://arxiv.org/abs/2608.07993` | three description granularities, heterogeneous sources, and cross-granularity retrieval gaps are current benchmark evidence |
| N11 | TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis, Petrovich et al., 2023 | `https://arxiv.org/abs/2305.00976` | foundational retrieval task and shared embedding baseline |

The five local references remain part of the full related-work ledger. This table adds the candidates that most directly constrain the PhasePair claim.

## 3. Candidate-specific implications

### 3.1 MIME is the direct closest base, not a weak baseline

MIME explicitly targets two-person interactive motion and language. Its primary record describes stream-based co-attention, explicit interaction features, curriculum contrastive training, Inter-X retrieval, and transfer to InterHuman generation systems. Therefore:

- “first two-person motion-language encoder” is unavailable;
- “interaction-aware retrieval” is unavailable as a broad claim;
- early- and late-fusion comparators alone are insufficient to establish PhasePair novelty;
- MIME must remain the qualified base in H1, trained under the exact three-seed, full-gallery, resolved-text, optimizer, and dropout protocol;
- the paper must explain why PhasePair's relation representation is not merely a reimplementation of MIME's explicit interaction branch.

The local frozen MIME evidence and adaptation spec remain authoritative for implementation. The arXiv abstract is used here only to classify the novelty boundary.

### 3.2 WaMo removes a generic multi-frequency retrieval claim

WaMo explicitly decomposes joint trajectories with wavelets at multiple frequencies, reconstructs them with learnable inverse transforms, and uses a sequence-ordering auxiliary task for fine-grained text-motion retrieval. Therefore:

- “first multi-frequency text-motion retrieval” is unavailable;
- “frequency decomposition improves retrieval” is not a standalone novelty claim;
- PhasePair requires a separate capacity- and training-matched WaMo-style marginal-wavelet residual, system 03;
- a gain over a generic relation head does not isolate phase or dyadic structure if WaMo is omitted.

The comparator must use marginal actor trajectories only where specified. It must not be weakened by sharing InterEdit's DCT mask or by merging both methods into one head.

### 3.3 InterEdit removes mean/difference-DCT and six-token claims

InterEdit introduces multi-person motion editing and reports an interaction-aware frequency alignment using average/sum-like and difference signals, DCT energy pooling, and six low/mid/high frequency controls. Therefore:

- two-person mean/difference decomposition is not new;
- DCT band energy is not new;
- six frequency tokens are not new merely because the count is six;
- PhasePair's novelty cannot rest on the slot count or S/D construction;
- system 04 must remain a pure-DCT InterEdit-style comparator with paired S/D band masks and zero Morlet reads.

This is an especially close conceptual control even though InterEdit's task is editing rather than retrieval. If PhasePair-full cannot beat the matched InterEdit head, the claimed contribution of its richer phase relation is not supported.

### 3.4 SGAR removes generic part and directional-relation claims

SGAR uses body-part concept decomposition, generation-augmented descriptions, part-mixture learning, and directional relation alignment between full-body and part motions. Therefore:

- “fine-grained body-part text-motion alignment” is unavailable;
- “directional relation alignment for retrieval” is unavailable in broad form;
- H2 requires a generic relation residual rather than only no-relation and frequency controls;
- the PhasePair paper must explain that its ordered relation is inter-person and periodic, not only whole-to-part directionality.

### 3.5 CountLLM, Period-LLM, OVR, and PAMS remove broad periodic-language claims

Together these works cover:

- structured periodic prompts and learned periodic tokens for counting;
- cross-modal MLLM periodic reasoning and progressive easy-to-hard training;
- large-scale free-form text-conditioned repetition localization/counting;
- self-supervised skeleton embeddings that align cycle phases across multiple temporal scales.

Consequently, PhasePair cannot claim periodic reasoning, periodic language conditioning, skeleton phase learning, tempo robustness, or multi-scale cycle consistency as new categories. Their tasks are primarily counting or question answering, which leaves a narrower retrieval/interaction boundary, but task difference must be stated rather than used to ignore them.

Required stress tests inherited from this cluster:

- incomplete cycles and clipped boundaries;
- tempo variation and within-sequence tempo drift;
- short-versus-long period imbalance;
- visually similar cycles with different semantic interaction roles;
- text queries that name only the repeating sub-action;
- out-of-domain action categories where feasible without altering the frozen split.

### 3.6 PHASOR removes a broad phase-manifold retrieval claim

PHASOR factorizes motion into an FFT-parametric phase manifold plus non-periodic pose context, aligns it with motion semantics, and performs cross-embodiment retrieval across humans and humanoid robots. Therefore:

- “phase-anchored retrieval representation” is unavailable as a broad claim;
- “FFT/phase manifold yields interpretable action embeddings” is unavailable as a broad claim;
- PhasePair must emphasize ordered two-person language retrieval and its symmetric/directional interaction evidence;
- H6 phase stripping is essential. Without it, any gain could be attributed to generic extra features rather than phase information.

PHASOR is not a direct task match: it addresses cross-embodiment robot/action retrieval, not two-person text-motion retrieval. It is nevertheless a direct conceptual neighbor on phase-structured representation learning.

### 3.7 MRBench removes a generic three-caption or granularity claim

MRBench reports 3,390 heterogeneous motions, 118 fine-grained categories, and 10,170 captions: concise, standard, and fine-grained descriptions. It finds large cross-dataset and query-granularity gaps and proposes granularity-aware score fusion.

Therefore:

- three descriptions per motion are not novel;
- mixed-granularity alignment is not novel;
- PhasePair's three-caption objective is a protocol choice, not a contribution by itself;
- source-group integrity is crucial because heterogeneous, derived, or generated motions can share latent provenance;
- the paper should report caption-granularity sensitivity as descriptive evidence if the InterHuman captions support a defensible frozen classification, but must not invent a post-hoc granularity taxonomy after seeing results.

MRBench appeared on arXiv in August 2026 and must be cited in the final current-work section if its primary record remains active at paper freeze.

## 4. Current closest-work matrix

Legend: `Y` explicit; `P` partial/adjacent; `N` absent from the primary description; `D` different task/domain.

| Work | Text-motion retrieval | Two-person | Explicit relation | Frequency/wavelet | Explicit phase | Ordered direction | Multi-caption/granularity | Main task |
|---|---|---|---|---|---|---|---|---|
| MIME | Y | Y | Y | N | N | P | P | interactive retrieval |
| WaMo | Y | N | P by body part | Y wavelet | P | N | N | retrieval |
| InterEdit | D | Y | Y | Y DCT | P periodic energy | P | N | motion editing |
| SGAR | Y | N | Y whole/part | N | N | Y | generated part text | retrieval |
| CountLLM | D | N | N | P periodic tokens | P | N | structured prompt | counting |
| Period-LLM | D | N | N | P periodic tasks | P | N | multimodal QA | periodic reasoning |
| OVR/OVRCounter | D | N | N | P repetition | P | N | free-form target text | localization/counting |
| PAMS | D | N | N | Y multi-scale | Y cycle phase | N | N | skeleton counting |
| PHASOR | D | N | N | Y FFT | Y | N | semantic prior | cross-embodiment retrieval |
| MRBench | Y | N | N | N | N | N | Y, three levels | retrieval benchmark |
| PhasePair target | Y | Y | Y inter-person | Y matched relation slots | Y | Y AB/BA | Y, three positives | bidirectional retrieval |

No single row above, based on the current primary records, states the full PhasePair combination. That absence is necessary but not sufficient for novelty.

## 5. Remaining narrow contribution candidate

The remaining candidate is not “periodic motion-language retrieval” in general. The narrow candidate is:

> An ordered two-person periodic-relation representation for bidirectional skeleton-motion/language retrieval that keeps AB/BA identity, symmetric and directional interaction signals, and six fixed multi-frequency/phase relation slots visible, and that is tested against a qualified two-person MIME base plus separate generic-relation, WaMo-style marginal-wavelet, InterEdit-style mean/difference-DCT, no-relation, and phase-stripped controls.

The following are evidence design, not algorithmic novelty:

- three fixed seeds;
- three captions per source;
- full-gallery evaluation;
- source-cluster paired bootstrap;
- Holm correction;
- immutable lineage and sealed-test access.

They strengthen trust in the result but should not be sold as a new learning method.

## 6. Falsification map

| Hypothesis | Nearest-work pressure | What a failure means |
|---|---|---|
| H1 PhasePair-full vs MIME | N1 | the proposed relation head adds no demonstrated value beyond a strong interaction encoder |
| H2 vs generic relation | N4 | gains may come from any extra relational capacity, not periodic structure |
| H3 vs WaMo marginal-wavelet | N2 | gains may come from generic multi-frequency motion features |
| H4 vs InterEdit S/D-DCT | N3 | gains may come from mean/difference band energy rather than the PhasePair relation design |
| H5 vs no relation | all retrieval baselines | the relation branch may be unnecessary |
| H6 vs phase stripped | N8/N9 | the claimed benefit may not depend on phase information |

The project remains scientifically valid if one or more hypotheses fail. The paper must then narrow or reject the corresponding mechanism claim.

## 7. Mandatory paper wording

### 7.1 Safe current wording

Use:

- “We test whether ...” before results exist;
- “to our knowledge, the reviewed primary sources do not combine ...” only after the final refresh;
- “two-person skeleton-motion/language retrieval” rather than generic motion understanding;
- “ordered periodic interaction relations” rather than merely frequency-aware features;
- “matched comparator” only when architecture, optimizer, data, captions, seeds, steps, and evaluator are actually matched.

### 7.2 Prohibited wording

Do not use:

- first periodic motion-language model;
- first two-person motion-language encoder;
- first wavelet/multi-frequency retrieval model;
- first mean/difference interaction representation;
- first phase-based retrieval representation;
- first multi-granular motion-text benchmark;
- generalizable, robust, or state of the art without the registered evidence;
- significant without the exact paired inference and multiplicity correction.

## 8. Experiment and implementation consequences

1. Keep MIME/early/late base qualification at exactly three systems by three seeds.
2. Keep systems 01–06 separate and run exactly 18 residual attempts only after MIME qualification.
3. Never merge WaMo and InterEdit into one comparator.
4. Make system 03 unable to read the InterEdit DCT mask.
5. Make system 04 unable to read Morlet/wavelet inputs.
6. Keep system 06's AB/BA swap and phase semantics testable at score level.
7. Bind all systems to the same final allowlist, three captions, crop schedule, text initialization, optimizer policy, global-step map, and full-gallery evaluator.
8. Report both retrieval directions and every seed.
9. Stratify errors by phase, direction, period, interaction type, and caption ambiguity.
10. Preserve negative comparator wins rather than hiding them in an appendix.

## 9. Final refresh requirement

Immediately before paper freeze:

1. rerun primary-source searches over arXiv and official proceedings for `two-person motion language retrieval`, `dyadic motion retrieval`, `interaction motion text`, `phase motion retrieval`, `wavelet text motion retrieval`, `frequency motion language`, and `periodic action language`;
2. search citing/related records for MIME, InterEdit, WaMo, PHASOR, SGAR, MRBench, CountLLM, Period-LLM, OVR, and PAMS;
3. verify version dates, titles, authors, venues, withdrawn status, and final proceedings metadata;
4. read any new direct neighbor beyond the abstract before changing the novelty statement;
5. add a dated delta record; do not overwrite this evidence silently;
6. do not add or remove preregistered systems based on observed validation/test results.

## 10. Refresh conclusion

The current primary-source search finds no single work that explicitly combines the full PhasePair target: two-person skeleton-language retrieval, ordered AB/BA identity, symmetric and directional interaction signals, six fixed multi-frequency/phase relation slots, and the frozen matched-control family.

However, every broad component is already pressured by prior work:

- two-person interaction retrieval by MIME;
- wavelet multi-frequency retrieval by WaMo;
- two-person S/D-DCT frequency controls by InterEdit;
- directional relation retrieval by SGAR;
- periodic language reasoning by CountLLM, Period-LLM, and OVR;
- multi-scale phase learning by PAMS;
- phase-manifold retrieval by PHASOR;
- three-level motion captions and granularity sensitivity by MRBench.

Accordingly, PhasePair's novelty is provisional, combinatorial, narrow, and empirically falsifiable. It becomes an accepted project claim only after the exact controls, three-seed evidence, source-cluster inference, and final pre-submission primary-source refresh are complete.

# PhasePair InterHuman B0 evidence refresh — public redacted

**Observed:** `2026-08-25 01:28:46 +08:00` / `2026-08-24T17:28:46.120Z`  
**Scope:** Authorized read-only B0 byte, split, census, quarantine, commitment-presence, allowlist, mount, and rights refresh.  
**Privacy:** No sample identifier, caption, motion value, absolute private path, HMAC secret or row, account, host, or server detail is present.  
**Execution boundary:** No training, GPU work, model inference, Git operation, server access, or server mutation occurred.

## Verdict

`PASS_INTERNAL_RECEIPT_CONSISTENCY / PASS_RAW_ASSET_BYTES / HOLD_B0 / NOT_READY_FOR_SERVER_STAGING`

The existing exact and public-redacted receipts are internally consistent, and every receipt-bound archive, license, and split object has exactly one local private byte match. The frozen conservative exact cross-split duplicate quarantine is also internally consistent. B0 is not complete: the current `7,123` rows are a legacy pre-near allowlist, not the training-final allowlist, and the required exhaustive near-duplicate, PhasePair lineage, rights-attestation, topology, and live-mount gates are absent.

## Governing artifact witnesses

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| End-to-end objective v2 | 37,024 | `ee024c28d8abccb0f8b1af5634d479afce3c634185c1782ca094757ef8d51dd9` |
| Experiment plan 165840 | 10,042 | `335d48ec91609364509a7bb922995d1a60a8b4013f8c689e74afc4394ade8ac3` |
| Research contract 165840 | 10,613 | `1e8b90184acb04e5af298aa2be28585500a529b1ca32594952634e6b1fa06db4` |
| Asset contract 065237 | 22,545 | `251fce1f9506f6d1efa498b035e1d664d06b5ca2018d9d58ffb19d3d0fa3f24f` |

The objective hash exactly matches the authorized v2 witness supplied for this execution.

## Receipt-byte reverification

| Evidence object | Bytes | SHA-256 | Exact private matches | Result |
|---|---:|---|---:|---|
| Private exact receipt | 1,118,001 | `b8cfce6c8c50a2f8891c85bab5c90636d26703925dc7502ae4613a861b36404b` | 1 | PASS |
| Public-redacted receipt | 23,390 | `b4f15d6a44463eb972eac8012c66d9446b2a1f97f35ea1be9a2474050c3c8720` | 1 | PASS |
| Independent aggregate audit | 912 | `591351fccee06131a5d37ad382037e1377da4f4404144ee2d2808462d5ef3065` | 1 | PASS |
| Retired legacy wrapper | 716 | `fa5a45812ef56bf2c03208b16d3e6f6cb175a8751b4a2746b74f5fa1e58d8361` | 1 | RETIRED / NONAUTHORITATIVE |

Strict JSON parsing found no accepted duplicate object key or non-finite token in the authoritative receipt chain. The public-redacted receipt reports zero redaction violations, and its aggregate projection matches the private receipt recomputation.

## Raw-object byte reverification

| Official object | Bytes | SHA-256 | Exact private matches | Result |
|---|---:|---|---:|---|
| Source archive | 19,768,940 | `3c536bc01c4f2197898edd478284aa0ea87fe7577170e6ecb58e3e819b261e18` | 1 | PASS |
| Processed-motion archive | 2,462,028,219 | `ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325` | 1 | PASS |
| Annotation export | 2,687,043 | `aa0f4cd3f92745c9ad6d7acd6ac5c5b960cf1236c2c0fd1753c5c341e7900364` | 1 | PASS |
| Governing dataset license | 5,583 | `288f1d53a46457a5426e83c16aec7339b63eb9ea5e7c13e6632b0192e49382d4` | 1 | PASS |
| Train split | 35,466 | `d20765b028920fa2eeda0e9c73f66ea222348d3b36735562ee81fdeee630701b` | 1 | PASS |
| Validation split | 3,337 | `4182c70efb124bf167018d0586af20348e405af16ee98e48f36d50503ec51df2` | 1 | PASS |
| Test split | 6,762 | `c34e91a66e51ec1c9283b026e632ae330431e568ece0821858c5b2fb212f1dd1` | 1 | PASS |

This proves the receipt-bound archive and split bytes are locally available. It does not substitute for the absent live extracted installation or for rerunning the final near-duplicate and lineage producers.

## Recomputed split and census consistency

| Stage | Train | Validation | Test | Total | Captions |
|---|---:|---:|---:|---:|---:|
| Official split | 6,022 | 580 | 1,177 | 7,779 | — |
| Strict usable after defect quarantine | 5,635 | 547 | 1,098 | 7,280 | 21,840 |
| Exact cross-split members removed | 71 | 30 | 56 | 157 | — |
| Legacy pre-near allowlist | 5,564 | 517 | 1,042 | 7,123 | 21,369 |

Additional exact checks:

- official split pairwise overlap: `0`;
- paired processed motions: `7,777`;
- missing motions: `2`;
- sequences shorter than 15 frames: `494`;
- blank-caption rows: `2`;
- raw-only non-UTF-8 caption rows: `2`;
- every strict-usable row has exactly three counted captions;
- no split reassignment is present in the legacy pre-near allowlist.

## Conservative exact cross-split quarantine

- actor-swap-invariant exact groups: `78`;
- exact cross-split pairs: `79`;
- unique quarantined members: `157`;
- every group spans at least two official splits;
- all `157` strict-valid group members are quarantined;
- none remains in the `7,123`-row legacy pre-near allowlist.

The receipt's projection/top-5 near-candidate figures remain retired and nonauthoritative under the governing contract. The required authority is a new exhaustive `9,870,181`-pair conservative screen; therefore no near-threshold count from the old receipt is used as a final allowlist decision.

## Commitment and final-allowlist status

The legacy receipt contains nine well-formed aggregate membership commitment fields, but it does not contain the later PhasePair eligibility artifacts. The following gates remain absent:

- exhaustive near-duplicate v1 receipt and its training-final allowlist;
- source-lineage v1 receipt;
- PhasePair actor-v2 manifest proving unconditional global uniqueness across `2*N_final` actor slots;
- globally unique order-invariant pair commitments bound to the final allowlist;
- caption-lineage manifest with exactly `3*N_final` source-lineage/ordinal commitments;
- composite lineage receipt binding final allowlist, actor/pair manifest, and caption manifest.

Consequently, actor uniqueness, pair uniqueness, caption uniqueness, final membership, and lineage closure are `NOT_VERIFIED`, not failed data claims.

## Live-mount prerequisite

The receipt declares six POSIX private-data directories. In the configured WSL environment, `0/6` are currently available. Their literal locations remain private.

The exact prerequisite is to restore the receipt-bound POSIX private data installation inside that WSL environment so all six declared roots exist and resolve the source, annotation, split, InterHuman, and processed-motion objects without changing their recorded bytes. After the mount is restored, the next valid B0 action is the frozen exhaustive `9,870,181`-pair screen, followed by automatic regeneration of the training-final allowlist and all source/actor/pair/caption/composite lineage artifacts. The user need not repeat the existing data-access authorization.

## Rights status

The governing data-license object is byte-verified at SHA-256 `288f1d53a46457a5426e83c16aec7339b63eb9ea5e7c13e6632b0192e49382d4`. Under the existing conservative receipt interpretation:

- noncommercial scientific research is conditionally eligible only when all single-user, non-transferable, noncommercial, and controlled-computer facts are satisfied;
- private server copying remains on hold because the controlled-computer condition is asserted but not independently receipt-bound;
- `OWNER_LICENSE_ATTESTATION_PASS` is absent;
- `ONE_ARCHIVE_TOPOLOGY_PASS` is absent;
- dataset/sample redistribution and commercial use are `NO_GO`;
- public trained weights are `NO_GO` absent written permission or qualified legal review;
- public aggregate metrics remain on hold for owner or qualified-legal confirmation.

## Server-staging decision

`NO — B0 is not ready for server staging.`

Required before staging:

1. restore and revalidate the receipt-bound live mount;
2. complete exhaustive-near-v1 and freeze the new final split census/allowlist;
3. produce and verify source, actor, pair, caption, and composite lineage receipts;
4. obtain the owner-license attestation and one-archive-topology PASS receipts;
5. rerun the no-raw-egress/public-redaction gate on the completed receipt chain.

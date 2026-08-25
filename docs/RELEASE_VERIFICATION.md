# PhaseSet v0.2.1 release verification

This procedure covers the two release boundaries that
`RELEASE_FILES.sha256` cannot establish by itself: wheel packaging and the
post-publication GitHub asset set. It does not authorize real-data claims or
replace the test, public-tree, all-ref, codeload, and legacy-tag gates.

## 1. Before tagging: bind the wheel to the staged index

Build only after every intended file and the regenerated
`RELEASE_FILES.sha256` are in the Git index. Both commands below are required:

```text
python scripts/audit_release_artifact.py dist
python scripts/verify_wheel_against_index.py dist
```

The first command scans archive structure, forbidden paths, secrets, and local
locators. The second command reads `pyproject.toml` and every expected package
source via `git cat-file blob :path`, never from an unstaged worktree. It
requires the wheel's complete `.py` file set and bytes to equal the staged
`src/phasepair_core/` and `src/phaseset_core/` set, and checks the wheel
filename, distribution metadata, version, dependencies, complete entry-point
table, closed member allowlist, non-executable modes, and full `RECORD`
hash/size coverage.

Before uploading anything, create canonical
`expected-release-assets.json` from the clean annotated-tag checkout. It must
bind repository/tag/commit/tree and every intended payload name, byte size,
SHA-256, kind, and source path. Freeze its SHA-256 in the external release
record before publication. The wheel row uses `WHEEL_FROM_TAG_INDEX`; the
paper row uses `TAG_TREE_FILE` with `paper/main.pdf`. Other deterministic
tag-tree exports use `TAG_TREE_FILE`; a generated artifact without a direct
tree byte counterpart may use `PREPUBLICATION_BUILD`, whose exact digest is
still fixed by this independently frozen manifest.

## 2. Publish payload assets, then build the receipt

Create the annotated `v0.2.1` tag and GitHub release, and upload every payload
asset except `PHASESET_v0.2.1_ASSET_INVENTORY.json`. Capture GitHub REST bodies
and download those uploaded payloads into a new directory. The five required
responses are:

- `GET /repos/appleweiping/phaseset-multiperson-motion-language`;
- `GET /repos/appleweiping/phaseset-multiperson-motion-language/releases/tags/v0.2.1`;
- `GET /repos/appleweiping/phaseset-multiperson-motion-language/git/ref/tags/v0.2.1`;
- `GET /repos/appleweiping/phaseset-multiperson-motion-language/git/tags/{tag_object_sha}`;
- `GET /repos/appleweiping/phaseset-multiperson-motion-language/git/commits/{peeled_commit_sha}`.

The release response at this stage must not yet contain the inventory receipt.
Run:

```text
python scripts/release_asset_inventory.py build \
  --expected-repository-id 1345199252 \
  --expected-repository appleweiping/phaseset-multiperson-motion-language \
  --expected-tag v0.2.1 \
  --repository-json evidence/repository.json \
  --release-json evidence/release-before-inventory.json \
  --tag-ref-json evidence/tag-ref.json \
  --tag-object-json evidence/tag-object.json \
  --commit-json evidence/commit.json \
  --asset-dir uploaded-payload-download \
  --expected-assets-json evidence/expected-release-assets.json \
  --expected-assets-sha256 EXPECTED_MANIFEST_SHA256_FROM_EXTERNAL_RECORD \
  --tag-checkout clean-v0.2.1-checkout \
  --self-asset-name PHASESET_v0.2.1_ASSET_INVENTORY.json \
  --output uploaded-payload-download/PHASESET_v0.2.1_ASSET_INVENTORY.json
```

The builder first requires a clean checkout whose annotated tag, commit, and
tree equal GitHub evidence. It re-runs wheel-to-index verification on the
downloaded wheel, requires the released paper bytes to equal tag-tree
`paper/main.pdf`, checks every asset against the independently frozen expected
manifest, and performs structural privacy/secret scans on every supported
asset. Unknown asset types fail closed. Only then may GitHub's uploaded asset
IDs, byte sizes, and `sha256:` observations be accepted. It emits canonical UTF-8 JSON
with repository numeric ID/full name, release ID/tag, annotated tag object,
peeled commit, tree, and canonical name/size/SHA-256 rows. It deliberately
omits a row for itself, because a file cannot non-circularly authenticate its
own final bytes.

## 3. Upload the receipt and perform post-release verification

Upload the generated receipt. Then capture a fresh release response and
download every release asset, including the receipt, into another new and
otherwise empty directory. Run:

```text
python scripts/release_asset_inventory.py verify \
  --expected-repository-id 1345199252 \
  --expected-repository appleweiping/phaseset-multiperson-motion-language \
  --expected-tag v0.2.1 \
  --repository-json evidence-after/repository.json \
  --release-json evidence-after/release-with-inventory.json \
  --tag-ref-json evidence-after/tag-ref.json \
  --tag-object-json evidence-after/tag-object.json \
  --commit-json evidence-after/commit.json \
  --asset-dir fresh-complete-release-download \
  --expected-assets-json evidence/expected-release-assets.json \
  --expected-assets-sha256 EXPECTED_MANIFEST_SHA256_FROM_EXTERNAL_RECORD \
  --tag-checkout clean-v0.2.1-checkout \
  --receipt fresh-complete-release-download/PHASESET_v0.2.1_ASSET_INVENTORY.json
```

Verification fails on any missing/extra/re-uploaded asset, unsafe filename,
asset-ID drift, size or digest mismatch, expected-manifest mismatch, wheel/tag
index mismatch, paper/tag-tree mismatch, non-canonical receipt, repository/tag
substitution, lightweight tag, peeled-commit drift, or tree drift. The refreshed
GitHub response authenticates the receipt asset's observed size and SHA-256,
while the separately frozen prepublication manifest is the intended-content
authority. GitHub's digest is never treated as that authority.

## Trust boundary

The verifier intentionally performs no network request and cannot prove where
a JSON file came from. The operator must acquire the five response bodies from
GitHub over authenticated TLS, retain the command/time and GitHub request
metadata, pass the already registered numeric repository ID, full name, and tag
as explicit expectations, and use a fresh full asset download. A receipt plus
copied or fabricated JSON bodies is not external evidence. Record the response
and receipt SHA-256 values, Actions run IDs, tag object, commit/tree, and final
verification output in the release provenance record only after observation.

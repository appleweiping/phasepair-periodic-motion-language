# Immutable periodic descriptor cache

`periodic_descriptor_cache_v2` stores weight-independent physical pair
descriptors, not learned tokens or activations. The existing small periodic
cache stores energy-floor metadata; this separate format stores all three
descriptor streams: full relations, marginal power, and mean/difference DCT.
No licensed motion payload or private cache artifact is included here.

## Exact cache identity

A cache key binds the complete canonical numeric group batch, masks, padding,
actor/group lineage commitments, fixed Morlet oracle and energy floors. It
also binds explicit source/environment, manifest, batch, split, window ordinal,
seed, epoch and exact yaw identities. Commitments determine canonical reduction
order but do not become neural-network inputs.

These caller-supplied digests provide integrity and equality binding only;
cache construction does not authenticate their origin or grant dataset access,
consent, split, training or execution authority.

Reordering actors preserves canonical identity; changing dynamic batch packing
is a distinct key, not an inferred cache hit. Shared yaw includes floating-point
rotation and rounding, so another epoch's apparently rotation-invariant motion
is not assumed bitwise interchangeable. There is no approximate cross-epoch
reuse or silent uncached fallback for a configured wrong cache.

The bounded format preserves directed descriptor arrays, validity and endpoint
indices in canonical order. Re-iteration can yield requested runtime chunks
without changing the logical stream, including chunks spanning batch groups.
The same physical full-relation stream can underlie the controls that transform
it; marginal and DCT require their separately identified streams.

## Storage and resource boundaries

The writer checks expected pair count and decoded byte requirements from actor
counts before materializing a stream. The fixed decoded census is 1914 bytes
per edge for all stored arrays. Per-shard, decoded, total-byte and edge bounds
are explicit; an additional exact serialized-size check follows encoding.
Resource rejection does not drop actors, sample edges or truncate an artifact.

Artifacts are exclusively created and the index is published last. The reader
performs bounded owned reads, raw digest checks, strict schema checks and exact
dtype/shape/offset/payload digest validation. It checks static symlinks,
single-link payloads and byte bounds; loaded arrays are non-writeable. On
platforms providing `O_NOFOLLOW`, the owned read also requests no-follow
opening. On Windows the module is import-safe, but cache use requires a trusted,
stable, non-concurrently-mutated tree and does not claim POSIX-equivalent
no-follow or junction protection. Static symlink checks, single-link checks,
hashes and read-only flags are integrity hardening, not filesystem isolation
or data authorization.

## Verified boundary and unfinished integration

Nineteen server tests passed in 36.21 s with source unchanged. They cover all
three streams, a 186-edge K12/K16 mixed batch, chunks 64/128/256 including a
cross-group boundary, malformed authenticated artifacts, bounds, and unchanged
CPU Torch RNG. A small test-only encoder seam also checks exact cached versus
uncached parameter gradients through the existing custom backward replay.
The hardlink and oversize checks are independent: only an explicit hardlink
creation capability error can skip that case; unknown I/O errors fail, and the
oversize check still runs. All nineteen ran successfully in this Linux
observation. Its private receipt digest is
`8edbb675a64773fe104a1eae7b4bf398262b61827199152833da1e0765237f41`;
the receipt itself and its private paths are not published.

That test seam is not a production cache provider: it deliberately supplies
fixture chunks. Production model entry points, training initialization,
checkpoint/resume bindings and complete epoch-cache census still require
explicit integration. No real-data cache, training acceleration, measured
throughput or model-quality result is claimed. The unchanged-source 1211-test
full Linux server regression described in
[base qualification](BASE_COHORT_QUALIFICATION.md) used this exact production
module and the preceding eighteen-test file. The only subsequent test change
split hardlink and oversize checks; those newer bytes were separately verified
by the nineteen-test observation above.

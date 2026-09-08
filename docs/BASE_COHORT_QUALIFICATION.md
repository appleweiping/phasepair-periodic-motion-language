# Base cohort latency and qualification

The latency runner and deterministic assembler extend the
[completed-base resolver and capture scorer](BASE_COHORT_VALIDATION.md).
They implement a software path from nine completed training chains to the
existing qualification selector. No real nine-run cohort, winner or latency
benchmark has yet been observed.

## One complete latency session

`run_base_cohort_latency_session` reuses the admitted, validation-selected
checkpoint bytes for all three bases and all three seeds. It checks the
installed source, selected state, runtime, validation census and live parameter
census against the resolver and scorer. It does not accept supplied timings
or parameter counts.

The frozen workload is an analytic K=32, T=200, body-22 group at the registered
512-dimensional model width. It is an engineering timing workload, not a
replacement dataset. Nine cyclic rounds visit all nine rows: 81 visits in
total. Each visit has five warmups and eleven synchronized timed forwards,
giving 99 samples per row. Output copying and finite-value checking are inside
the defined forward interval; progress-journal I/O is outside it. The statistic
is the exact integer median of all 99 positive nanosecond samples.

The runner records the actual CUDA UUID, resolved device, runtime and precision.
Its shared-device policy requires at least 8 GiB free and a 2 GiB allocator cap.
An admission, resource, identity or execution failure holds the entire session.
It does not discard slow rows, lower K, remove layers, switch checkpoints or
select a quiet subset of visits. The controller must predeclare and freeze an
appropriate outer wall timeout before the actual session. A short functionality
probe's timeout is not the formal session timeout. Existing unrelated device
workloads are not interrupted.

The injected runtime seam is explicitly nonformal and exists for contract
tests. It cannot produce an accepted actual-runtime qualification merely by
finishing the same number of loops.

An actual registered-server diagnostic found that NVIDIA returned the complete
`GPU-`-prefixed UUID while Torch's CUDA UUID object rendered the same payload
without that prefix. The runtime now accepts only those two complete canonical
lowercase representations from Torch and compares the resulting full UUID
exactly against both registration and the NVIDIA observation. Short forms,
whitespace, uppercase payloads, repeated prefixes, MIG identifiers and changed
payloads are rejected; this conversion cannot select or change the device.
Tests prove invalid identities fail before allocator setup or tensor creation,
and accepted representations preserve the original allocator cap. The original
failed K32 observations remain retained; this identity correction alone is not
a completed K32 model run or a formal latency session.

The following actual K32 retry passed device initialization but failed the
strict zero-allocation check when releasing B0. No completed model row was
returned. Its failure and unchanged-source receipts are retained; allocator
cleanup diagnosis is ongoing without changing the workload or resource limit.

The integrated UUID/source-lineage/text-boundary CPU regression passed 136
focused tests and 1257 full tests, two existing skips and 39 subtests in
413.42 seconds with source unchanged. Its CUDA API stand-ins are explicitly
software tests, not a substitute for the separate actual device observation.

## Qualification assembly

`assemble_base_cohort_qualification` accepts the resolved cohort, admission,
complete capture-scoring result and complete actual latency session. It accepts
no submitted score rows, winner, timing samples or evaluator digest. It checks
the complete row order and binds each terminal, selected checkpoint, selected
state, query census, precision, runtime and installed-source identity across
the three observations.

Each `BaseScore` obtains its exact `Fraction` metric from the capture scorer,
its parameter count from matching scorer/latency live censuses, and its latency
from recomputing the retained 99-sample median. Its combined artifact digest
binds both the score and latency observations. The evaluator identity is the
installed production qualification artifact, not an unrelated module digest.

The assembler calls the existing `experiments.qualify_base` and production
canonical verifier. Selection remains: highest exact three-seed mean primary
score; then fewer parameters, lower three-seed median latency, and smaller base
ID. It does not implement a second competing selection rule. Selected bytes
and scoped source dependencies are checked for stability around assembly.

The result remains a software artifact with `authority=0`, `production=False`
and `result_claimed=False`. The trusted private controller must use actual
independently retained execution evidence through the existing backend flow.
This does not introduce an extra user-signature requirement, turn hashes into
data rights, or authorize access to sealed test data.

## Progress, verification and remaining work

The optional typed observer connects to the
[durable progress journal](LATENCY_PROGRESS_JOURNAL.md). A journal terminal is
not a substitute for the actual latency call outcome and its receipt.

The latency contract passed 45 focused server tests; qualification assembly
passed a 53-test combined suite. The final observer/journal integration passed
73 tests in 22.63 s. The merged source, also containing the descriptor cache,
passed 1211 tests, two existing skips and 39 subtests in 362.75 s on the Linux
experiment server, with its complete source census unchanged. These counts
overlap and must not be added as independent experiments.

Tests use explicit software fixtures and injected observations. No actual
81-visit CUDA timing session, real completed training cohort, qualified winner
or host `qualify-base` command completion is claimed. Private host command
composition and all data-dependent execution remain unfinished.

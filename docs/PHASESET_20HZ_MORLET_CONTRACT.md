# PhaseSet 20 Hz Morlet rate contract

**Schema:** `phaseset-morlet-bank-v1/fs20/three-cycle`  
**Frozen before real-data execution:** 2026-08-25 +08:00  
**Status:** `DATA_FREE / AUTHORITY0 / NO_RESULT`

PhaseSet motion is antialiased and resampled from 30 fps to 20 fps before its
five body-group activity channels are formed. Applying PhasePair v0.1.0's
30-Hz tap bytes directly to those samples would silently move every physical
center frequency to two thirds of its registered value. PhaseSet therefore has
an independent 20-Hz formula bank. It does not relabel, overwrite, or claim byte
equivalence with the PhasePair bank.

## Frozen formula

The six physical center frequencies remain

```text
0.75, 1.125, 1.6875, 2.53125, 3.796875, 5.6953125 Hz
```

For sampling rate `Fs=20 Hz`, frequency `f`, support index `u`, support length
`L`, and Gaussian width `sigma`, PhaseSet fixes

```text
L       = ceil(3 Fs / f)
sigma   = Fs / (2 f)
g(u)    = exp(-u^2 / (2 sigma^2))
c(u)    = g(u) exp(-j 2 pi f u / Fs)
beta    = sum(c) / sum(g)
psi_raw = c - beta g
psi     = psi_raw / sqrt(sum(|psi_raw|^2))
```

The resulting supports are `80,54,36,24,16,11` samples. Every kernel is
complex128, zero-DC to at most `2.220446049250313e-16`, and unit-energy to at
most the same bound.

The oracle serializes each band as an unsigned 16-bit big-endian support length
followed by every complex tap as big-endian float64 real and imaginary values.
Its SHA-256 is

```text
633d754bba12dab0555722f4b8963409a110c8681796cc907d163096e2e42655
```

The PhasePair v0.1.0 30-Hz digest remains
`4f75ab687d333af8b436feea371dbaee56c55256e91c42c5e1ce45935beed5d7`.
The two digests are intentionally different.

## Compatibility boundary

- PhasePair's six frequencies, 30-Hz taps, 13D descriptor tests, endpoint-swap
  algebra, tag, and golden oracle remain unchanged.
- PhaseSet reuses the 13D directed relation field definitions and delay algebra,
  but its spectral responses come only from the independent 20-Hz bank.
- Missing activity samples propagate to Morlet response masks; pair pooling uses
  only response positions observed for both endpoints.
- A band edge is energy-valid only when both endpoint auto powers are strictly
  greater than the frozen floor. Thus exact zero energy is invalid even when a
  diagnostic floor is zero. Coherence is retained as a numerical feature and
  is never an edge-deletion threshold; a positive-energy, zero-coherence edge
  remains present.
- No implicit upsampling or hidden `2/3` frequency relabelling is permitted.
- A PhaseSet descriptor is therefore not expected to be bitwise equal to a
  PhasePair descriptor for the same numeric sample array.

## Deterministic probes

The public test suite rebuilds and hashes both banks independently, verifies
zero DC and unit energy, scans each PhaseSet kernel's response peak on a 20-Hz
frequency grid, and requires the registered center response to exceed the
silently shifted `2f/3` response by more than 1.7 times.

For every registered band it also generates a nonnegative sinusoidal activity
at the exact physical frequency. The target band must have the largest marginal
response. Delaying the second actor by one 20-Hz sample must recover phase
`-2 pi f / 20`, delay `0.05 s`, and the opposite directed-delay sign after
endpoint exchange.

# OpenMM-on-chipStar: Failure Analysis and Outstanding Issues

A complete write-up of OpenMM single-precision HIP test failures on Intel
GPUs (Arc and PVC), the underlying root cause of each, and which layer is
responsible (chipStar, IGC, VkFFT, OpenMM itself, or the SPIR-V spec).
Companion to `upstreaming.md`, which is OpenMM-PR-focused; this file is
failure- and root-cause-focused.

Status taken on 2026-04-29:

- **Intel Arc A770**, single precision, chipStar `fix-1191`: **52 / 60 pass (87%)**.
- **Aurora PVC** (Intel Data Center GPU Max 1550), single precision,
  chipStar `2026.04.29`: **49 / 60 pass (82%)**.
- 8 failures are common to Arc and PVC; 3 additional failures are PVC-only.

---

## 1. Failure inventory

### 1.1 Common to Arc and PVC (8 tests)

| Test | Failure mode | Root cause | Owner |
|---|---|---|---|
| `TestHipCustomManyParticleForceSingle` | Either fails compilation (`hipErrorLaunchFailure: kernel not found`) or wrong forces | IGC `RetryManager` drops kernel that's too large for SIMD16; partial workaround (FORCE_WORKGROUP_SIZE pinned) helps but does not fully fix | **IGC** (silent drop), **OpenMM** (kernel size) |
| `TestHipCustomNonbondedForceSingle` | Wrong forces | JIT correctness bug in custom-expression code path with certain expression trees; `expandConstant` does not handle `ConstantAggregate` (chipStar #582) | **chipStar** (#582) + **IGC** (custom-JIT correctness) |
| `TestHipFFT3DSingle` | SEGFAULT | VkFFT/IGC crash on certain non-power-of-2 sizes; PVC and Arc have different WGS limits | **VkFFT** + **IGC** |
| `TestHipGayBerneForceSingle` | Wrong forces (small mismatches) | Not yet isolated — likely SIMD16 race in the GB tile loop, but has not been instrumented like AMOEBA was | **chipStar** (compile model) — needs investigation |
| `TestHipAmoebaExtrapolatedPolarizationSingle` | Wrong forces | Same SIMD16 lockstep race as AMOEBA fixed-field/induced-field/electrostatics; extrapolated-polarization tile loops have not been patched yet | **chipStar** (compile model) — fix follows AMOEBA pattern |
| `TestHipAmoebaGeneralizedKirkwoodForceSingle` | Wrong forces | Same SIMD16 race; GK kernels (`amoebaGk.cc`, `gkPairForce1.cc`, `gkPairForce2.cc`) have analogous tile loops; partial uncommitted refactor in progress | **chipStar** (compile model) — fix follows AMOEBA pattern |
| `TestHipAmoebaMultipoleForceSingle` | `testPMEMutualPolarizationLargeWater` fails ~85% of runs with ~0.1% force variance above the 1e-4 tolerance | Non-deterministic neighbor-list build (separate bug from the SIMD16 race that was fixed for the smaller AMOEBA tests) | **OpenMM** (neighbor list) — needs isolation |
| `TestHipHippoNonbondedForceSingle` | Wrong forces | Same SIMD16 race; `hippoNonbonded.cc`, `hippoComputeField.cc` have analogous tile loops; not yet patched | **chipStar** (compile model) — fix follows AMOEBA pattern |

### 1.2 PVC-only (3 additional tests on Aurora 2026.04.29)

These pass on Arc fix-1191 but fail on PVC. They are not yet isolated.

| Test | Failure mode | Suspected cause |
|---|---|---|
| `TestHipCustomIntegratorSingle` | Fails after 101 s | JIT-compile slowness or timing-sensitive path on PVC; could be private-memory pressure picking SIMD8 |
| `TestHipVariableVerletIntegratorSingle` | Wrong forces / energies | Adaptive-step integrator; PVC-specific FP rounding or WGS choice |
| `TestHipAmoebaVdwForceSingle` | Wrong forces | AMOEBA family but distinct kernel from the SIMD16-race set; needs isolation |

---

## 2. Root causes by layer

This section groups the actual bugs by where they live. Each issue lists the
OpenMM workaround (commit + file) so the patch can be removed when the
underlying bug is fixed.

### 2.1 chipStar bugs with closed PRs that need reviving

These are the three commits in branch `fix-1191` not in `origin/main`. Each
has an associated PR that was closed without merging. The fix is required
for OpenMM to run reliably.

| Issue | PR | Status | Fix needed for |
|---|---|---|---|
| [#1191](https://github.com/CHIP-SPV/chipStar/issues/1191) — process wedges on Intel GPU hangcheck reset | [#1232](https://github.com/CHIP-SPV/chipStar/pull/1232) | **CLOSED** | `testPMEMutualPolarizationLargeWater` and any long-running kernel — `UINT64_MAX` `zeCommandListHostSynchronize` never returns when GPU resets; replace with polling loop. |
| [#1090](https://github.com/CHIP-SPV/chipStar/issues/1090) — `clSetEventCallback` deadlock on `ArgSpillBuffer` | [#1229](https://github.com/CHIP-SPV/chipStar/pull/1229) | **CLOSED** | Concurrent kernel launches that spill arg buffers — replace callback with event `KeepAlives`. |
| [#1142](https://github.com/CHIP-SPV/chipStar/issues/1142) — HIPRTC disk-cache key collisions | [#1231](https://github.com/CHIP-SPV/chipStar/pull/1231) | **CLOSED** | Reliable cross-run kernel cache — `std::hash` collides; use FNV-1a. |

### 2.2 chipStar bugs without an upstream PR

| Issue | What it is | OpenMM impact |
|---|---|---|
| `#555` | `hipGraph*` API missing parameter validation | Low — OpenMM does not exercise graphs heavily |
| `#556` | `hipGraphAddDependencies` / `RemoveDependencies` pair iteration bug | Low |
| `#582` | `expandConstant` does not handle `ConstantAggregate` | **Medium** — surfaces during JIT of custom-expression kernels (`CustomNonbonded`) |

### 2.3 chipStar API gaps OpenMM has to work around

#### 2.3.1 `hipFuncGetAttribute(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)`

Returns `hipErrorNotSupported`. OpenMM falls back to the device max block
size.

- chipStar fix: implement the query.
- OpenMM workaround: `HipContext.cpp` (commit `d7996ce80`) —
  `getMaxThreadBlockSize()` falls back to `props.maxThreadsPerBlock`.

#### 2.3.2 `hipMemcpyHtoD` / `hipMemcpyHtoDAsync` ABI mismatch on HIP-7 update

When chipStar pulled in the HIP-7 submodule (April 2026), the public header
declared `const void* src` but `CHIPBindings.cc` still defined `void* Src`.
Linkers got two different symbols.

- **Already fixed** in chipStar [PR #1244](https://github.com/CHIP-SPV/chipStar/pull/1244) (`9e74322d`).
- Regression test added at `tests/regression/test_hipMemcpyHtoD_link.cpp`.

### 2.4 IGC bugs that chipStar has to work around (and OpenMM has to follow)

#### 2.4.1 SIMD16 lockstep race in 32-wide HIP warps

On Intel GPUs (both Arc and PVC), a 32-wide HIP warp is compiled as **two
SIMD16 hardware threads** that do not execute in lockstep. Any kernel that
writes `localData[index]` and later reads `localData[other_index]` across
iterations without a barrier races.

- Affected: `multipoleFixedField`, `multipoleInducedField`,
  `multipoleElectrostatics` (fixed); `Hippo`, AMOEBA-GK, AMOEBA extrapolated
  polarization (still failing — same pattern, not yet patched); base
  nonbonded exclusion-tile reload (fixed).
- IGC fix: enforce HIP warp-lockstep semantics across the SIMD16 split, or
  document the contract.
- OpenMM workaround: `SYNC_WARPS` / `__syncwarp()` inside the tile loop,
  guarded by `__HIP_PLATFORM_SPIRV__` so AMD pays no cost. Commits
  `78ecd47c0` (AMOEBA) and `60b454e9a` (base nonbonded).
- L0+SPIR-V reproducer at `~/reproducers/cas_atomicadd_l0/` confirmed
  atomics themselves are not the problem — the race is in shared-memory
  accumulation.

#### 2.4.2 IGC inliner explodes register pressure, kernel silently dropped

IGC's inliner inlines AMOEBA / nonbonded helper functions into a single
kernel-sized megafunction, blowing up register pressure. `RetryManager`
then silently drops the kernel, surfacing later as `hipErrorLaunchFailure:
kernel not found`.

- chipStar root cause: SPIR-V `DontInline` decoration is not preserved
  across chipStar's lowering, so source-level `__attribute__((noinline))`
  has no effect on what IGC sees.
- IGC fix: surface a hard error when `RetryManager` drops a kernel.
- chipStar fix: have its SPIR-V emitter honor source `noinline`.
- OpenMM workaround: post-process SPIR-V in `HipContext.cpp` to set
  `FunctionControl=DontInline` on specific helper symbols
  (`_computeInteractionHelper`, `computeOneInteractionF1/F2/T1/T2/B1/B2`).
  Commits `14f724fc8`, `645a5580c` plus expansions.

#### 2.4.3 `int64 OpAtomicIAdd` fires 4× per call (SubgroupSize 32)

[`intel/intel-graphics-compiler#397`](https://github.com/intel/intel-graphics-compiler/issues/397)
— at SIMD32, IGC emits a single int64 atomic add as four separate atomic
ops, one per quarter-warp. Breaks any kernel using
`atomicAdd(unsigned long long*, ...)`.

- Blocks **all** double-precision OpenMM on Intel.
- Single-precision workaround: int32 CAS loops in
  `platforms/hip/src/kernels/common.hip` behind
  `USE_INT64_ATOMIC_ADD_WORKAROUND`. Commit `a4cdbf279`.

#### 2.4.4 Kernel size → `RetryManager` silent drop

Even without inlining, some kernels are too big for SIMD16. The user sees
`hipErrorLaunchFailure: kernel not found` rather than a compile error.

- IGC fix: hard error.
- OpenMM workaround:
  - RPMD: split `integrateStep` into two kernels and add `fftImag`
    parameter (`230234ce0`, `89311da2a`).
  - `CustomManyParticle`: pin `FORCE_WORKGROUP_SIZE` so IGC picks SIMD32
    (`732a9bdec`, `95c9bdaed`, `b97dfc149`).

#### 2.4.5 `-ffast-math` produces incorrect results

IGC + `-ffast-math` miscompiles several numeric kernels.

- chipStar fix: filter `-ffast-math` from HIPRTC options on Intel.
- OpenMM workaround: drop `-ffast-math` from JIT options on
  `__HIP_PLATFORM_SPIRV__` (commit `5cdc834b6`).

#### 2.4.6 Aligned struct byval args miscompile

IGC miscompiles aligned struct byval kernel arguments. Reproducer is
chipStar [PR #1185](https://github.com/CHIP-SPV/chipStar/pull/1185) (DRAFT).

- Manifests as some of the noinline-helper failures.

### 2.5 SPIR-V semantics divergences from CUDA / AMD

These are not bugs per se — they are places where SPIR-V (as IGC implements
it) returns a different value than CUDA or AMD GCN for the same operation.
They will probably never be "fixed" upstream because they are
spec-conformant, but they have to be guarded in OpenMM source.

#### 2.5.1 `static_cast<long long>(NaN * scale)` returns `INT64_MAX`

CUDA: 0. AMD: 0. SPIR-V/IGC: `INT64_MAX`. This silently corrupts OpenMM's
fixed-point force buffer in `realToFixedPoint` whenever an intermediate
goes NaN.

- OpenMM workaround: explicit integer-bit NaN check before the cast
  (commits `f76334dc0`, `b9859292f`, `a14baff9c`).
- Inf is fine (saturating cast covers it).

#### 2.5.2 `rsqrt(0.0)` returns finite garbage instead of `+inf`

CUDA / AMD: `+inf`, so the term self-masks at self-interaction. SPIR-V/IGC:
finite garbage that contaminates the force.

- OpenMM workaround: explicit `r2 == 0` guard (commit `f3630fcb6`).

### 2.6 VkFFT / FFT integration

VkFFT's default WGS / dimension limits exceed Arc's per-target limits on
some non-power-of-2 sizes; PVC behaves differently again.

- OpenMM workaround: per-target adjustments in `HipFFT3D.cpp` and a CPU FFT
  fallback for single precision (commits `90a916891`, `f2c7545e3`,
  uncommitted `HipFFT3D.cpp` changes).
- Surfaces as `TestHipFFT3DSingle` SEGFAULT when the workaround is not
  comprehensive enough.

### 2.7 Arc private-memory cap

On Arc, WGS > 512 in `computeBucketPositions` spills to private memory and
exceeds the per-thread limit.

- OpenMM workaround: cap WGS at 512 on Intel (commit `4e6eda8f5`).

### 2.8 OpenMM-side issues

A few failures are not chipStar/IGC's fault — they are OpenMM bugs that
only surface on Intel because the timing is different.

- **Non-deterministic neighbor-list build** in
  `testPMEMutualPolarizationLargeWater`: causes ~0.1% force variance above
  the 1e-4 tolerance ~85% of runs. Not a chipStar bug — happens because the
  GPU race fixed in `78ecd47c0` was *one* race, and there is a second one
  in the neighbor-list construction that has not yet been isolated.
- **Minimizer fragility on bad initial state**: the minimizer can drive a
  system to NaN/Inf energies on the GPU, and there was no fallback. Commit
  `884d773b3` adds a CPU fallback when the GPU energy is huge or NaN.

### 2.9 Kernel-source synchronization gotcha (not a bug, but expensive)

Lesson learned this session: when rsyncing the OpenMM tree to a remote
build host, **do not** use `--exclude='*.hip'` — that will exclude the
legitimate kernel sources in `platforms/hip/src/kernels/*.hip` (since
chipStar emits temp `*.hip` files in the working dir, the exclusion is
tempting). Use a more specific pattern that only excludes the temp
filenames (e.g. hash-prefixed `[0-9a-f]*.hip`).

This produced 5 hours of false debugging into "fp64 dropping" before
realizing the kernel string passed to HIPRTC literally did not contain the
kernel.

---

## 3. Open chipStar PRs adjacent to this work

- [#1161](https://github.com/CHIP-SPV/chipStar/pull/1161) — switch CI to
  hip-tests by default + HIP-7 follow-ups. **OPEN**, not a blocker.
- [#1185](https://github.com/CHIP-SPV/chipStar/pull/1185) — IGC byval
  reproducer. **DRAFT**. Related to noinline workarounds.
- [#1244](https://github.com/CHIP-SPV/chipStar/pull/1244) — `hipMemcpyHtoD`
  ABI fix. **MERGED** 2026-04-29.

---

## 4. Recommended chipStar work, in priority order

1. **Revive PRs #1229, #1231, #1232.** All three closed; all three needed.
2. **Implement `hipFuncGetAttribute(MAX_THREADS_PER_BLOCK)`.**
3. **Fix SPIR-V `noinline` propagation** so the runtime SPIR-V patcher in
   `HipContext.cpp` can be deleted.
4. **Surface a hard error when IGC `RetryManager` drops a kernel** — the
   silent-drop behavior is the single biggest source of bug-hunting time.
5. **Filter `-ffast-math`** in chipStar's HIPRTC options on Intel.
6. **Open PRs for chipStar issues `#555`, `#556`, `#582`.**
7. **Investigate the 3 PVC-only failures** (§1.2) — they may be
   per-target IGC issues but chipStar should triage them first.
8. Track `intel/intel-graphics-compiler#397` and remove the int64 atomic
   workaround when fixed.

## 5. Recommended OpenMM-side work

1. **Apply the AMOEBA `SYNC_WARPS` pattern** to GK, Hippo, and AMOEBA
   extrapolated polarization tile loops. Should clear 3 of the 8 common
   failures.
2. **Isolate the `testPMEMutualPolarizationLargeWater` neighbor-list race**
   (separate from the AMOEBA fix).
3. **Investigate the 3 PVC-only failures** with `CHIP_LOGLEVEL=info` to
   confirm whether they are SIMD-width or timing related.
4. **Audit the GayBerne tile loop** for the same SIMD16 race pattern.

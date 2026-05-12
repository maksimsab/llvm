# Plan: Mitigate sycl-post-link Debug Info Overhead for -O0 Builds

## Context

In large SYCL projects built with `-O0`, `sycl-post-link` consumes excessive RAM and time. The root cause is LLVM debug metadata (DICompileUnit, DIFile, DISubprogram, DIType, etc.) from all translation units accumulating in the single linked LLVM IR module. `sycl-post-link` then calls `CloneModule` once per split, and each clone deep-copies the entire debug metadata graph via `MapMetadata` on `!llvm.dbg.cu`. Peak memory is **O(N × debug-metadata-size)** where N = number of splits.

The bulk of that metadata in a typical SYCL project comes from **system headers** (STL, SYCL runtime headers) — type hierarchies from includes like `<vector>`, `<tuple>`, `<type_traits>`, the SYCL `accessor` template tree, etc. These types are rarely useful for device-side debugging.

## Root Cause Chain

1. Every TU compiled with `-g` emits full DWARF metadata, with most bulk from system-header types
2. `llvm-link` joins all user `.bc` files unconditionally — all debug metadata accumulates
3. `sycl-post-link` receives this module; `extractSubModule()` (`ModuleSplitter.cpp:368`) calls `CloneModule(M, VMap, predicate)`
4. `CloneModule.cpp:183–186`: iterates every `NamedMDNode` including `!llvm.dbg.cu`, calling `MapMetadata()` on each operand — recursively clones the entire type graph into every split module
5. `ModuleDesc::cleanup()` runs `StripDeadDebugInfoPass` afterward but only removes debug info for DCE'd symbols, not live ones — every split ends up with a full copy of the type graph

## Recommended Implementation (3 changes, ordered by priority)

### Change 1 — Make `-fno-system-debug` the default for SYCL device compilation (source fix, best debug quality preserved)

**What it does:** `-fno-system-debug` (`CodeGenOpts::NoSystemDebug`, implemented in `clang/lib/CodeGen/CGDebugInfo.cpp:6716`) suppresses debug info generation for declarations in system headers at compile time. In SYCL projects this eliminates the bulk of the DIType graph (STL types, SYCL header types) while preserving debug info for user code. Currently the flag is recognized but not automatically applied to device compilations.

**Files to modify:**
- `clang/lib/Driver/ToolChains/SYCL.cpp` — in the device compiler invocation path, add `-fno-system-debug` to device compiler flags unless the user explicitly passed `-fsystem-debug`. The right place is where device compiler arguments are constructed (near the `AddSPIRVImpliedTargetArgs` region, lines 1621–1803, or wherever device CC1 flags are assembled).

**Effect on debug quality:** User-defined types, kernels, and their local variables still have full debug info. Only STL/SYCL-header types are suppressed — exactly what is not useful in device debugging. Source-level stepping in user code is fully preserved.

**Impact on memory/time:** Prevents the bulky metadata from being emitted in the first place — this is the cleanest fix. Reduces per-TU `.bc` size significantly; `llvm-link` and `sycl-post-link` both benefit. The effect is especially large because SYCL headers are heavily templated.

**Implementation complexity:** Very low — one conditional flag addition.

---

### Change 2 — sycl-post-link: `--strip-debug-info` flag + driver wiring (safety net for residual metadata)

Even with Change 1, some debug metadata will remain (user-defined types). For very large projects or when the user did not pass `-g`, we want a further safety net.

**What to implement in sycl-post-link:**

In `llvm/tools/sycl-post-link/sycl-post-link.cpp`, add:

```cpp
static cl::opt<bool> StripDeviceDebugInfo(
    "strip-debug-info",
    cl::desc("Strip all debug info from device code before splitting"));
```

After `runPreSplitProcessingPipeline(*M)` and before `getDeviceCodeSplitter()` in `processInputModule()`:

```cpp
if (StripDeviceDebugInfo)
    llvm::StripDebugInfo(*M);
```

`StripDebugInfo()` is declared in `llvm/include/llvm/IR/DebugInfo.h`. With no `!llvm.dbg.cu` in the module, subsequent `CloneModule` calls skip all debug metadata traversal — reducing peak memory from O(N × debug-size) to O(N × code-size).

**Driver wiring in `clang/lib/Driver/ToolChains/Clang.cpp`:**

In `getNonTripleBasedSYCLPostLinkOpts()` (line 11119), pass `--strip-debug-info` when either:
- User explicitly passed `-fno-sycl-device-debug-info` (new flag, see below), OR
- No debug flag from `OPT_g_Group` was specified at all (the silent-accumulation case)

Add the new user-facing flag pair `fsycl_device_debug_info` / `fno_sycl_device_debug_info` to `clang/include/clang/Options/Options.td`.

**Files to modify:**
- `llvm/tools/sycl-post-link/sycl-post-link.cpp` — add `cl::opt` and conditional `StripDebugInfo` call
- `clang/include/clang/Options/Options.td` — add `fno_sycl_device_debug_info` / `fsycl_device_debug_info`
- `clang/lib/Driver/ToolChains/Clang.cpp:11119` — in `getNonTripleBasedSYCLPostLinkOpts()`, propagate the flag

**Impact on memory/time:** Full elimination of debug overhead for the split/clone loop.

---

### Change 3 — Use line-tables-only downgrade as an intermediate quality tier (preserves stepping, low overhead)

For users who want some debug experience without the full type-metadata cost, provide a middle ground:

**Option A (preferred):** Allow `-gline-tables-only` for SYCL device code. Currently listed as unsupported in `getUnsupportedOpts()` in `clang/lib/Driver/ToolChains/SYCL.cpp` (~line 1380). Remove `OPT_gline_tables_only` from that list for SPIR-V targets and forward it to the device compiler. This prevents bulk type metadata from being generated at all — cleanest fix since it acts at source.

**Option B:** Add `StripNonLineTableDebugInfoPass` to `runPreSplitProcessingPipeline()` in `ModuleSplitter.cpp` as an opt-in pass. `stripNonLineTableDebugInfo()` (in `llvm/include/llvm/Transforms/Utils/StripNonLineTableDebugInfo.h`) removes DIType, DILocalVariable, and parameter info while preserving instruction `DebugLoc` source-line mappings.

Both options preserve source-file:line-number stepping in GPU debuggers (e.g., Intel GPU debugger) while eliminating 70–90% of debug metadata size.

**Note:** Verify that `llvm-spirv`'s `transDebugMetadata()` (`llvm-spirv/lib/SPIRV/LLVMToSPIRVDbgTran.cpp`) handles line-table-only LLVM IR correctly with the `OpenCL.DebugInfo.100` format (current default in llvm-spirv).

**Files to modify (Option A):**
- `clang/lib/Driver/ToolChains/SYCL.cpp` — remove `OPT_gline_tables_only` from `getUnsupportedOpts()` for SPIR-V targets; add forwarding

---

## SPIR-V Debug Format Consideration

`llvm-spirv` supports four formats via `--spirv-debug-info-version` (`llvm-spirv/tools/llvm-spirv/llvm-spirv.cpp:240`):
- `legacy` — SPIRV.debug (old)
- `ocl-100` — OpenCL.DebugInfo.100 (current default)
- `nonsemantic-shader-100` — NonSemantic.Shader.DebugInfo.100
- `nonsemantic-shader-200` — NonSemantic.Shader.DebugInfo.200

The **NonSemantic** formats use `SPV_KHR_non_semantic_info` — backends that don't understand the debug extension can safely ignore it without breaking correctness. This is relevant if a backend is overwhelmed by debug info: the runtime/JIT can skip it. However, this doesn't reduce LLVM IR size at the sycl-post-link stage — it only affects what the SPIR-V output contains. So for our RAM problem, format selection alone doesn't help; but it's worth documenting as a backend-facing option.

The format is not currently controlled from the clang driver; it defaults to `ocl-100`. If NonSemantic formats prove more compatible with line-tables-only IR, that connection should be wired in Change 3.

---

## Debug Quality Summary

| Approach | User code types | Line stepping | STL/SYCL header types | Peak RAM |
|----------|----------------|---------------|----------------------|----------|
| Current baseline | ✓ full | ✓ | ✓ (most of the bloat) | O(N×full) |
| Change 1 only (`-fno-system-debug` default) | ✓ full | ✓ | ✗ | O(N×user-only) |
| Change 1 + Change 3 (line tables) | line-only | ✓ | ✗ | O(N×lines) |
| Change 2 (`--strip-debug-info`) | ✗ | ✗ | ✗ | O(N×code) |
| All three (with `-g` → Change 1 only active) | ✓ full | ✓ | ✗ | reduced |
| All three (without `-g` → Change 2 active) | ✗ | ✗ | ✗ | minimal |

The recommended default policy: **with `-g`, apply Change 1 (system debug off by default); without `-g`, apply Change 2 (strip all)**. Change 3 provides an explicit middle tier for users who want faster builds while retaining source stepping.

---

## Critical File Paths

| File | Role |
|------|------|
| `llvm/tools/sycl-post-link/sycl-post-link.cpp` | Add `--strip-debug-info` flag + `StripDebugInfo` call |
| `llvm/lib/SYCLPostLink/ModuleSplitter.cpp:368` | `extractSubModule` — where `CloneModule` is called |
| `llvm/lib/SYCLPostLink/ModuleSplitter.cpp:739–747` | `cleanup()` — existing `StripDeadDebugInfoPass` |
| `clang/lib/CodeGen/CGDebugInfo.cpp:6716` | `noSystemDebugInfo()` — `-fno-system-debug` implementation |
| `clang/lib/Driver/ToolChains/Clang.cpp:11070–11096` | `getSYCLPostLinkOptimizationLevel` — pattern to follow |
| `clang/lib/Driver/ToolChains/Clang.cpp:11119` | `getNonTripleBasedSYCLPostLinkOpts` — add flag propagation |
| `clang/lib/Driver/ToolChains/SYCL.cpp:~1380` | `getUnsupportedOpts` — remove `gline-tables-only` |
| `clang/lib/Driver/ToolChains/SYCL.cpp:1621–1803` | `AddSPIRVImpliedTargetArgs` — add `-fno-system-debug` |
| `clang/include/clang/Options/Options.td:4590` | Existing `fsystem_debug`/`fno_system_debug` definitions |
| `llvm/include/llvm/IR/DebugInfo.h` | `StripDebugInfo()` API |
| `llvm-spirv/tools/llvm-spirv/llvm-spirv.cpp:240` | `--spirv-debug-info-version` option |
| `llvm-spirv/lib/SPIRV/LLVMToSPIRVDbgTran.cpp` | LLVM→SPIR-V debug translation |

## Verification

1. Build a large SYCL project with `-O0 -fsycl -g` before/after; measure peak RSS via `/usr/bin/time -v`
2. With Change 1: verify user-code types appear in device debugger; system types absent
3. With Change 2 (`-fno-sycl-device-debug-info`): `llvm-dis output.bc | grep 'llvm.dbg.cu'` should return nothing
4. With Change 3 (`-gline-tables-only`): verify source-line stepping works in Intel GPU debugger; no type inspection
5. Run existing LIT tests: `llvm/test/tools/sycl-post-link/`, `clang/test/Driver/sycl*`
6. For Change 3: run `llvm-spirv/test/` and `llvm/test/tools/llvm-spirv/` to verify SPIR-V line-table compatibility

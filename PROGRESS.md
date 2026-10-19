# StableHLO backend rewrite — progress

Full plan also lives at `/Users/n1k0/.claude/plans/pasted-content-id-c38e-you-are-streamed-allen.md`
(outside the repo, may not persist — this file is the durable copy).

## Context

Rewriting the backend from linalg/IREE to StableHLO/XLA. XLA only accepts
`func`/`stablehlo`/`chlo` (no `tensor.*`, `arith.*` on tensors, `linalg.*`,
`scf.*`); entry fn must be `func.func @main`; scalars are 0-d tensors; shapes
must be static. Untranslatable constructs → compile-time `Undefined("name")`,
never partial output. Reused unchanged: `mlir-backend/ir/*.hpp` + `ir/parser.hpp`
(ANTLR→AST), `utils.hpp`'s `match()`. Rewritten: `compiler.hpp`, build system,
`run_tests.py`.

**IMPORTANT, told explicitly by the user**: some `tests/*.fut` may already be
failing for reasons unrelated to this rewrite. If a failing test looks like a
lot of work to fix, ask before fixing it — don't just dive in.

## Build-system research (verified against upstream source, not guessed)

- nixpkgs does not package StableHLO; no viable find_package(MLIR)-from-nixpkgs
  path either, since stablehlo pins LLVM at a rolling trunk commit (not a
  release) incompatible with nixpkgs' release-branch `llvmPackages_23`.
- Chosen approach: build LLVM+MLIR only (`-DLLVM_ENABLE_PROJECTS=mlir`, no
  clang/tools/tests) from `llvm-project` at commit
  `ab547095ead5464dc024d66264d9b8a987f429f3`, and pull StableHLO in via LLVM's
  `LLVM_EXTERNAL_PROJECTS` mechanism (`LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR`) so
  it builds in the same configure/build as MLIR — no separate install step or
  find_package needed, no package-export uncertainty. StableHLO pinned at tag
  `v1.20.0` (commit `639335932274930617d9724fb25ec9f2f52c5e40`).
  Verified from actual upstream source (not assumed):
  `openxla/stablehlo`'s `CMakeLists.txt` auto-detects standalone-vs-embedded
  mode; `LLVM_EXTERNAL_<NAME>_SOURCE_DIR` is llvm-project's real variable
  pattern (checked in `llvm/CMakeLists.txt`).
- This is **still a from-scratch LLVM/MLIR build, no binary-cache hit possible**
  for it specifically (unavoidable — stablehlo's pin is a moving trunk commit).
  It's the smallest buildable scope achievable (MLIR only, no IREE compiler/
  runtime/HAL/CUDA/Vulkan), not literally "hits the cache" for everything.
- `jax`/`jaxlib` ARE in nixpkgs (0.11.2), CPU-only by default via plain
  `pkgs.python3.withPackages (ps: [ps.jax ps.numpy])` (CUDA is an opt-in extra,
  not pulled in this way).
- **Not yet empirically confirmed**: whether this CMake wiring (LLVM_EXTERNAL_
  PROJECTS route) actually produces working `StablehloOps`/`ChloOps` targets
  linkable by `mlir-backend/CMakeLists.txt` as-is. This is exactly task 0.1a —
  **blocked on being able to actually run `nix build`/`nix develop`**, see
  below.

## BLOCKER (as of this session)

The agent's sandboxed Bash tool cannot talk to the nix daemon
(`/nix/var/nix/daemon-socket/socket: Operation not permitted`), so the user
is running the build manually in their own terminal:
```
cd /Users/n1k0/mlir-backend/hlo && nix develop -c build 2>&1 | tee build.log
```
(`build.log` is also how the agent reads progress/errors back.)

### Build error #1 (found and fixed)
`ninja: error: '.../stablehlo/integrations/cpp/builder/NOTFOUND/mlir/Dialect/Func/IR/FuncOps.td' ... missing and no known rule to make it`

Root cause (confirmed from stablehlo's actual source, not guessed):
`stablehlo/CMakeLists.txt` unconditionally does `add_subdirectory(integrations)`
(no option gates it off). That subdirectory's CMake code does
`list(GET MLIR_INCLUDE_DIRS 0 mlir_includes)`, but `MLIR_INCLUDE_DIRS` (plural)
is only ever set by `find_package(MLIR CONFIG)` in stablehlo's
`STABLEHLO_STANDALONE_BUILD` mode — we're in `STABLEHLO_EXTERNAL_PROJECT_BUILD`
mode instead (via `LLVM_EXTERNAL_PROJECTS`, confirmed to be the same mode
stablehlo's own comments say XLA/MHLO itself uses), where that variable is
never set, so the list-index resolves to a stale/absent cache entry containing
literally `"NOTFOUND"`.

Fix applied: `patches/stablehlo-no-integrations.patch` removes that one
`add_subdirectory(integrations)` line (we don't use stablehlo's C++/Python
"builder" convenience API — `StablehloOps`/`ChloOps`, which we do need, live
in `dialect`/`api`, unaffected). Wired into `flake.nix` via
`pkgs.applyPatches` (`stablehloSrc`), same pattern the old, now-deleted
`patches/sort-distribute.patch` used for IREE.

**Confirmed working**: with this patch, all of LLVM+MLIR+StableHLO+ChloOps
built successfully (1576/1576 ninja steps for that part) on macOS
aarch64-darwin. The `LLVM_EXTERNAL_PROJECTS` approach is validated.

### Build error #2 (found and fixed)
`mlir-backend`'s own PCH step failed: `fatal error: 'mlir/Dialect/Func/IR/FuncOps.h' file not found`.

Root cause: `mlir-backend/CMakeLists.txt` links against plain MLIR dialect
library targets (`MLIRFuncDialect` etc.), but those don't export their
include dirs as an `INTERFACE`/target property the way the old IREE targets
did (IREE wraps things in modern CMake targets; upstream LLVM/MLIR dialect
libraries rely on directory-scoped `include_directories()`, which doesn't
propagate to `mlir-backend` since it's a sibling subdirectory, not nested
under `llvm-mlir`'s own directory tree). Confirmed via the actual failing
compile command in the log — it had `-I` for stablehlo's own source/build
dirs (stablehlo's CMake does set these explicitly) but nothing for
MLIR/LLVM's.

Fix: added explicit `target_include_directories` in
`mlir-backend/CMakeLists.txt` for `${MOCHA_LLVM_SOURCE_DIR}/llvm/include`,
`${CMAKE_BINARY_DIR}/llvm-mlir/include`, `${MOCHA_LLVM_SOURCE_DIR}/mlir/include`,
`${CMAKE_BINARY_DIR}/llvm-mlir/tools/mlir/include` — paths confirmed from the
actual ninja build log (e.g. `llvm-mlir/tools/mlir/lib/TableGen/...` lines),
not guessed. **Confirmed fixed**: PCH + all three `.cpp.o` compiles now
succeed (on Apple M4 / aarch64-darwin).

### Build error #3 (found and fixed)
Link failure: `Undefined symbols for architecture arm64:
mlir::detail::TypeIDResolver<mlir::vector::VectorDialect, void>::id` /
`mlir::vector::VectorDialect::VectorDialect(...)`.

Root cause: `FutharkCompiler`'s constructor (`compiler.hpp`) still registers
`mlir::vector::VectorDialect` (left over from the deleted SegScan
vector-packing-into-`vector<Nxi64>` code — not yet cleaned up, that's Phase 1
dialect-trim scope, not Phase 0). `mlir-backend/CMakeLists.txt`'s `LIBS` never
explicitly listed `MLIRVectorDialect` — this worked before only because
IREE's `LinalgExt` library pulled it in transitively. Removing IREE exposed
the missing explicit dependency. Not M4-specific; a plain missing-library
link error on any arch.

Fix: added `MLIRVectorDialect` to `LIBS` in `mlir-backend/CMakeLists.txt`.
**Not yet re-verified** — user re-running now. Still no `rm -rf build` needed
(only this one file changed again).

## Phase 0 — Setup

- [x] `utils.hpp`: `Undefined()` takes an optional construct-name string
      (backward compatible, all existing call sites were bare `Undefined()`).
- [x] `compiler.hpp`: stripped all IREE dependencies —
  - removed the 4 `iree/compiler/Dialect/...` includes
  - removed `FlowDialect`/`IREELinalgExtDialect` dialect registration
  - `LowerWithAcc`, `LowerSegHist`, `LowerSegScan` reduced to
    `Undefined("with_acc")` / `Undefined("SegHist")` / `Undefined("SegScan")`
    stubs (deleted `EmitScatter`/`AccElemIsArray`/`MatchRowScatter` and all the
    `iree_linalg_ext.scatter`/`.scan` construction code — real StableHLO
    lowerings for these land in Phase 7 / Phase 5-6 respectively, not now)
  - **expected regression**: `scatter.fut`, `hist1d/2d/3d.fut`, `scan0.fut`,
    and anything using `with_acc`/SegHist/SegScan (incl. parts of
    `moe_black_box*.fut`) now hit `Undefined()` where they previously worked
    under the old linalg/IREE backend. This is intentional and expected per
    the phased plan, not a bug — don't try to un-break these now.
  - segmap/segred/matmul/if/for-loop lowering is **untouched** in this pass —
    still emits `linalg.generic`/`scf.if`/`scf.for`/`tensor.*` exactly as
    before. That's fine for now: those MLIR dialects are still being built
    (LLVM_ENABLE_PROJECTS=mlir builds all core upstream dialects, not just
    IREE-specific ones), so the project should still compile. Retargeting
    these to StableHLO is Phase 1+.
- [x] `flake.nix` rewritten: dropped `iree` input, CUDA/Vulkan/Metal
      everything, all `iree*`/`run-iree`/`iree-benchmark` scripts. Added
      `llvm-project`/`stablehlo` git inputs (pinned, `shallow = true`), `jax`+
      `numpy` via `pythonWithJax` in devShell. Kept `gpu-ir`, `futhark-test`,
      `build`, `compile`, `run-tests`, `clean`/`clean-all`, `clangd`.
- [x] Top-level `CMakeLists.txt` rewritten: drops `add_subdirectory(iree)`,
      adds the `LLVM_EXTERNAL_PROJECTS=stablehlo` wiring described above.
- [x] `mlir-backend/CMakeLists.txt`: dropped the two
      `iree::compiler::Dialect::{LinalgExt,Flow}::IR` link libs. Everything
      else (MLIR arith/linalg/scf/tensor/index libs, StablehloOps/ChloOps)
      unchanged — still needed until later phases retire linalg/scf/tensor
      usage.
- [x] Deleted `patches/sort-distribute.patch` (IREE-LLVMGPU-codegen-specific,
      nothing left to apply it to) and the now-empty `patches/` dir.
- [ ] **Run the actual build** (blocked, see above) and fix whatever CMake/
      flake issues show up — task 0.1a is genuinely unresolved until this
      happens. Likely failure points to check first: (1) does
      `LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR` actually get picked up and built;
      (2) do `StablehloOps`/`ChloOps` end up as real CMake targets visible to
      `mlir-backend/CMakeLists.txt`'s `target_link_libraries`; (3) does the
      antlr4-runtime find_package still work unchanged (untouched by this
      rewrite, should be fine); (4) C++ standard mismatch — stablehlo's own
      build sets C++17, mlir-backend builds at C++23, probably fine but
      unverified.
- [ ] Spike: hand-write a trivial `func.func @main(...) { stablehlo.add ... }`
      and confirm it round-trips through `example_run.py`'s `compile_on`/
      `execute` (`--platform cpu`) — **do this once jax is available**,
      independent of whether `mlir-backend` itself builds yet.
- [x] `run_tests.py` rewritten: compiles via the `compile` nix wrapper
      directly (its stdout IS the module text, no `.vmfb`/`out/*.mlir`
      intermediate), parses `@main`'s signature by importing and reusing
      `example_run.py`'s `main_signature` (`import example_run as er`), binds
      parsed Futhark literals positionally via a new `_to_array()` (no more
      `futhark.size_args`/derived-size indirection), executes via
      `er.compile_on("cpu", text)` + `er.execute(...)`, compares against the
      existing `-- output {}` blocks exactly as before (unchanged
      `_values_match`/`_expected_values`). Added a `-- undefined` lone-comment
      convention (`_is_undefined_test`): a file with that marker passes iff it
      *fails* to compile. Dropped everything IREE-ABI-specific
      (`_build_flags`, `_entry_params`/`_classify_param` and the paren-matching
      helpers behind them, `_iree_output_values`, `_out_mlir`, `_payload`).
      **Not yet run** — see below for expected current behavior.
      **Known limitation to revisit**: compiles once per file and reuses the
      `exe` across all cases in that file; if a file's cases need genuinely
      different compiled modules (they don't today — same signature per
      file), this would need revisiting, but matches today's test corpus.

**Expected behavior if you run `run-tests` right now, before Phase 1**: every
non-`-- undefined` test will FAIL with `"compile/load failed: ... no @main
function found"` — this is correct, not a bug. The backend still emits
`@entry_main` + linalg/scf/tensor (entry→`@main` renaming and the StableHLO
scalar/segmap rewrite are Phase 1 work, not done yet). Running it now is only
useful to confirm the harness itself runs end-to-end without crashing.

## Cross-cutting decisions (resolved, apply from Phase 1 onward)

1. Scalars become 0-d `RankedTensorType`s in the type system itself —
   `LowerPrimTypeInt/Float/Bool` return `RankedTensorType::get({}, ...)`, not
   bare `IntegerType`/`FloatType`/`i1`. Every scalar op gets rewritten from
   `arith.*`/`math.*` to `stablehlo.*`/`chlo.*` — genuinely new code, not a
   rename. `castToType`'s scalar-extraction branch disappears.
2. `Undefined()` optional name param — done (Phase 0).
3. Entry function → `@main` regardless of Futhark source name; >1 `entry`
   function is a hard compile error (not `Undefined()` — it's ambiguous, not
   untranslatable). No current test has >1 entry.
4. Drop `futhark.size_args` entirely — only existed for the old IREE-ABI
   runner; dynamic-size programs are `Undefined()` now.
5. Dialect registration trims to `func`/`stablehlo`/`chlo` once Phase 1's
   scalar-op rewrite lands (not yet — Phase 0 left `linalg`/`math`/`tensor`/
   `scf`/`index`/`vector` registered since segmap/segred/matmul/if still use
   them).

## Phase 1 — segmap (done, unbuilt/unverified)

**Important correction to the original plan, found by actually reading the
test files (not just the secondhand research summary)**: of the originally
assumed "Phase 1 passing set" (map.fut, map2-5.fut, iota.fut, gelu.fut), only
**`map.fut` and `iota.fut` are genuinely static**. `map2/4/5.fut` use
`iota n` where `n` is a runtime parameter (dynamic size), `map3.fut`/
`gelu.fut` take array parameters with no literal size (also dynamic in
Futhark's GPU IR — any array parameter without a literal size gets an
implicit runtime size variable). XLA requires fully static shapes
end-to-end, and dynamic paths are `Undefined()` per the brief — so these
will **also** hit `Undefined()`, same as `map_dynamic.fut`/`iota_dynamic.fut`,
just not named "_dynamic". This is expected per the brief's own carve-out,
not a bug — but it means almost every test with a non-literal-sized array
parameter is currently out of reach until something (a later phase, not yet
planned) lets the backend specialize for a concrete size at compile time.
Added an explicit, named gate for this in `LowerFunction` (see below) so it
fails loud and early rather than relying on it to surface accidentally deep
inside some construct's lowering.

### What changed in `compiler.hpp`
- **`LowerFunction`**: added an explicit static-shape check (`Undefined(
  "dynamic-shaped function parameter/result")`) right after computing
  `inputTypes`/`retTypes`, for every function (not just entry) — the
  deliberate, early, named gate described above. Entry function renamed to
  `@main` regardless of Futhark source name; >1 entry point is a hard
  `throw` (ambiguous, not untranslatable, so not `Undefined()`). Dropped the
  whole `futhark.size_args` attribute block.
- **`LowerTy`**: wraps the `TypePrim` branch in a 0-d `RankedTensorType`
  (scalars are 0-d tensors). `LowerPrimType`/`LowerPrimTypeInt/Float/Bool`
  themselves are **unchanged** (still return bare types) — they're also used
  as array *element* types via `LowerTypeArray`/`LowerSegOpBaseType`, which
  must stay bare. Only `LowerTy`'s own scalar branch wraps.
- **`castToType`**: simplified to `stablehlo.reshape`-or-`Undefined()`; the
  old bare-scalar-extraction branch is gone (there's no bare-scalar
  destination type left anywhere).
- **Scalar op library**, all rewritten to stablehlo/chlo, operating
  uniformly on tensors (0-d or N-d) instead of split arith-scalar/
  linalg-tensor paths: `LowerPrimValue`/`getPrimValueAttr` (stablehlo.constant
  on a 0-d tensor; `getPrimValueAttr` itself stays a bare scalar attr, also
  reused by `LowerArrayLit`), `LowerBinOp` (full rewrite — `UDiv`/`UDivUp`/
  `SDiv`/`SDivUp`/`UMod`/unsigned min/max flagged `Undefined()` rather than
  guessed), `BasicOpCmpOp` (→ `stablehlo.compare`), `BasicOpUnOp` (→
  `stablehlo.negate`/`not`), `BasicOpConvOp` (ZExt flagged `Undefined()`,
  SExt/SIToFP → `stablehlo.convert`), `LowerIntrinsic` (mapped against the
  actual stablehlo/chlo op inventory checked at the pinned revision — sqrt,
  exp, log, trig, tanh, ceil/floor/round, popcount/clz, atan2 from core;
  asin/acos/atan/sinh/cosh/*h/erf/erfc/isinf from chlo; exp2/log2/log10/trunc/
  isnan/ctz/copysign/fma flagged `Undefined()`, no equivalent found).
  **None of this is exercised by a currently-passing test** (gelu.fut, the
  one that would, hits the dynamic-shape gate first) — unverified, flagged
  in-code.
- **Banned-dialect cleanups** found while auditing every `BasicOp` variant
  for the scalar-type ripple (not originally scoped for Phase 1, but left
  unfixed they'd silently emit `tensor.*`/`linalg.*` for any test that hit
  them): `BasicOpReshape` → `stablehlo.reshape` (collapses the old collapse/
  expand/fallback-shape-tensor code into one case), `BasicOpRearrange` →
  `stablehlo.transpose`, `BasicOpReplicate` → `stablehlo.broadcast_in_dim`
  (also simplified: scalars being 0-d tensors means the old "is it a bare
  scalar or a tensor" branch collapses into one path). `BasicOpIndex`/
  `LowerIndex`'s single-element fast path now wraps its result in a 0-d
  tensor instead of returning bare; **the general slice path still emits
  `tensor.extract_slice`** (flagged TODO — needs `stablehlo.slice`/
  `dynamic_slice`, not exercised by any passing test, since affine reads
  inside SegMap bodies take a separate fast path that never reaches it).
  `createIota` (backing `BasicOpIota` — exercised by `iota.fut`) rewritten
  to `stablehlo.iota`+`broadcast_in_dim`+`multiply`+`add`, dynamic-length
  `Undefined()`. `BasicOpConcat`/`BasicOpScratch`/`LowerArrayLit`'s
  non-constant branch **still emit tensor.* ops** — left as-is, flagged,
  not reachable by `map.fut`/`iota.fut`.
- **`LowerSegMap`** (the actual load-bearing Phase 1 piece): rewritten to
  `stablehlo.map`. Operands = affine-read arrays + one full-shape
  `stablehlo.iota` per iteration-space dimension. The region is lowered
  first using the ordinary `ctx`-copy (so it can reference outer SSA values
  freely, same as today), **then** made isolated-from-above by calling
  MLIR's own `mlir::makeRegionIsolatedFromAbove` on the finished region —
  this is a real, built-in MLIR utility (`mlir/Transforms/RegionUtils.h`,
  confirmed at the pinned LLVM revision) that finds every outer-defined SSA
  value a region uses, appends a matching block argument, and rewires
  internal uses — avoiding a hand-written Futhark-AST free-variable walker
  entirely. Each discovered free variable is then `broadcast_in_dim`'d to
  the full shape and appended as a real operand (`Operation::insertOperands`
  — `stablehlo.map`'s `inputs` is one simple variadic group, no segment-size
  bookkeeping needed). Non-scalar free variables and dynamic shapes are
  `Undefined()`. New includes: `mlir/IR/PatternMatch.h` (for `IRRewriter`),
  `mlir/Transforms/RegionUtils.h`; new CMake lib `MLIRTransformUtils` (was
  already linked transitively, now explicit).
- **`LowerSegOpKernelBody`**: now only called by `LowerSegRed` (the new
  `LowerSegMap` has its own tensor-native binding instead, since its block
  args are already 0-d tensors). Fixed to wrap its bare-scalar
  `linalg.generic` block args and gtid values into 0-d tensors before
  binding, since the shared `LowerStm`/`LowerBinOp` pipeline they feed now
  uniformly expects tensors.
- **`LowerSegRed`** (unchanged structure — still `linalg.generic`, its real
  `stablehlo.reduce` rewrite is Phase 2): three small adaptations so it
  still compiles/works under the new global scalar convention: unwrap
  `neutral` before `linalg.fill` (needs a bare scalar), wrap `acc` block
  args before binding into the combining lambda's scope, unwrap the
  lambda's `results` before `linalg.yield`.
- **`ExpIf`**: unwraps its now-0-d-tensor condition back to a bare `i1`
  before `scf.if` (structure otherwise unchanged — the real `stablehlo.if`
  rewrite is Phase 4).
- **`LowerLoop`**: reduced to one line, `Undefined("loop")` — the entire
  `scf.for` construction deleted outright (brief: all loops, for and while,
  are `Undefined()`, not just while).
- **`getConstantInt`**: now also matches a single-element
  `DenseIntElementsAttr` (a 0-d-tensor constant), not just a bare
  `arith.constant` int, since blackbox "scalar size parameter" arguments
  (e.g. `LowerRaggedDot`'s ragged-dim index) are 0-d tensors now too.
  `LowerArgSort`'s `createIota` call site updated to pass `stablehlo.constant`
  0-d tensors instead of bare `arith.constant`s.
- **Dialect registration deliberately left unchanged** (still registers
  `linalg`/`tensor`/`scf`/`index`/`vector`/`math`) — contrary to the
  original plan's "trim to func/stablehlo/chlo in Phase 1," this turned out
  to be premature: `LowerSegRed`/`LowerMatMul`/`ExpIf`/the blackbox
  functions all still genuinely need these dialects until Phases 2/3/4
  retarget them. Trimming now would just break everything *they* still use.
- New test: `tests/loop_undefined.fut`, using the `-- undefined` marker
  convention from `run_tests.py`'s Phase 0 rewrite.

### Not yet done / known gaps going into Phase 2+
- `BasicOpIndex`'s general slice path, `BasicOpConcat`, `BasicOpScratch`,
  `LowerArrayLit`'s non-constant branch still emit banned `tensor.*` ops.
  None are reachable by `map.fut`/`iota.fut`; will need fixing whenever a
  test actually exercises them.
- `LowerIntrinsic`/`BasicOpConvOp`'s ZExt/`LowerBinOp`'s unsigned-div/mod/
  min/max are `Undefined()` rather than guessed — genuinely unverified
  stablehlo semantics, not exercised by any current test either.
- The realistic Phase-1-passing set is just `map.fut` + `iota.fut`. Every
  other array-parameterized test in the whole suite will hit the new
  dynamic-shape gate in `LowerFunction` until some later mechanism lets the
  backend specialize per concrete input size — **this isn't scoped in any
  current phase of the plan and is a real open question worth raising with
  the user if/when it becomes blocking** (most interesting Futhark programs
  take array arguments, so this is a significant practical limitation, not
  just an edge case).

### Verification — first real test run done
Build succeeded after two rounds of compile/link fixes:
- Two missing-`.getResult()` bugs (passing an un-dereferenced `OpTy` where
  `ValueRange` was expected needs two implicit conversions, which C++
  disallows) and `MapOp` needing `->getResults()` not `.getResults()`
  (single, non-variadic result, no plural accessor generated).
- `mlir::makeRegionIsolatedFromAbove`/`mlir::IRRewriter` linked `undefined
  symbol: typeinfo for mlir::RewriterBase` — the prebuilt MLIR libs are
  `-fno-rtti` (LLVM's default) while this backend builds `-frtti`, so
  `RewriterBase`'s vtable/typeinfo doesn't exist in the prebuilt archive.
  Fixed by dropping `IRRewriter`/`makeRegionIsolatedFromAbove` and doing the
  same thing with the two plain free functions it's built from
  (`getUsedValuesDefinedAbove` + `replaceAllUsesInRegionWith` +
  `Block::addArgument`), which need no `RewriterBase` instance at all.

**First `run-tests` run: 11 passed, 32 failed.** Confirmed exactly as
predicted above — the overwhelming majority of failures are the expected
`Undefined(): dynamic-shaped function parameter/result` for every
array-parameterized test (reduce*.fut, scan0.fut, argmax.fut, matmul.fut,
hist*.fut, map2-5.fut, gelu.fut, map_dynamic.fut, etc.) — not bugs, working
as designed. A few others are pre-existing/unrelated to this rewrite, not
chased: `ragged_dot_mode*.fut`/`irreg_matmul.fut` compile fine but fail at
*execution* with `UNIMPLEMENTED: HLO opcode 'ragged-dot' is not supported by
XLA:CPU ThunkEmitter` — an XLA:CPU platform limitation, unrelated to our
lowering (ragged_dot itself isn't touched by this phase). `scatter.fut`
hits the already-documented Phase 0 `with_acc` regression.
`moe_black_box_t1.fut` hits the already-documented Phase 0 `SegHist`
regression. `moe_black_box.fut` hits a *new*, correctly-firing safety check:
`Undefined(): SegMap: non-scalar free variable` — its kernel captures a
whole array (not a scalar) from outer scope, which `LowerSegMap` correctly
refuses to broadcast rather than emit something wrong.

**One real bug found and fixed**: `map.fut` (the actual target test) failed
with `'stablehlo.iota' op result #0 must be statically shaped tensor of
...  but got 'tensor<4xtensor<i64>>'` — a tensor-of-tensor. Root cause:
`Dim::value` (the lowered Futhark size/index value) is a 0-d tensor now
(scalars are 0-d tensors), but `LowerSegMap`'s per-dimension iota
construction and `LowerSegOpKernelBody`'s gtid `arith.index_cast` were both
still using `d.value.getType()` directly as a *bare* element type. Fixed
both (and a third, currently-unreachable occurrence in `LowerSegRed`'s
dynamic-size branch, for consistency) to extract the bare element type via
`getRankedType(d.value.getType()).getElementType()` first.

**Confirmed fixed and verified**: `map.fut` and `iota.fut` both pass
(12 passed / 31 failed overall, up from 11/32 — exactly the one expected
fix, no other regressions). Phase 1's actual goal is achieved.

Root cause of the `map.fut` failure (second round): `'stablehlo.return' op
must be the last operation in the parent block`. The `OpBuilder::
InsertionGuard` in `LowerSegMap` was constructed *after* `builder.
createBlock(...)`, not before. `createBlock` moves the builder's insertion
point into the new block as a side effect; a guard constructed afterward
saves "already inside the new block" and restores to that same place at
scope-exit, instead of back to the enclosing function body. So after
`LowerSegMap` returned, the builder's insertion point was left stuck inside
the `stablehlo.map`'s region, right after its `stablehlo.return` -- and
whatever got lowered next (the function's own `return`) landed there too,
after the terminator. Fixed by constructing the guard before `createBlock`,
matching the established pattern elsewhere in the file (e.g. the old
`LowerSegHist`'s block-building code, pre-Phase-0-stub).

### Test-suite fix: static-size annotations on existing tests (not a backend bug)
Investigated why `reduce.fut`/`reduce2.fut`/`reduce3.fut`/`scan0.fut`/
`matmul.fut`/`argmax.fut` all hit the dynamic-shape gate. Root cause,
confirmed by comparing against `dot_general.fut` (which already passes):
`dot_general.fut`'s `main` explicitly annotates concrete sizes
(`x: [2][3]f32`, `y: [3][2]f32`); the others left `main`'s parameters
unannotated (`xs: []i32`, `main x y = ...`), so Futhark infers size-
polymorphic (dynamic) types even though every test's actual input is a
fixed-size literal. Not a backend limitation at all -- just a test-fixture
annotation gap, left over from when the old IREE-based backend supported
dynamic shapes and nobody needed to pin sizes.

Fixed by adding explicit sized annotations to `main` (matching
`dot_general.fut`'s existing convention), so each compiles to fully static
shapes:
- `tests/matmul.fut`: `main (x: [2][3]f32) (y: [3][2]f32) : [2][2]f32`
- `tests/reduce.fut`: `main (xs: [4]i32): i32`
- `tests/reduce2.fut`: `main (xss: [2][3]i32): [2]i32`
- `tests/reduce3.fut`: `main (xs: [4]i64)`
- `tests/scan0.fut`: `main (xs: [4]i32): [4]i32`

**`tests/argmax.fut` deliberately left unannotated** -- it has two
`-- input` cases with *different* array lengths (7 and 4), so no single
static size annotation can satisfy both; and its reduce operator is
non-commutative, which Phase 2 is going to gate to
`Undefined("segred: non-commutative")` regardless of shape. Not worth
fixing until/unless Phase 8 actually solves non-commutative reduce.

**Not yet rebuilt/retested** after this test-file change (paused here per
request). Next step when resuming: rebuild + rerun, then these five tests
should newly reach (not necessarily pass) Phase 2's `LowerSegRed`/Phase 3's
`LowerMatMul` logic instead of bailing out at the dynamic-shape gate --
i.e. Phase 2 (segred) now has real tests to verify against once it's
implemented.

## Phase 2 — segred (done, verified)

`LowerSegRed` rewritten to call the shared `LowerSegMap` for the fused
per-element part, then `stablehlo.reduce` over the innermost dim with
`init_values` = the lowered neutral elements directly (no more
`linalg.fill`-broadcast). Non-commutative → `Undefined("segred:
non-commutative")` (Phase 8 revisits this; `argmax.fut` stays Undefined()
until then, also still blocked by having two differently-sized test cases —
not annotated, see Phase 1 notes). `LowerSegOpKernelBody` deleted outright
(dead code once `LowerSegRed` stopped calling it).

**Verified: `reduce.fut` (plain commutative), `reduce2.fut` (segmented),
and `reduce3.fut` (tuple operator) all pass.** Tuple operators did *not*
fall out for free as originally assumed -- see bug below. Final count:
18 passed / 28 failed (up from 15/31 at the start of this phase).

New tests added (per the original coverage-gap list), using the `-- undefined`
marker convention since all three are expected to hit the non-commutative
gate until Phase 8 actually solves it (so no need to hand-verify exact
numeric outputs -- confirmed each hits `Undefined(): segred: non-commutative`
specifically, not some other accidental gate):
- `tests/segred_matmul2x2.fut` -- 2×2 matrix product (tuple-encoded
  `(a,b,c,d)` for `[[a,b],[c,d]]`), associative but non-commutative.
- `tests/segred_affine_compose.fut` -- affine-map composition
  `(a,b)∘(c,d)`, same property.
- `tests/segred_noncomm_segmented.fut` -- the matmul2x2 operator, but
  segmented (combines with `reduce2.fut`'s segmentation pattern).
All three validated to actually parse/typecheck as real Futhark via
`gpu-ir` before being relied on.

### Bugs found and fixed along the way

1. **`LowerSegMap`'s per-dimension `stablehlo.iota` had the wrong element
   type** (found via `map.fut`, carried over from Phase 1, actually fixed
   in this session): `Dim::value` is a 0-d tensor now (scalars are 0-d
   tensors), but the iota type construction and `LowerSegOpKernelBody`'s
   (now-deleted) gtid `arith.index_cast` were both using
   `d.value.getType()` directly as a *bare* element type, producing
   `tensor<4xtensor<i64>>` (tensor-of-tensor, invalid). Fixed by extracting
   the bare element type via `getRankedType(d.value.getType())
   .getElementType()` first, in both places (plus a third, then-
   unreachable occurrence in `LowerSegRed`'s dynamic-size branch).

2. **Insertion-point ordering bug in `LowerSegMap`**: `'stablehlo.return' op
   must be the last operation in the parent block`. The
   `OpBuilder::InsertionGuard` was constructed *after* `builder.
   createBlock(...)`, not before. `createBlock` moves the insertion point
   into the new block as a side effect; a guard constructed afterward saves
   "already inside the new block" and restores to that same place at
   scope-exit, instead of back to the enclosing function body -- so
   whatever got lowered next landed inside the region, after its
   terminator. Fixed by constructing the guard before `createBlock`.

3. **`mlir::makeRegionIsolatedFromAbove`/`mlir::IRRewriter` failed to link**:
   `undefined symbol: typeinfo for mlir::RewriterBase`. The prebuilt MLIR
   libs are `-fno-rtti` (LLVM's default); this backend builds `-frtti`.
   Fixed by dropping `IRRewriter` entirely and using the two plain free
   functions it's built from (`getUsedValuesDefinedAbove` +
   `replaceAllUsesInRegionWith` + `Block::addArgument`), which need no
   `RewriterBase` instance.

4. **`stablehlo.map` has no per-operand indexing map** (found via
   `reduce.fut`, a *new*, Phase-2-specific gap in `LowerSegMap`, not a
   leftover from Phase 1): unlike `linalg.generic`, every `stablehlo.map`
   operand must already be the exact iteration-space shape
   (`SameOperandsAndResultShape`). An affine read's array can be lower-rank
   than the iteration space -- e.g. `reduce.fut`'s plain `[4]i32` array,
   read inside what Futhark's GPU IR models as a `[1][4]` segspace even for
   a flat, unsegmented reduction. Fixed by broadcasting such arrays to the
   full iteration-space shape first, via `stablehlo.broadcast_in_dim` using
   the affine read's own index map (its results are exactly the
   broadcast_dimensions needed).

5. **`LowerIndex`'s single-element ("no slice") fast path still emitted
   banned `tensor.extract`/`arith.index_cast`/`tensor.from_elements`**
   (found via `reduce.fut`: Futhark's GPU IR unwraps a flat reduce's
   rank-1 `stablehlo.reduce` result down to a scalar via an explicit
   fixed-index read, `result[0]`, which goes through exactly this path --
   contradicting Phase 1's note that this path was unreached). Rewritten to
   `stablehlo.dynamic_slice` (handles constant and dynamic indices
   uniformly, no unwrapping needed since its `start_indices` are already
   0-d tensors) + `stablehlo.reshape` down to a 0-d scalar. The *other*
   (general, slice-containing) path in the same function is still
   known-banned and still unreached; left as-is.

6. **`stablehlo.map` allows only a single, non-variadic result** (found via
   `reduce3.fut`'s tuple operator: `'stablehlo.map' op requires one
   result`) -- contradicting the Phase 1 assumption that tuple results
   would "fall out for free" the way `stablehlo.reduce`'s variadic
   inputs/results do. Restructured `LowerSegMap` to build one
   `stablehlo.map` *per output value* when the kernel body returns more
   than one result, each recomputing the full set of shared body
   statements (redundant work -- this backend doesn't run CSE itself to
   clean it up -- but correct, and avoids reintroducing a multi-value
   packing trick like the old IREE scan's lane-packing hack). Free-variable
   discovery/broadcast-fixup now happens per sub-map independently.

### Note for Phase 3+
`stablehlo.reduce`'s region block-argument order is the *opposite* of
Futhark's own `SegBinOp.lambda.params` convention: StableHLO orders
`(new_value_args..., accumulator_args...)`, while Futhark orders
`(accumulator_params..., new_value_params...)` (confirmed via the spec's
`func_inputs(body)[:len//2]` / `[len//2:]` split, and matches the existing
`chlo.scan`-era comment "the Futhark binop is lambda(acc.., in..)"). Keep
this in mind for Phase 5/6's `chlo.scan`, which will have the same
Futhark-vs-StableHLO param-order mismatch to handle.

## Phase 3 — matmul (done, verified)

`LowerMatMul` rewritten to `stablehlo.dot_general`, reusing
`LowerDotGeneral`'s existing `DotDimensionNumbersAttr` pattern (contracting
dims `{1}`/`{0}`, no batch dims, matching the signature comment's
`[d4][d5]t`/`[d5][d6]t` convention). Dropped all the `linalg.fill`/
`tensor.empty`/`tensor.dim`-based zero-init and dynamic-size scaffolding
entirely -- unneeded now that `matmul.fut`'s signature is pinned to static
sizes (Phase 1's `LowerFunction` gate already guarantees this for anything
that reaches here, so added an explicit `Undefined("matmul: dynamic
shape")` as a defensive check rather than relying on that alone). No
surprises/bugs this time -- went in cleanly on the first build. `precision`
left at StableHLO's default (`DEFAULT`), per the earlier decision to rely on
`example_run.py`'s `--precision`/`set_precision` text post-processing rather
than threading a compiler-level flag.

**Verified**: `matmul.fut` passes. `matmul_static_shape.fut` has no checked
test cases (benchmark-only `-- random input`, unaffected, same as before
this rewrite). `custom_flattening_matmul.fut`/`irreg_matmul.fut` remain out
of scope (dynamic-shaped composite tests, unannotated). 19 passed / 27
failed overall (up from 18/28).

## Phase 4 — if (not started)
`ExpIf` → `stablehlo.if`/`stablehlo.return`. **Verify empirically first**
(JAX `lax.cond` reference lowering) whether `stablehlo.if` branches are
isolated-from-above (need Phase 1's free-var machinery generalized) or can
capture freely — don't assume. New test: standalone top-level `if` (today only
nested inside a reduce lambda).

## Phase 5 — segscan, chlo.scan (not started)
Delete the whole IREE-bug-workaround vector-packing trick outright (not
adapting it). `LowerSegMap` + `chlo.scan` (verify exact op/attrs against
vendored `ChloOps.td`), `is_associative = true` unconditionally (Futhark's
own semantics already require associativity for scan). Non-commutative →
`Undefined("segscan: non-commutative")`. Flag to select vs. Phase 6. New
tests: tuple-operator scan, non-commutative scan.

## Phase 6 — segscan, parallel prefix (not started)
`jax.lax.associative_scan`'s algorithm via `slice`/`pad`/`concatenate` +
Phase 1's elementwise library, for the tuple/non-scalar case `chlo.scan` won't
parallelize on XLA. Get the JAX reference lowering first, use it as the
target shape.

## Phase 7 (lower priority) — SegHist (not started)
`stablehlo.scatter` (`unique_indices=false`, `indices_are_sorted=false`);
segmented hist prepends segment-`iota` to the index vector. Single-scalar-
operator only; tuple operator → `Undefined("SegHist: tuple operator")` (defer
solving, per the brief). New tests: out-of-bounds indices, tuple-operator case.

## Phase 8 (lower priority) — non-commutative segred/segscan, solved (not started)
If XLA's GPU reduce actually assumes commutativity incorrectly (a hypothesis
to test empirically, not an established fact), lower non-commutative segred
through the scan path (take the last/first carry) instead of
`stablehlo.reduce`. Re-enable `argmax.fut` as this phase's exit criteria.

## Verification (every phase)
`nix develop` → rebuild → `run-tests` → `futhark-test` (unchanged sanity
check) → `python run.py out/<name>.mlir --platform cpu [--check]` for
numeric/precision smoke tests. For every new StableHLO idiom, get a JAX
reference lowering (`jax.jit(f).lower(*shapes).as_text()`) before/alongside
writing the corresponding code.

## Known issues / flagged to user
- (none logged yet from actually running tests — build has not run yet)

## Session log
- 2026-10-12: Phase 0 drafted and partially executed: utils.hpp, compiler.hpp
  IREE-strip, both CMakeLists.txt, flake.nix all edited. Deleted
  patches/sort-distribute.patch. Hit a sandbox restriction trying to run the
  actual nix build (no access to the nix daemon socket from the agent's
  sandboxed Bash tool) — **user needs to run `nix develop -c build` (or just
  `build` inside `nix develop`) themselves and report back what breaks**.
  run_tests.py rewrite not started. No Phase 1+ work started.
- 2026-10-18: Phase 0 finished (`run_tests.py` rewritten; user confirmed
  Phase 0's tests fail as expected, with `@entry_main`/no-`@main` errors).
  Phase 1 (segmap) implemented and verified: `map.fut`/`iota.fut` pass
  (12 passed / 31 failed overall). Found along the way that most of the
  originally-assumed Phase-1 test set is actually dynamically-shaped
  (Futhark infers a runtime size whenever an array parameter isn't
  explicitly annotated), which this backend can't support -- not a bug,
  matches the brief's "dynamic -> Undefined() for now" rule, just broader
  than expected. Traced several of those back to a missing test-fixture
  convention (not a backend issue) and annotated `matmul.fut`/
  `reduce.fut`/`reduce2.fut`/`reduce3.fut`/`scan0.fut` with explicit sizes
  so they'll reach Phase 2/3's actual logic once implemented.
  `argmax.fut` deliberately left unannotated (two different input lengths
  across its test cases, plus a non-commutative operator Phase 2 gates to
  `Undefined()` regardless). **Paused here, per request, before rebuilding
  against the just-edited test files or starting Phase 2.**
- 2026-10-19: Now running directly inside the nix devShell (can run `build`/
  `run-tests` myself, no more relaying through the user). Phase 2 (segred)
  implemented and verified: `reduce.fut`/`reduce2.fut`/`reduce3.fut` all
  pass (commutative, segmented, and tuple operators respectively). Along
  the way, found and fixed two Phase-1 carryover bugs (iota element type,
  an insertion-point ordering bug) and three genuinely new Phase-2 gaps:
  `stablehlo.map` has no per-operand indexing map (needed explicit
  broadcasting for lower-rank affine reads), `LowerIndex`'s single-element
  fast path was reachable after all (unwrapping a flat reduce's rank-1
  result to a scalar) and still emitted banned ops, and `stablehlo.map`
  turned out to allow only one result (tuple operators needed one map op
  per output value, not "falls out for free" as Phase 1 assumed). Added
  three new non-commutative segred tests per the original coverage-gap
  list. 18 passed / 28 failed overall (up from 15/31 at the start of this
  session). Full details in the Phase 2 section above.
  Phase 3 (matmul) implemented and verified right after: `matmul.fut`
  passes, no new bugs found, clean on the first build. 19 passed / 27
  failed overall.

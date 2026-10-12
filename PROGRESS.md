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

## Phase 1 — segmap (not started)

Load-bearing: builds the scalar-op library (decision 1) and a new
`FindSegOpFreeVars` helper (`stablehlo.map`'s region is isolated-from-above,
unlike today's `linalg.generic` — free variables must be `broadcast_in_dim`'d
to full shape and passed as extra operands; thread indices become
`stablehlo.iota` operands). Also folds in: entry→`@main` rename + multi-entry
hard error + drop `futhark.size_args`; while/for loop →
`Undefined("loop")` (delete the `scf.for` machinery outright — brief wants
all loops undefined); new standalone `tests/loop_undefined.fut` and
`tests/if_basic.fut`-style coverage gaps noted in later phases.
Tests: `map.fut`, `map2-5.fut`, `iota.fut`, `gelu.fut`; `map_dynamic.fut`/
`iota_dynamic.fut` expected to cleanly `Undefined()` (dynamic sizes).

## Phase 2 — segred (not started)
`LowerSegRed` → `LowerSegMap` (per-element part) + `stablehlo.reduce` over the
innermost dim. Tuple operators fall out for free (reduce is natively
variadic). Non-commutative → `Undefined("segred: non-commutative")` for now
(Phase 8 revisits) — **this regresses `argmax.fut` from passing to
Undefined() until Phase 8, expected per the brief's own "lower priority"
framing**. New tests: 2×2 matrix-product reduce, affine-composition
`(a,b)∘(c,d)` reduce, segmented non-commutative variant.

## Phase 3 — matmul (not started)
`LowerMatMul` → `stablehlo.dot_general`, reusing `LowerDotGeneral`'s existing
`DotDimensionNumbersAttr` pattern (already working StableHLO code today).
`precision`: default to always emitting `DEFAULT`, rely on `example_run.py`'s
existing `--precision`/`set_precision` text post-processing instead of a new
backend flag.

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

You are a coding agent for this session. (You are allowed to write code.)
You are working on an MLIR backend for Futhark. The output of the backend is MLIR module text files.

## Goal
Rewrite our MLIR backend (this directory) so it emits StableHLO for XLA instead of linalg for IREE. It ingests Futhark GPU IR before memory expansion (fully flattened); reuse the existing infrastructure for traversing that IR. Remove the linalg/IREE path entirely.

Anything that can't be translated must be a compile-time error `Undefined()`, with the construct named in the message (e.g. `Undefined(): SegHist`). Never emit something partial or approximate. Dynamic-size paths may be `Undefined()` for now.

## Hard constraints from XLA (verified)
- The entry function must be `func.func @main`.
- Allowed dialects: whatever XLA accepts, i.e. `func`, `stablehlo`, `chlo`. XLA **rejects** `tensor.*` (reshape, collapse_shape, expand_shape…), `arith.*` on tensors, `linalg.*` and `scf.*`. Use the StableHLO equivalents (`stablehlo.reshape`, `stablehlo.constant`, `stablehlo.add`, …).
- Scalars are 0-d tensors (`tensor<f32>`). Shapes must be static.

## Mapping (Futhark GPU IR → StableHLO)
Overall you can assume the same limitations as there are
in the current backend (if helpful).
- **segmap** (elementwise body): emit a `stablehlo.map` whose region is the kernel body on 0-d tensors (`dimensions` = all dims). The region can't capture outer values: broadcast free variables to the full shape (`broadcast_in_dim`) and pass them as extra operands, and pass thread indices as `stablehlo.iota` operands. A read at the thread's own index is just the corresponding operand; other indexing → `Undefined()`. (Optionally keep a flag to emit the body as direct elementwise ops on full tensors instead, for comparison.)
- **segred**: use segmap path for the fused map part, then `stablehlo.reduce` over the segment (innermost) dimensions. Use variadic `reduce` for tuple operators, and the neutral elements as `init_values`.
  - Commutative operators: `stablehlo.reduce`.
  - Non-commutative operators: the StableHLO spec preserves element order, but XLA's GPU implementation may assume commutativity. Add a test (e.g. 2×2 matrix product, or affine-map composition `(a,b)∘(c,d)`). If XLA gets it wrong, lower non-commutative segred through the scan path (take the last element) instead. Make this a lower priority though, at first just comment it as a todo/`Undefined()`.
- **segscan**: XLA only generates a parallel kernel for a single-array, scalar, associative scan with an identity init. Everything else becomes a sequential loop. So:
  1. First: `chlo.scan` with `is_associative = true`, `inits` = the neutral element (broadcast), forward, and the final carry unused. Body: `(x, c) -> (c ⊕ x, c ⊕ x)` (inclusive). Tuple operators use variadic inputs and carries.
  2. Then, for tuple or non-scalar operators: emit a parallel prefix scan from `slice`, `pad`, `concatenate` and elementwise ops, following `jax.lax.associative_scan`'s algorithm. Keep the `chlo.scan` lowering selectable by a flag.
- **SegHist** → one `stablehlo.scatter` per HistOp: operand = destination histogram, scatter_indices = bucket indices (from the fused map body), updates = values, update_computation = the operator (scalar region). Variadic scatter for tuple operators; for segmented histograms, prepend the segment index (iota) to the index vector. Set `unique_indices = false`, `indices_are_sorted = false`. Out-of-bounds updates are skipped by StableHLO semantics, matching Futhark. Add tests with with out-of-bounds indices. Also add tests with a non-trivial operator (e.g. a tuple operator), but defer actually solving that until later (`Undefined()`). 
- **Black-box matmul**: `stablehlo.dot_general` with the right `batching_dims` / `contracting_dims`. Make `precision` configurable (`DEFAULT` allows TF32 on A100, `HIGHEST` is full FP32). 
- **if** → `stablehlo.if` (or `stablehlo.case`).
- **while / loops** → `Undefined()` for now.
- Anything else (non-elementwise kernel bodies, …) → `Undefined()`.

## Reference and validation
- For XLA-friendly StableHLO idioms, write the equivalent in JAX and lower it: `jax.jit(f).lower(*shapes).as_text()`. For example `jnp.cumsum`, `lax.associative_scan`, `lax.reduce`, `lax.cond`, `jnp.einsum`.
  Use `nix` to get a jax shell (cpu only).
- Validate emitted modules by compiling and running them with `run.py` ([path]). It compiles StableHLO with JAX's XLA, using random inputs from `@main`'s signature. `--check` compares against XLA:CPU, and `--platform cpu` works without a GPU.
- Extend the existing tests ([path]) so each construct is checked against Futhark's own results: segmap; segred (commutative, non-commutative, segmented); segscan (scalar, tuple, non-commutative); matmul; `if`; and at least one `Undefined()` case.

The output of the compiler:  a text file *.mlir.

Setup: change @flake.nix to drop the IREE dependency. Pick MLIR/LLVM packages that do not depend on IREE, are compatible with our backend and gives us the least things to compile from scratch (i.e., try to hit nix binary caches). We will ingest the *.mlir files using JAX's XLA, so add jax (cpu-only) to the nix environment. See @example_run.py for an example of how to compile/run programs. The new test framework can compare against jax code exported to stableHLO and run similarly (or just interpreted).

Work incrementally: setup, segmap → segred → matmul → if → segscan (chlo.scan) → segscan (parallel prefix). At all times keep an eye on making the existing tests pass. Run the tests after each step. Keep track of tasks, so that I can start and stop you (this will take a long time).


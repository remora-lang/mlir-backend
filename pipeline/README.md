# pipeline

Run a program with `./run PATH/TO/prog.fut [input]` (or `prog.fut_gpu` to skip
straight to the IR stage).

It lowers the program through `gpu-ir` (a .fut source only) -> `compile` ->
`mlir-opt` -> `mlir-translate` -> `clang` (linked against `<name>_scaffold.c`),
executes it, and leaves intermediates in `out/`.

`gpu-ir` (defined in `flake.nix`) is `futhark dev --gpu --strip-provenance
--no-grid --no-assert --simplify`.

To support a new `prog.fut_gpu`:

1. Run `compile` on it to read the `entry_main` signature.
2. Copy the closest existing `*_scaffold.c`.
3. Adjust the memref element type, the `_mlir_ciface_entry_main` argument types, and the print format.

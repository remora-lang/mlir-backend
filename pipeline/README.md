# pipeline

Run a program with `./run PATH/TO/prog.fut_gpu [input]`.

It lowers the program through `compile` -> `mlir-opt` -> `mlir-translate` -> `clang` (linked against `<name>_scaffold.c`), executes it, and leaves intermediates in `out/`.

To support a new `prog.fut_gpu`:

1. Run `compile` on it to read the `entry_main` signature.
2. Copy the closest existing `*_scaffold.c`.
3. Adjust the memref element type, the `_mlir_ciface_entry_main` argument types, and the print format.

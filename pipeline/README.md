# pipeline

Run a program end-to-end through the MLIR backend and the IREE runtime with
`run-iree` (defined in `flake.nix`):

    run-iree PATH/TO/prog.fut [--input=... ...]

It lowers the program through `compile` (Futhark source -> GPU IR -> MLIR),
then `iree-compile` (MLIR -> `.vmfb` for the `llvm-cpu` backend), and finally
`iree-run-module` (`--device=local-task`, `--function=entry_main`).
Intermediates (`.mlir`, `.vmfb`) land in `out/`.

Pass a `prog.fut_gpu` instead to skip the Futhark stage.

## Inputs

Each `entry_main` argument is one `--input` flag, in order. Scalars take a bare
value; tensors take `[shape]xdtype=values`. Futhark passes the size of every
dynamic array dimension as an explicit leading `i64` argument, so those come
first. Check the exact signature with `compile prog.fut | grep entry_main`.

    run-iree mlir-backend/tests/map.fut    --input="10"                                                    # 4xi32=11 12 13 14
    run-iree mlir-backend/tests/map2.fut   --input="8"                                                     # 8xi64=8 9 10 11 12 13 14 15
    run-iree mlir-backend/tests/map3.fut   --input="2" --input="3" --input="2x3xi32=1 2 3 4 5 6" --input="100"  # 2x3xi32=[101 102 103][104 105 106]
    run-iree mlir-backend/tests/reduce.fut --input="4" --input="4xi32=1 2 3 4"                             # i32=10

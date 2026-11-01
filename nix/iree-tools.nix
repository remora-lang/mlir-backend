# Stock IREE tools (iree-compile / iree-run-module / iree-benchmark-module),
# built from the `iree` input only. Independent of ./., so mlir-backend edits
# never rebuild IREE. Flags mirror the IREE_* options package.nix relies on.
{ pkgs, iree }:

let
  cudatoolkit = pkgs.cudaPackages_12.cudatoolkit;
in
pkgs.stdenv.mkDerivation {
  pname = "iree-tools";
  version = "dev";

  src = iree;

  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    pkg-config
    python3
    git
  ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ patchelf ];

  buildInputs = pkgs.lib.optionals (!pkgs.stdenv.isDarwin) [
    cudatoolkit
  ];

  IREE_CUDA_DEPS_DIR = if pkgs.stdenv.isDarwin then "" else "${cudatoolkit}";

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DIREE_BUILD_COMPILER=ON"
    "-DIREE_BUILD_BUNDLED_LLVM=ON"
    "-DIREE_INPUT_STABLEHLO=ON"
    "-DIREE_INPUT_TORCH=OFF"
    "-DIREE_INPUT_TOSA=OFF"
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_BUILD_SAMPLES=OFF"
    "-DIREE_BUILD_PYTHON_BINDINGS=OFF"
    "-DIREE_ENABLE_LIBBACKTRACE=OFF"
  ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
    "-DIREE_TARGET_BACKEND_CUDA=ON"
    "-DIREE_HAL_DRIVER_CUDA=ON"
  ];

  # Only the tools the harness drives.
  ninjaFlags = [ "iree-compile" "iree-run-module" "iree-benchmark-module" ];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/lib
    cp tools/iree-compile tools/iree-run-module tools/iree-benchmark-module $out/bin/
    # Tools link libIREECompiler etc. via @rpath; ship every shared lib and
    # repoint rpath at $out/lib (postFixup). Copy all to avoid a second rebuild
    # if another lib turns out to be needed.
    find lib -maxdepth 1 \( -name '*.dylib' -o -name '*.so*' \) \
      -exec cp -a {} $out/lib/ \;
    runHook postInstall
  '';

  postFixup = if pkgs.stdenv.isDarwin then ''
    for b in $out/bin/*; do install_name_tool -add_rpath $out/lib "$b" || true; done
  '' else ''
    for b in $out/bin/*; do patchelf --add-rpath $out/lib "$b" || true; done
  '';
}

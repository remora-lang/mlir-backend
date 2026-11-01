# The mlir-backend compiler executable.
{ pkgs, iree, src }:

pkgs.stdenv.mkDerivation {
  pname = "futhark-mlir-backend";
  version = "dev";

  inherit src;

  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    pkg-config
    openjdk
    antlr
    python3
    git
  ];

  buildInputs = with pkgs; [
    antlr4.runtime.cpp
  ];

  cmakeFlags = [
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    "-DMOCHA_IREE_SOURCE_DIR=${iree}"
    "-DIREE_ENABLE_LIBBACKTRACE=OFF"
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp mlir-backend/mlir-backend $out/bin/

    runHook postInstall
  '';
}

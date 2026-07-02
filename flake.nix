{
  description = "Futhark MLIR backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      build = pkgs.writeShellScriptBin "build" ''
        set -euo pipefail
        if [ ! -f build/build.ninja ]; then
          cmake -S . -B build -G Ninja
        fi
        ninja -C build
      '';

      run = pkgs.writeShellScriptBin "run" ''
        set -euo pipefail
        build
        exec ./build/mlir-backend/mlir-backend "$@"
      '';

      clean = pkgs.writeShellScriptBin "clean" ''
        set -euo pipefail
        rm -rf build
      '';
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          llvmPackages_22.llvm
          llvmPackages_22.mlir
          antlr4.runtime.cpp
          antlr4
          openjdk
          cmake
          ninja
          clang
          pkg-config
          build
          run
          clean
        ];
      };
    };
}

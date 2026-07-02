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
          cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        fi
        ninja -C build
        ln -sf build/compile_commands.json compile_commands.json
      '';

      run = pkgs.writeShellScriptBin "run" ''
        set -euo pipefail
        build
        exec ./build/mlir-backend/mlir-backend "$@"
      '';

      clean = pkgs.writeShellScriptBin "clean" ''
        set -euo pipefail
        rm -rf build compile_commands.json
      '';

      # Wrap clangd so it can discover the nix-store compiler's builtin
      # include paths — without --query-driver it can't find <iostream> etc.
      clangd = pkgs.writeShellScriptBin "clangd" ''
        exec ${pkgs.llvmPackages_22.clang-tools}/bin/clangd --query-driver='**' "$@"
      '';
    in {
      devShells.${system}.default = pkgs.mkShell {
        # clangd wrapper must come before `clang` so it shadows the clangd
        # symlinked into pkgs.clang's bin dir.
        packages = with pkgs; [
          clangd
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

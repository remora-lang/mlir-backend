{
  description = "Futhark MLIR backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    systems = [ "x86_64-linux" "aarch64-darwin" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
      f (import nixpkgs { inherit system; }));

    perSystem = pkgs:
    let
      mlir-backend = pkgs.stdenv.mkDerivation {
      pname = "futhark-mlir-backend";
      version = "dev";

      src = ./.;

      nativeBuildInputs = with pkgs; [
        cmake
        ninja
        pkg-config
        openjdk
        antlr
      ];

      buildInputs = with pkgs; [
        llvmPackages_22.llvm
        llvmPackages_22.mlir
        antlr4.runtime.cpp
      ];

      cmakeFlags = [
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
      ];

      # Optional if your executable is somewhere unusual.
      installPhase = ''
        runHook preInstall

        mkdir -p $out/bin
        cp mlir-backend/mlir-backend $out/bin/

        runHook postInstall
      '';
    };

    build = pkgs.writeShellScriptBin "build" ''
      set -euo pipefail
      if [ ! -f build/build.ninja ]; then
        cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      fi
      ninja -C build
      ln -sf build/compile_commands.json compile_commands.json
    '';

    # `compile x.fut` first lowers to GPU IR (kept as x.fut_gpu beside the
    # source), then compiles that and prints the MLIR. Any other input
    # (e.g. x.fut_gpu) is compiled directly.
    compile = pkgs.writeShellScriptBin "compile" ''
      set -euo pipefail
      build >&2
      in=''${1:-}
      if [ -n "$in" ] && [ "''${in%.fut}" != "$in" ]; then
        ir="''${in%.fut}.fut_gpu"
        ${gpu-ir}/bin/gpu-ir "$in" > "$ir"
        exec ./build/mlir-backend/mlir-backend "$ir"
      fi
      exec ./build/mlir-backend/mlir-backend "$@"
    '';

    gpu-ir = pkgs.writeShellScriptBin "gpu-ir" ''
      futhark dev --gpu --strip-provenance --no-grid --simplify "$1"
    '';

    clean = pkgs.writeShellScriptBin "clean" ''
      set -euo pipefail
      rm -rf build compile_commands.json
    '';

    clangd = pkgs.writeShellScriptBin "clangd" ''
      exec ${pkgs.llvmPackages_22.clang-tools}/bin/clangd \
        --query-driver='**' "$@"
    '';

    in {
      package = mlir-backend;

      devShell = pkgs.mkShell {
        inputsFrom = [ mlir-backend ];

        packages = with pkgs; [
          clangd
          antlr4
          openjdk
          build
          compile
          gpu-ir
          clean
          llvmPackages_22.lldb
        ] ++ pkgs.lib.optionals (!pkgs.stdenv.isDarwin) [
          # Linux stdenv is gcc-based, so add clang for an LLVM toolchain.
          # On Darwin the stdenv is already clang-based; a second clang here
          # is redundant and can clash with the stdenv/llvmPackages_22 clang.
          clang
        ];
      };
    };
  in {
    packages = forAllSystems (pkgs: { default = (perSystem pkgs).package; });
    devShells = forAllSystems (pkgs: { default = (perSystem pkgs).devShell; });
  };
}

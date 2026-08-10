{
  description = "Futhark MLIR backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    iree = {
      type = "git";
      url = "https://github.com/iree-org/iree.git";
      rev = "3699452d60c6dfe483df4675c7e98299fe790ec4";
      submodules = true;
      flake = false;
    };
  };

  outputs = { self, nixpkgs, iree }:
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
        python3
        git
        futhark
      ];

      buildInputs = with pkgs; [
        antlr4.runtime.cpp
      ];

      cmakeFlags = [
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DMOCHA_IREE_SOURCE_DIR=${iree}"
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
      if [ ! -f build-iree-nix/build.ninja ]; then
        cmake -S . -B build-iree-nix -G Ninja \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DMOCHA_IREE_SOURCE_DIR=${iree}
      fi
      cmake --build build-iree-nix \
        --target mlir-backend iree-compile iree-run-module
      ln -sfn build-iree-nix/compile_commands.json compile_commands.json
    '';

    # `compile x.fut` first lowers to GPU IR (kept as x.fut_gpu beside the
    # source), then compiles that and prints the MLIR. Any other input
    # (e.g. x.fut_gpu) is compiled directly.
    compile = pkgs.writeShellScriptBin "compile" ''
      set -euo pipefail
      ${build}/bin/build >&2
      in=''${1:-}
      if [ -n "$in" ] && [ "''${in%.fut}" != "$in" ]; then
        ir="''${in%.fut}.fut_gpu"
        ${gpu-ir}/bin/gpu-ir "$in" > "$ir"
        exec ./build-iree-nix/mlir-backend/mlir-backend "$ir"
      fi
      exec ./build-iree-nix/mlir-backend/mlir-backend "$@"
    '';

    ireeCompile = pkgs.writeShellScriptBin "iree-compile-local" ''
      set -euo pipefail
      ${build}/bin/build >&2
      exec ./build-iree-nix/iree/tools/iree-compile "$@"
    '';

    ireeRunModule = pkgs.writeShellScriptBin "iree-run-module-local" ''
      set -euo pipefail
      ${build}/bin/build >&2
      exec ./build-iree-nix/iree/tools/iree-run-module "$@"
    '';

    runIree = pkgs.writeShellScriptBin "run-iree" ''
      set -euo pipefail
      if [ "$#" -lt 1 ]; then
        echo "usage: run-iree FILE.fut|FILE.fut_gpu [iree-run-module arguments...]" >&2
        exit 2
      fi

      file="$1"
      shift
      name=$(basename "$file")
      name=''${name%.fut_gpu}
      name=''${name%.fut}
      mkdir -p pipeline/out

      mlir_file="pipeline/out/$name.mlir"
      vmfb_file="pipeline/out/$name.vmfb"
      ${compile}/bin/compile "$file" > "$mlir_file"

      ./build-iree-nix/iree/tools/iree-compile "$mlir_file" \
        --iree-hal-target-device=local \
        --iree-hal-local-target-device-backends=llvm-cpu \
        -o "$vmfb_file"

      exec ./build-iree-nix/iree/tools/iree-run-module \
        --module="$vmfb_file" \
        --device=local-task \
        --function=entry_main \
        "$@"
    '';

    gpu-ir = pkgs.writeShellScriptBin "gpu-ir" ''
      futhark dev --gpu --strip-provenance --no-grid --simplify "$1"
    '';

    clean = pkgs.writeShellScriptBin "clean" ''
      set -euo pipefail
      rm -rf build-iree-nix compile_commands.json
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
          ireeCompile
          ireeRunModule
          runIree
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

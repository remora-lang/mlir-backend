{
  description = "Futhark MLIR backend (StableHLO/XLA target)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # StableHLO pins LLVM at a rolling trunk commit, not a tagged release;
    # this is the llvm-project commit named by stablehlo's own
    # build_tools/llvm_version.txt at tag v1.20.0.
    llvm-project = {
      type = "git";
      url = "https://github.com/llvm/llvm-project.git";
      rev = "ab547095ead5464dc024d66264d9b8a987f429f3";
      shallow = true;
      flake = false;
    };

    stablehlo = {
      type = "git";
      url = "https://github.com/openxla/stablehlo.git";
      rev = "639335932274930617d9724fb25ec9f2f52c5e40"; # v1.20.0
      shallow = true;
      flake = false;
    };
  };

  outputs = { self, nixpkgs, llvm-project, stablehlo }:
  let
    systems = [ "x86_64-linux" "aarch64-darwin" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
      f (import nixpkgs { inherit system; }));

    perSystem = pkgs:
    let
      # stablehlo/integrations doesn't build cleanly under
      # LLVM_EXTERNAL_PROJECTS mode (see patches/stablehlo-no-integrations.patch)
      # and we don't need it (no C++/Python "builder" API usage).
      stablehloSrc = pkgs.applyPatches {
        name = "stablehlo-no-integrations";
        src = stablehlo;
        patches = [ ./patches/stablehlo-no-integrations.patch ];
      };

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
        ];

        buildInputs = with pkgs; [
          antlr4.runtime.cpp
        ];

        cmakeFlags = [
          "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
          "-DMOCHA_LLVM_SOURCE_DIR=${llvm-project}"
          "-DMOCHA_STABLEHLO_SOURCE_DIR=${stablehloSrc}"
        ];

        installPhase = ''
          runHook preInstall

          mkdir -p $out/bin
          cp mlir-backend/mlir-backend $out/bin/

          runHook postInstall
        '';
      };

      build = pkgs.writeShellScriptBin "build" ''
        set -euo pipefail
        jobs=''${1:-}
        export CCACHE_BASEDIR="$PWD"
        export CCACHE_NOHASHDIR=1
        export CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
        if [ ! -f build/build.ninja ]; then
          cmake -S . -B build -G Ninja \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DMOCHA_LLVM_SOURCE_DIR=${llvm-project} \
            -DMOCHA_STABLEHLO_SOURCE_DIR=${stablehloSrc}
        fi
        cmake --build build ''${jobs:+-j "$jobs"} --target mlir-backend
        ln -sfn build/compile_commands.json compile_commands.json
      '';

      # `compile x.fut` first lowers to GPU IR (kept as x.fut_gpu beside the
      # source), then compiles that and prints the MLIR. Any other input
      # (e.g. x.fut_gpu) is compiled directly.
      compile = pkgs.writeShellScriptBin "compile" ''
        set -euo pipefail
        ${build}/bin/build >&2
        in=''${1:-}
        if [ -n "$in" ] && [ "''${in%.fut}" != "$in" ]; then
          shift
          ir="''${in%.fut}.fut_gpu"
          ${gpu-ir}/bin/gpu-ir "$in" > "$ir"
          exec ./build/mlir-backend/mlir-backend "$ir" "$@"
        fi
        exec ./build/mlir-backend/mlir-backend "$@"
      '';

      gpu-ir = pkgs.writeShellScriptBin "gpu-ir" ''
        futhark dev --gpu --strip-provenance --no-grid --simplify "$1"
      '';

      run-tests = pkgs.writeShellScriptBin "run-tests" ''
        set -euo pipefail
        ${build}/bin/build >&2
        exec ${pkgs.python3}/bin/python3 run_tests.py "$@"
      '';

      # Validates the `-- input`/`-- output` blocks against Futhark's own
      # reference backend (independent of the MLIR/XLA pipeline). Defaults to
      # every test if no files are given.
      futhark-test = pkgs.writeShellScriptBin "futhark-test" ''
        set -euo pipefail
        if [ "$#" -eq 0 ]; then
          set -- tests/*.fut
        fi
        exec futhark test --backend=c "$@"
      '';

      clean = pkgs.writeShellScriptBin "clean" ''
        set -euo pipefail
        rm -rf build/mlir-backend
      '';

      clean-all = pkgs.writeShellScriptBin "clean-all" ''
        set -euo pipefail
        rm -rf build compile_commands.json
      '';

      clangd = pkgs.writeShellScriptBin "clangd" ''
        exec ${pkgs.llvmPackages_22.clang-tools}/bin/clangd \
          --query-driver='**' "$@"
      '';

      pythonWithJax = pkgs.python3.withPackages (ps: [ ps.jax ps.numpy ]);

    in {
      package = mlir-backend;

      devShell = pkgs.mkShell {
        inputsFrom = [ mlir-backend ];

        packages = with pkgs; [
          ccache
          clangd
          antlr4
          openjdk
          build
          compile
          gpu-ir
          clean
          clean-all
          run-tests
          futhark-test
          pythonWithJax
          llvmPackages_22.lldb
        ] ++ pkgs.lib.optionals (!pkgs.stdenv.isDarwin) [
          # Linux stdenv is gcc-based, so add clang for an LLVM toolchain.
          # On Darwin the stdenv is already clang-based; a second clang here
          # is redundant and can clash with the stdenv/llvmPackages_22 clang.
          clang
        ];

        # Expose the pinned StableHLO/MLIR sources (read-only nix store paths)
        # for reference during development.
        shellHook = ''
          ln -sfn ${stablehloSrc} third-party/stablehlo-src
          ln -sfn ${llvm-project}/mlir third-party/mlir-src
          for l in third-party/stablehlo-src third-party/mlir-src; do
            grep -qxF "!$l" .ignore 2>/dev/null || echo "!$l" >> .ignore
          done
          [ -f .ripgreprc ] || printf -- '--follow\n' > .ripgreprc
          export RIPGREP_CONFIG_PATH="$PWD/.ripgreprc"
        '';
      };
    };
  in {
    packages = forAllSystems (pkgs: { default = (perSystem pkgs).package; });
    devShells = forAllSystems (pkgs: { default = (perSystem pkgs).devShell; });
  };
}

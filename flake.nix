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
      f (import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      }));

    perSystem = pkgs:
    let
      cudatoolkit = pkgs.cudaPackages_12.cudatoolkit;
      cudaDepsDir = if pkgs.stdenv.isDarwin then "" else "${cudatoolkit}";

      # IREE source with the cuda sort fix: route iree_linalg_ext.sort to the
      # Distribute pipeline (TileAndFuse fails to bufferize it). See
      # patches/sort-distribute.patch.
      ireeSrc = pkgs.applyPatches {
        name = "iree-sort-distribute";
        src = iree;
        patches = [ ./patches/sort-distribute.patch ];
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
      ] ++ pkgs.lib.optionals (!pkgs.stdenv.isDarwin) [
        cudatoolkit
      ];

      IREE_CUDA_DEPS_DIR = cudaDepsDir;

      cmakeFlags = [
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DMOCHA_IREE_SOURCE_DIR=${ireeSrc}"
        "-DIREE_ENABLE_LIBBACKTRACE=OFF"
      ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
        "-DMOCHA_ENABLE_CUDA=ON"
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
      jobs=''${1:-}
      ${nixpkgs.lib.optionalString (cudaDepsDir != "") ''export IREE_CUDA_DEPS_DIR=${cudaDepsDir}''}
      export CCACHE_BASEDIR="$PWD"
      export CCACHE_NOHASHDIR=1
      export CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
      cuda_flag="-DMOCHA_ENABLE_CUDA=OFF"
      if [ -n "''${CUDA:-}" ]; then cuda_flag="-DMOCHA_ENABLE_CUDA=ON"; fi
      if [ ! -f build/build.ninja ]; then
        cmake -S . -B build -G Ninja \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER_LAUNCHER=ccache \
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
          $cuda_flag \
          -DMOCHA_IREE_SOURCE_DIR=${ireeSrc}
      fi
      cmake --build build ''${jobs:+-j "$jobs"} \
        --target mlir-backend iree-compile iree-run-module iree-benchmark-module
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
        ir="''${in%.fut}.fut_gpu"
        ${gpu-ir}/bin/gpu-ir "$in" > "$ir"
        exec ./build/mlir-backend/mlir-backend "$ir"
      fi
      exec ./build/mlir-backend/mlir-backend "$@"
    '';

    ireeCompile = pkgs.writeShellScriptBin "iree-compile-local" ''
      set -euo pipefail
      ${build}/bin/build >&2
      exec ./build/iree/tools/iree-compile "$@"
    '';

    ireeRunModule = pkgs.writeShellScriptBin "iree-run-module-local" ''
      set -euo pipefail
      ${build}/bin/build >&2
      exec ./build/iree/tools/iree-run-module "$@"
    '';

    # Shared body for run-iree/iree-benchmark: parse an optional backend flag,
    # compile FILE.fut -> out/$name.vmfb for that backend, and set $vmfb_file
    # and $device for the caller's final exec. Expects $prog (program name, for
    # usage) and $tool (downstream iree tool name, for usage) to be set first.
    ireeSetup = ''
      set -euo pipefail

      target=local
      case "''${1:-}" in
        --vulkan) target=vulkan; shift ;;
        --metal) target=metal; shift ;;
        --cuda) target=cuda; shift ;;
      esac

      if [ "$#" -lt 1 ]; then
        echo "usage: $prog [--vulkan|--metal|--cuda] FILE.fut|FILE.fut_gpu [$tool arguments...]" >&2
        exit 2
      fi

      file="$1"
      shift
      name=$(basename "$file")
      name=''${name%.fut_gpu}
      name=''${name%.fut}
      mkdir -p out

      mlir_file="out/$name.mlir"
      vmfb_file="out/$name.vmfb"
      ${compile}/bin/compile "$file" > "$mlir_file"

      case "$target" in
        vulkan)
          # sm_89 = Ada Lovelace (RTX 40xx)
          ./build/iree/tools/iree-compile "$mlir_file" \
            --iree-hal-target-device=vulkan \
            --iree-vulkan-target=sm_89 \
            --iree-dispatch-creation-enable-split-reduction \
            -o "$vmfb_file"
          device=vulkan
          ;;
        metal)
          # Embed MSL source instead of a .metallib so we don't need Xcode's
          # `metal`/`metallib` tools (absent from the nix apple-sdk xcrun); the
          # Metal runtime compiles the source once at module load.
          ./build/iree/tools/iree-compile "$mlir_file" \
            --iree-hal-target-device=metal \
            --iree-metal-compile-to-metallib=false \
            --iree-dispatch-creation-enable-split-reduction \
            -o "$vmfb_file"
          device=metal
          ;;
        cuda)
          ./build/iree/tools/iree-compile "$mlir_file" \
            --iree-hal-target-device=cuda \
            --iree-cuda-target=sm_89 \
            --iree-dispatch-creation-enable-split-reduction \
            -o "$vmfb_file"
          device=cuda
          ;;
        *)
          ./build/iree/tools/iree-compile "$mlir_file" \
            --iree-hal-target-device=local \
            --iree-hal-local-target-device-backends=llvm-cpu \
            -o "$vmfb_file"
          device=local-task
          ;;
      esac
    '';

    # run-iree [--compile-only|--run-only] [--vulkan|...] FILE [iree args...]
    #
    # The test runner needs entry_main's signature (written to out/$name.mlir
    # during compilation) *before* it can build the --input flags. So the two
    # phases are separable: --compile-only produces out/$name.{mlir,vmfb} and
    # exits; --run-only skips compilation and runs the already-built vmfb. With
    # neither flag it compiles and runs, as before.
    runIree = pkgs.writeShellScriptBin "run-iree" ''
      prog=run-iree tool=iree-run-module

      phase=both
      case "''${1:-}" in
        --compile-only) phase=compile; shift ;;
        --run-only)     phase=run;     shift ;;
      esac

      if [ "$phase" = run ]; then
        # No compilation: reconstruct the vmfb path/device from FILE and run.
        # The backend flag is accepted for parity with the matching
        # --compile-only invocation; it only selects the device here.
        device=local-task
        case "''${1:-}" in
          --vulkan) device=vulkan; shift ;;
          --metal)  device=metal;  shift ;;
          --cuda)   device=cuda;   shift ;;
        esac
        file="''${1:?usage: $prog --run-only FILE [args...]}"
        shift
        name=$(basename "$file")
        name=''${name%.fut_gpu}
        name=''${name%.fut}
        exec ./build/iree/tools/iree-run-module \
          --module="out/$name.vmfb" \
          --device="$device" \
          --function=entry_main \
          "$@"
      fi

      ${ireeSetup}

      if [ "$phase" = compile ]; then
        exit 0
      fi

      exec ./build/iree/tools/iree-run-module \
        --module="$vmfb_file" \
        --device="$device" \
        --function=entry_main \
        "$@"
    '';

    iree-benchmark = pkgs.writeShellScriptBin "iree-benchmark" ''
      prog=iree-benchmark tool=iree-benchmark-module
      ${ireeSetup}
      exec ./build/iree/tools/iree-benchmark-module \
        --module="$vmfb_file" \
        --device="$device" \
        --function=entry_main \
        "$@"
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
    # reference backend (independent of the MLIR/IREE pipeline). Defaults to
    # every test if no files are given.
    futhark-test = pkgs.writeShellScriptBin "futhark-test" ''
      set -euo pipefail
      if [ "$#" -eq 0 ]; then
        set -- tests/*.fut
      fi
      exec futhark test --backend=c "$@"
    '';

    # Only removes mlir-backend's build artifacts, leaving the already-built
    # IREE tree (build/iree) intact so it isn't recompiled.
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

    in {
      package = mlir-backend;

      devShell = pkgs.mkShell ({
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
          ireeCompile
          ireeRunModule
          runIree
          run-tests
          iree-benchmark
          futhark-test
          llvmPackages_22.lldb
        ] ++ pkgs.lib.optionals (!pkgs.stdenv.isDarwin) [
          # Linux stdenv is gcc-based, so add clang for an LLVM toolchain.
          # On Darwin the stdenv is already clang-based; a second clang here
          # is redundant and can clash with the stdenv/llvmPackages_22 clang.
          clang
          vulkan-loader
        ];

        # Expose IREE's StableHLO and MLIR sources (read-only nix store paths)
        # for reference during development.
        shellHook = ''
          ln -sfn ${iree}/third_party/stablehlo/stablehlo third-party/stablehlo-src
          ln -sfn ${iree}/third_party/llvm-project/mlir third-party/mlir-src
          for l in third-party/stablehlo-src third-party/mlir-src; do
            grep -qxF "!$l" .ignore 2>/dev/null || echo "!$l" >> .ignore
          done
          [ -f .ripgreprc ] || printf -- '--follow\n' > .ripgreprc
          export RIPGREP_CONFIG_PATH="$PWD/.ripgreprc"
        '';
      } // pkgs.lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
        IREE_HAL_VULKAN_LIBVULKAN_PATH = "${pkgs.vulkan-loader}/lib";
        VK_ICD_FILENAMES = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json";
        LD_LIBRARY_PATH = "/run/opengl-driver/lib";
        IREE_CUDA_DEPS_DIR = cudaDepsDir;
      });
    };
  in {
    packages = forAllSystems (pkgs: { default = (perSystem pkgs).package; });
    devShells = forAllSystems (pkgs: { default = (perSystem pkgs).devShell; });
  };
}

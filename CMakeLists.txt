cmake_minimum_required(VERSION 3.26)

project(mocha LANGUAGES CXX C)

set(MOCHA_LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project"
    CACHE PATH "Path to the llvm-project source tree (llvm+mlir only)")
if(NOT EXISTS "${MOCHA_LLVM_SOURCE_DIR}/llvm/CMakeLists.txt")
  message(FATAL_ERROR
    "llvm-project source tree not found at ${MOCHA_LLVM_SOURCE_DIR}. "
    "Set MOCHA_LLVM_SOURCE_DIR or use the Nix development shell.")
endif()

set(MOCHA_STABLEHLO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stablehlo"
    CACHE PATH "Path to the stablehlo source tree")
if(NOT EXISTS "${MOCHA_STABLEHLO_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "stablehlo source tree not found at ${MOCHA_STABLEHLO_SOURCE_DIR}. "
    "Set MOCHA_STABLEHLO_SOURCE_DIR or use the Nix development shell.")
endif()

# Build only LLVM+MLIR (no clang/lld/other subprojects). StableHLO is pulled
# in as an LLVM "external project" so it configures/builds against this same
# in-tree MLIR, with no separate install step or find_package needed.
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

set(LLVM_EXTERNAL_PROJECTS "stablehlo" CACHE STRING "" FORCE)
set(LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR "${MOCHA_STABLEHLO_SOURCE_DIR}"
    CACHE PATH "" FORCE)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

add_subdirectory("${MOCHA_LLVM_SOURCE_DIR}/llvm" llvm-mlir EXCLUDE_FROM_ALL)
add_subdirectory(mlir-backend)

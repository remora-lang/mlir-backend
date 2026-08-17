cmake_minimum_required(VERSION 3.26)

project(mocha LANGUAGES CXX C)

# Build IREE and its matching LLVM/MLIR revision.
set(IREE_BUILD_COMPILER ON)
set(IREE_BUILD_BUNDLED_LLVM ON)

option(MOCHA_ENABLE_CUDA "Build the IREE CUDA target backend and HAL driver" OFF)
set(IREE_TARGET_BACKEND_CUDA ${MOCHA_ENABLE_CUDA})
set(IREE_HAL_DRIVER_CUDA ${MOCHA_ENABLE_CUDA})

set(IREE_INPUT_STABLEHLO ON)

# Disable components not needed by this backend.
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_BUILD_PYTHON_BINDINGS OFF)
set(IREE_INPUT_TORCH OFF)
set(IREE_INPUT_TOSA OFF)

set(MOCHA_IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree"
    CACHE PATH "Path to the IREE source tree")
if(NOT EXISTS "${MOCHA_IREE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "IREE source tree not found at ${MOCHA_IREE_SOURCE_DIR}. "
    "Set MOCHA_IREE_SOURCE_DIR or use the Nix development shell.")
endif()

add_subdirectory("${MOCHA_IREE_SOURCE_DIR}" iree EXCLUDE_FROM_ALL)
add_subdirectory(mlir-backend)

# MLIR Backend

`mlir-backend` is an MLIR based backend for the Remora programming language.

# Building
### Build LLVM `22.x` from source:

```
cd llvm
mkdir build
cd build
cmake -G "Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ../llvm
ninja
```

The compiler is currently based on LLVM commit `cd70e2d8367493785f1591045db2151540b14b91`.

### Install Antlr4
Install vcpkg:
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
```

Install antlr:
```
vcpkg install antlr4
```


### Clone the repository
```
git clone --recursive https://gitlab.draper.com/mocha/mocha-working.git
```

### Generate build files
```
cd mlir-backend
cmake -S . -G "Ninja" -DLLVM_DIR=[LLVM_DIR] -DMLIR_DIR=[MLIR_DIR] -DCMAKE_TOOLCHAIN_FILE=[VCPKG_ROOT] -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build
```

Example invocation:
```
cmake -S . -G "Ninja" -DLLVM_DIR=../llvm-project/build/lib/cmake/llvm -DMLIR_DIR=../llvm-project/build/lib/cmake/mlir -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build

```

### Build the solution

To build from the commandline:
```
cd build
ninja
```

# Running
To compile a futhark IR file to MLIR:
```
mlir-backend [FUTHARK_FILE_PATH]
```

#include "antlr4-runtime.h"
#include "generated/FutharkLexer.h"
#include "generated/FutharkParser.h"

#include <variant>
#include <string>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <filesystem>

#include "ir/core.hpp"
#include "ir/syntax.hpp"
#include "ir/soac.hpp"
#include "ir/parser.hpp"
#include "ir/compiler.hpp"
#include <config.h>

int main(int argc, char *argv[]) {
  std::filesystem::path base = PROJECT_ROOT_DIR;
  std::string fname = "conv2d_monomorph_10.futhark";

  std::filesystem::path inputPath = base / "tests" / fname;

  if (argc == 2)
    inputPath = argv[1];

  std::ifstream file(inputPath);
  if (!file.good()) {
    std::cout << "Failed to find input file " << inputPath << std::endl;
    return 1;
  }

  auto prog = ParseFuthark(file);

  mlir::MLIRContext ctx;
  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&ctx), &ctx);
  FutharkCompiler compiler(prog, ctx, builder);
  for (auto fun : prog.funs) {
    compiler.LowerFunction(fun);
  }

  compiler.module.print(llvm::outs());
  return 0;
}

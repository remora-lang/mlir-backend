#include "compiler.hpp"
#include "ir/parser.hpp"
#include "ir/syntax.hpp"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <config.h>
#include <filesystem>
#include <string>

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
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &ctx);
  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&ctx), &ctx);
  FutharkCompiler compiler(prog, ctx, builder);
  for (auto fun : prog.funs) {
    compiler.LowerFunction(fun);
  }

  compiler.module.print(llvm::outs());

  return mlir::failed(mlir::verify(compiler.module));
}

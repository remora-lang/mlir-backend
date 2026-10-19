#include "compiler.hpp"
#include "ir/parser.hpp"
#include "ir/syntax.hpp"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <cctype>
#include <config.h>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  std::filesystem::path base = PROJECT_ROOT_DIR;
  std::string fname = "conv2d_monomorph_10.futhark";

  std::filesystem::path inputPath = base / "tests" / fname;
  mlir::stablehlo::Precision precision = mlir::stablehlo::Precision::DEFAULT;
  bool raggedDotCpuFallback = false;

  std::vector<std::string> args(argv + 1, argv + argc);
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--ragged-dot-cpu-fallback") {
      raggedDotCpuFallback = true;
    } else if (args[i] == "--precision") {
      if (i + 1 >= args.size()) {
        std::cout << "--precision requires an argument (default|high|highest)"
                   << std::endl;
        return 1;
      }
      std::string val = args[++i];
      for (char &c : val)
        c = (char)std::toupper((unsigned char)c);
      auto sym = mlir::stablehlo::symbolizePrecision(val);
      if (!sym) {
        std::cout << "invalid --precision value: " << args[i]
                   << " (expected default|high|highest)" << std::endl;
        return 1;
      }
      precision = *sym;
    } else {
      inputPath = args[i];
    }
  }

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
  compiler.precision = precision;
  compiler.raggedDotCpuFallback = raggedDotCpuFallback;
  for (auto fun : prog.funs) {
    compiler.LowerFunction(fun);
  }

  compiler.module.print(llvm::outs());

  return mlir::failed(mlir::verify(compiler.module));
}

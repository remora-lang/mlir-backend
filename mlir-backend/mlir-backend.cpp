#include "compiler.hpp"
#include "ir/parser.hpp"
#include "ir/syntax.hpp"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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
  // Only the entry point is lowered eagerly. Every other top-level function
  // (blackboxes always, and now also a plain #[noinline] helper whose own
  // declared param types are named-size-dependent, e.g. a custom-flattening
  // flatten target) is pulled in lazily, from whichever call site actually
  // reaches it -- see LowerFunction's sizeEnv comment for why a plain
  // helper specifically needs that call-site context, and ExpApply for
  // where it's supplied. A function nothing calls (e.g. a now-dead
  // original callee the flattening rewrite replaced) is simply never
  // visited, which is correct: it has no business appearing in the
  // module at all.
  for (auto &fun : prog.funs) {
    if (fun.entry)
      compiler.LowerFunction(fun);
  }

  // builtin_repiota.hpp/ragged_unflatten_gather.hpp's dummy placeholders
  // (tagged with their shared kDummyAttr) have no valid StableHLO
  // representation of their *real* semantics -- they're only ever correct
  // because the one thing that actually consumes them gets replaced
  // separately (see those files' own comments) and leaves the dummy
  // itself unused. Run the real dead-code-elimination pass StableHLO ops
  // (side-effect-free) already support to confirm that, rather than
  // trusting it -- if a dummy survives, something unexpected depended on
  // it, which would otherwise be a silently wrong result.
  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(compiler.module))) {
    llvm::errs() << "error: post-lowering cleanup pass failed\n";
    return 1;
  }
  bool dummySurvived = false;
  compiler.module.walk([&](mlir::Operation *op) {
    if (op->hasAttr(ragged_unflatten::kDummyAttr)) {
      dummySurvived = true;
      op->emitError("a compiler-generated placeholder survived dead-code "
                    "elimination -- its result was actually needed by "
                    "something live, which would otherwise be a silently "
                    "wrong answer");
    }
  });
  if (dummySurvived)
    return 1;

  compiler.module.print(llvm::outs());

  return mlir::failed(mlir::verify(compiler.module));
}

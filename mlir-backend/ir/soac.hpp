#pragma once
#include "syntax.hpp"

struct SoacHistOp {
  Shape shape;
  SubExp raceFactor;
  std::vector<VName> dest;
  std::vector<SubExp> neutral;
  Lambda op;
};

struct Scan {
  Lambda lambda;
  std::vector<SubExp> neutral;
};

struct Reduce {
  Commutativity comm = Commutativity::Noncommutative;
  Lambda lambda;
  std::vector<SubExp> neutral;
};

struct ScremaForm {
  Lambda scremaLambda;
  std::vector<Scan> scremaScans;
  std::vector<Reduce> scremaReduces;
};

struct SoacStream {
  SubExp size;
  std::vector<VName> arrs;
  std::vector<SubExp> accs;
  Lambda lambda;
};

struct SoacHist {
  SubExp w;
  std::vector<VName> arrs;
  std::vector<SoacHistOp> ops;
  Lambda bucketFun;
};

struct SoacJvp {
  std::vector<SubExp> args;
  std::vector<SubExp> vec;
  Lambda lambda;
};

struct SoacVjp {
  std::vector<SubExp> args;
  std::vector<SubExp> vec;
  Lambda lambda;
};

struct SoacScrema {
  SubExp w;
  std::vector<VName> arrs;
  ScremaForm form;
};

struct Soac {
  std::variant<SoacStream, SoacHist, SoacJvp, SoacVjp, SoacScrema> v;
};
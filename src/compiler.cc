#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <string_view>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "peglib.h"

#define show(a) std::cerr << '[' << __LINE__ << ']' << ' ' << #a << ' ' << a << std::endl;

namespace ir {

using vptr = llvm::Value*;
using tptr = llvm::Type*;

using fptr = llvm::Function*;
using ftptr = llvm::FunctionType*;

class Context {
public:
  llvm::LLVMContext TheContext;
  llvm::IRBuilder<> Builder;
  llvm::Module TheModule;
  std::map<std::string, vptr> NamedValues;

  // optimization
  llvm::legacy::FunctionPassManager TheFPM;

  Context(): TheContext(), Builder(TheContext), TheModule("Main", TheContext), NamedValues(), TheFPM(&TheModule) {
    // TheFPM.add(llvm::createPromoteMemoryToRegisterPass());    // SSA conversion
    TheFPM.add(llvm::createCFGSimplificationPass());    // Dead code elimination
    TheFPM.add(llvm::createSROAPass());
    TheFPM.add(llvm::createLoopSimplifyCFGPass());
    // TheFPM.add(llvm::createConstantPropagationPass());
    TheFPM.add(llvm::createNewGVNPass());    // Global value numbering
    TheFPM.add(llvm::createReassociatePass());
    TheFPM.add(llvm::createPartiallyInlineLibCallsPass());    // Inline standard calls
    TheFPM.add(llvm::createDeadCodeEliminationPass());
    TheFPM.add(llvm::createCFGSimplificationPass());    // Cleanup
    TheFPM.add(llvm::createInstructionCombiningPass());
    TheFPM.add(llvm::createFlattenCFGPass());    // Flatten the control flow graph.

    TheFPM.doInitialization();
  }

  static void err(const char* str) {
    std::cerr << str << std::endl;
  }

  static std::nullptr_t errV(const char* str) {
    err(str);
    return nullptr;
  }

  tptr getDoubleTy() {
    return llvm::Type::getDoubleTy(TheContext);
  }
};

}    // namespace ir

namespace ast {

class Node {
public:
  virtual void display(std::ostream& os) const {
    os << "Node" << std::endl;
  }

  virtual ~Node() = default;

  friend std::ostream& operator<<(std::ostream& os, const Node& obj) {
    obj.display(os);
    return os;
  }
};

using nptr = std::shared_ptr<Node>;

////////////////////////////////////////////////////////////////////////////////
// Expr

class ExprNode: public Node {
public:
  void display(std::ostream& os) const override {
    os << "Expr" << std::endl;
  }

  virtual ir::vptr codegen(ir::Context& ctx) const = 0;
};

using eptr = std::shared_ptr<ExprNode>;

class NumberNode: public ExprNode {
public:
  double val;

  NumberNode(const double& val_): val(val_) {}

  void display(std::ostream& os) const override {
    os << "Number: " << val << std::endl;
  }

  ir::vptr codegen(ir::Context& ctx) const override {
    return llvm::ConstantFP::get(ctx.TheContext, llvm::APFloat(val));
  }
};

class VariableNode: public ExprNode {
public:
  std::string name;

  VariableNode(const std::string& name_): name(name_) {}

  void display(std::ostream& os) const override {
    os << "Variable: " << name << std::endl;
  }

  ir::vptr codegen(ir::Context& ctx) const override {
    auto it = ctx.NamedValues.find(name);
    if (it == ctx.NamedValues.end())
      return ctx.errV("Unknown variable name");
    else
      return it->second;
  }
};

class BinaryNode: public ExprNode {
public:
  char op;
  eptr lhs, rhs;

  BinaryNode(const char& op_, const eptr& lhs_, const eptr& rhs_): op(op_), lhs(lhs_), rhs(rhs_) {}

  void display(std::ostream& os) const override {
    os << "Binary: " << op << std::endl << *lhs << *rhs;
  }

  ir::vptr codegen(ir::Context& ctx) const override {
    ir::vptr L = lhs->codegen(ctx);
    ir::vptr R = rhs->codegen(ctx);
    if (L == nullptr || R == nullptr) return nullptr;

    switch (op) {
      // Arithmetic
      case '+':
        return ctx.Builder.CreateFAdd(L, R, "addtmp");
      case '-':
        return ctx.Builder.CreateFSub(L, R, "subtmp");
      case '*':
        return ctx.Builder.CreateFMul(L, R, "multmp");
      case '/':
        return ctx.Builder.CreateFDiv(L, R, "divtmp");
      // Logical
      case '<':
        L = ctx.Builder.CreateFCmpOLT(L, R, "olttmp");
        return ctx.Builder.CreateUIToFP(L, llvm::Type::getDoubleTy(ctx.TheContext), "booltmp");
      default:
        return ctx.errV("invalid binary operator");
    }
  }
};

class CallNode: public ExprNode {
public:
  std::string callee;
  std::vector<eptr> args;

  CallNode(const std::string& callee_, const std::vector<eptr>& args_): callee(callee_), args(args_) {}

  void display(std::ostream& os) const override {
    os << "Call: " << callee << std::endl;
    for (auto&& p : args) os << *p;
  }

  ir::vptr codegen(ir::Context& ctx) const override {
    llvm::Function* F = ctx.TheModule.getFunction(callee);
    if (F == nullptr) return ctx.errV("Unknown function referenced");

    // Argument mismatch
    if (F->arg_size() != args.size()) return ctx.errV("Incorrect # arguments passed");

    std::vector<ir::vptr> argsv;
    for (auto&& arg : args) {
      auto ptr = arg->codegen(ctx);
      if (ptr == nullptr)
        return nullptr;
      else
        argsv.push_back(ptr);
    }

    return ctx.Builder.CreateCall(F, argsv, "calltmp");
  }
};

////////////////////////////////////////////////////////////////////////////////
// Function

class PrototypeNode: public Node {
public:
  std::string name;
  std::vector<std::string> args;

  PrototypeNode(const std::string& name_, const std::vector<std::string> args_): name(name_), args(args_) {}

  void display(std::ostream& os) const override {
    os << "Prototype: " << name << "(";
    for (auto it = args.begin(); it != args.end(); ++it) {
      if (it != args.begin()) os << ", ";
      os << *it;
    }
    os << " )" << std::endl;
  }

  ir::fptr codegen(ir::Context& ctx) const /* override */ {
    std::vector<ir::tptr> Doubles(args.size(), ctx.getDoubleTy());
    ir::ftptr FT = llvm::FunctionType::get(ctx.getDoubleTy(), Doubles, false);
    ir::fptr F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, name, ctx.TheModule);

    auto it = args.begin();
    for (auto&& arg : F->args()) arg.setName(*(it++));

    return F;
  }
};

using pptr = std::shared_ptr<PrototypeNode>;

class FunctionNode: public Node {
public:
  pptr proto;
  eptr body;

  FunctionNode(const pptr& proto_, const eptr& body_): proto(proto_), body(body_) {}

  void display(std::ostream& os) const override {
    os << "Function: " << *proto << "Function Body: " << std::endl << *body;
  }

  ir::fptr codegen(ir::Context& ctx) const /* override */ {
    ir::fptr F = ctx.TheModule.getFunction(proto->name);
    if (F == nullptr) {    // not extern before
      F = proto->codegen(ctx);
      if (F == nullptr) return nullptr;
    }
    else {
      // already extern, check arguments' name
      auto it = proto->args.begin();
      for (auto&& arg : F->args())
        if (arg.getName() != *(it++)) return ctx.errV("Function arguments are named differently from extern.");
    }

    if (!F->empty()) return ctx.errV("Function cannot be redefined.");

    llvm::BasicBlock* BB = llvm::BasicBlock::Create(ctx.TheContext, "entry", F);
    ctx.Builder.SetInsertPoint(BB);

    ctx.NamedValues.clear();    // NOTE: What?
    for (auto&& arg : F->args()) ctx.NamedValues[arg.getName().str()] = &arg;

    if (ir::vptr ret = body->codegen(ctx)) {
      ctx.Builder.CreateRet(ret);
      llvm::verifyFunction(*F);
      ctx.TheFPM.run(*F);
      return F;
    }

    // Error handle, remove function
    F->eraseFromParent();
    return nullptr;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Root

class RootNode: public Node {
public:
  std::vector<nptr> stats;

  void display(std::ostream& os) const override {
    os << "ROOT:" << std::endl;
    for (auto&& node : stats) node->display(os);
  }
};

using rptr = std::shared_ptr<RootNode>;

};    // namespace ast

namespace lexer {

static const char* get_grammar() {
  static std::string grammar = []() -> std::string {
    std::ifstream fs("src/grammar.peg");
    return {
        std::istreambuf_iterator<char>(fs),
        std::istreambuf_iterator<char>(),
    };
  }();
  return grammar.c_str();
}

class Lexer {
  peg::parser parser;

public:
  Lexer() {
    parser.log = [](size_t row, size_t col, const std::string& msg) {
      std::cerr << row << ':' << col << ' ' << msg << std::endl;
    };

    bool ok = parser.load_grammar(get_grammar());
    assert(ok);

    ////////////////////////////////////////////////////////////////////////////
    // Root
    parser["ROOT"] = [this](const peg::SemanticValues& sv) -> ast::rptr {
      auto ptr = std::make_shared<ast::RootNode>();
      for (auto&& p : sv) {
        ptr->stats.push_back(std::any_cast<ast::nptr>(p));
      }
      return ptr;
    };

    parser["Stat"] = [](const peg::SemanticValues& sv) -> ast::nptr {
      return std::any_cast<ast::nptr>(sv[0]);
    };

    ////////////////////////////////////////////////////////////////////////////
    // Statement

    parser["Definition"] = [](const peg::SemanticValues& sv) -> ast::nptr {
      auto proto = static_pointer_cast<ast::PrototypeNode>(std::any_cast<ast::nptr>(sv[0]));
      auto body = std::any_cast<ast::eptr>(sv[1]);
      auto ptr = std::make_shared<ast::FunctionNode>(proto, body);
      return static_pointer_cast<ast::Node>(ptr);
    };

    parser["External"] = [](const peg::SemanticValues& sv) -> ast::nptr {
      return std::any_cast<ast::nptr>(sv[0]);
    };

    parser["ExprStat"] = [](const peg::SemanticValues& sv) -> ast::nptr {
      // Make an anonymous proto.
      auto proto = std::make_shared<ast::PrototypeNode>("", std::vector<std::string>());
      auto eptr = std::any_cast<ast::eptr>(sv[0]);
      auto func = std::make_shared<ast::FunctionNode>(proto, eptr);
      return static_pointer_cast<ast::Node>(func);
    };

    ////////////////////////////////////////////////////////////////////////////
    // Prototype

    parser["Prototype"] = [](const peg::SemanticValues& sv) -> ast::nptr {
      if (sv.size() == 1) {
        auto name = std::any_cast<std::string>(sv[0]);
        std::vector<std::string> vec = {};
        auto ptr = std::make_shared<ast::PrototypeNode>(name, vec);
        return static_pointer_cast<ast::Node>(ptr);
      }
      else {
        auto name = std::any_cast<std::string>(sv[0]);
        auto vec = std::any_cast<std::vector<std::string>>(sv[1]);
        auto ptr = std::make_shared<ast::PrototypeNode>(name, vec);
        return static_pointer_cast<ast::Node>(ptr);
      }
    };

    parser["ParameterList"] = [](const peg::SemanticValues& sv) -> std::vector<std::string> {
      std::vector<std::string> vec;
      for (auto&& any : sv) vec.push_back(std::any_cast<std::string>(any));
      return vec;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Expr
    parser["Expr"] = [](const peg::SemanticValues& sv) -> ast::eptr {
      return std::any_cast<ast::eptr>(sv[0]);
    };

    auto binary = [](const peg::SemanticValues& sv) -> ast::eptr {
      if (sv.size() == 2) {
        auto ptr = make_shared<ast::BinaryNode>(sv.token()[0],
                                                std::any_cast<ast::eptr>(sv[0]),
                                                std::any_cast<ast::eptr>(sv[1]));
        return static_pointer_cast<ast::ExprNode>(ptr);
      }
      else {
        return std::any_cast<ast::eptr>(sv[0]);
      }
    };

    parser["Additive"] = binary;

    parser["Multitive"] = binary;

    parser["Primary"] = [](const peg::SemanticValues& sv) -> ast::eptr {
      if (sv.size() == 1)
        return std::any_cast<ast::eptr>(sv[0]);
      else
        return std::any_cast<ast::eptr>(sv[1]);
    };

    parser["IdentifierExpr"] = [](const peg::SemanticValues& sv) -> ast::eptr {
      auto id = std::any_cast<std::string>(sv[0]);
      if (sv.size() == 1) {
        auto ptr = std::make_shared<ast::VariableNode>(id);
        return static_pointer_cast<ast::ExprNode>(ptr);
      }
      else {
        auto vec = std::any_cast<std::vector<ast::eptr>>(sv[1]);
        auto ptr = std::make_shared<ast::CallNode>(id, vec);
        return static_pointer_cast<ast::ExprNode>(ptr);
      }
    };

    parser["FunctionCallList"] = [](const peg::SemanticValues& sv) -> std::vector<ast::eptr> {
      if (sv.size())
        return std::any_cast<std::vector<ast::eptr>>(sv[0]);
      else
        return std::vector<ast::eptr>();
    };

    parser["ExprList"] = [](const peg::SemanticValues& sv) -> std::vector<ast::eptr> {
      std::vector<ast::eptr> vec;
      for (auto&& any : sv) vec.push_back(std::any_cast<ast::eptr>(any));
      return vec;
    };

    ////////////////////////////////////////////////////////////////////////////
    // literal
    parser["Number"] = [](const peg::SemanticValues& sv) -> ast::eptr {
      double val = std::stod(sv.token_to_string());
      auto ptr = std::make_shared<ast::NumberNode>(val);
      return static_pointer_cast<ast::ExprNode>(ptr);
    };

    parser["Identifier"] = [](const peg::SemanticValues& sv) -> std::string {
      return sv.token_to_string();
    };
  }

  ast::rptr Parse(const char* repr) {
    parser.enable_packrat_parsing();
    ast::rptr ptr;
    parser.parse(repr, ptr);
    return ptr;
  }
};

};    // namespace lexer

using namespace std;
using namespace lexer;
using namespace ast;
using namespace llvm;

int main() {
  ir::Context ctx;

  auto extern_handler = [&](const shared_ptr<PrototypeNode>& ptr) {
    cout << "function extern detected" << endl;
    auto ir = ptr->codegen(ctx);
  };

  auto define_handler = [&](const shared_ptr<FunctionNode>& ptr) {
    cout << "function define detected" << endl;
    auto ir = ptr->codegen(ctx);
  };

  auto topexp_handler = [&](const shared_ptr<FunctionNode>& ptr) {
    cout << "top expression detected" << endl;
    define_handler(ptr);
  };

  // MainLoop
  while (cin) {
    cout << ">>> " << flush;
    string str;
    getline(cin, str);

    rptr root = Lexer().Parse(str.c_str());
    if (root == nullptr) {
      cout << "root is nullptr" << endl;
      continue;
    }

    for (auto&& nptr : root->stats) {
      if (auto fptr = dynamic_pointer_cast<FunctionNode>(nptr))
        if (fptr->proto->name == "")
          topexp_handler(fptr);
        else
          define_handler(fptr);
      else if (auto pptr = dynamic_pointer_cast<PrototypeNode>(nptr))
        extern_handler(pptr);
      else
        cerr << "Unknown pointer type when dynamic pointer cast" << endl;
    }
  }

  cout << endl << endl << "Module IR:" << endl;
  ctx.TheModule.print(errs(), nullptr);
}
/**
 * Serene programming language.
 *
 *  Copyright (c) 2020 Sameer Rahmani <lxsameer@gnu.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "serene/symbol.hpp"
#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/namespace.hpp"
#include "serene/state.hpp"
#include <assert.h>
#include <fmt/core.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

string Symbol::string_repr() const { return name_; }

string Symbol::dumpAST() const {
  return fmt::format("<Symbol [loc: {} | {}]: {}>",
                     this->location->start.toString(),
                     this->location->end.toString(), this->getName());
}

const string &Symbol::getName() const { return name_; }

Symbol::Symbol(reader::LocationRange loc, const string &name) : name_(name) {
  this->location.reset(new reader::LocationRange(loc));
}

/**
 * `classof` is a enabler static method that belongs to the LLVM RTTI interface
 * `llvm::isa`, `llvm::cast` and `llvm::dyn_cast` use this method.
 */
bool Symbol::classof(const AExpr *expr) {
  return expr->getType() == SereneType::Symbol;
}

Symbol::~Symbol() { EXPR_LOG("Destroying symbol"); }

std::unique_ptr<Symbol> makeSymbol(reader::LocationRange loc,
                                   std::string name) {
  return std::make_unique<Symbol>(loc, name);
}
} // namespace serene

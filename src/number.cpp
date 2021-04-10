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

#include "serene/number.hpp"
#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/namespace.h"
#include "serene/state.hpp"
#include <assert.h>
#include <fmt/core.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <string>

namespace serene {

std::string Number::string_repr() const { return num_; }

std::string Number::dumpAST() const {
  return fmt::format("<Number [loc: {} | {}]: {}>",
                     this->location->start.toString(),
                     this->location->end.toString(), this->num_);
}

Number::Number(reader::LocationRange loc, const std::string &num, bool isNeg,
               bool isFloat)
    : num_(num), isNeg(isNeg), isFloat(isFloat) {
  this->location.reset(new reader::LocationRange(loc));
}
int64_t Number::toI64() {
  // TODO: Handle float case as well
  return std::stoi(num_);
};

/**
 * `classof` is a enabler static method that belongs to the LLVM RTTI interface
 * `llvm::isa`, `llvm::cast` and `llvm::dyn_cast` use this method.
 */
bool Number::classof(const AExpr *expr) {
  return expr->getType() == SereneType::Number;
}

Number::~Number() { EXPR_LOG("Destroying number"); }

std::unique_ptr<Number> makeNumber(reader::LocationRange loc, std::string num,
                                   bool isNeg, bool isFloat) {
  return std::make_unique<Number>(loc, num, isNeg, isFloat);
}

} // namespace serene

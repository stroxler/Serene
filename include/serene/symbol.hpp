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

#ifndef SYMBOL_H
#define SYMBOL_H

#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/reader/location.hpp"
#include "serene/state.hpp"
#include <string>

namespace serene {
class Symbol : public AExpr {
  const std::string name_;

public:
  Symbol(reader::LocationRange loc, const std::string &);
  virtual ~Symbol();

  const std::string &getName() const;

  SereneType getType() const override { return SereneType::Symbol; }
  std::string string_repr() const override;
  std::string dumpAST() const override;

  static bool classof(const AExpr *);
};

std::unique_ptr<Symbol> makeSymbol(reader::LocationRange, std::string);
} // namespace serene

#endif

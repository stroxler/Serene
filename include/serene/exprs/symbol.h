/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SERENE_EXPRS_SYMBOL_H
#define SERENE_EXPRS_SYMBOL_H

#include "serene/context.h"
#include "serene/exprs/expression.h"

#include <llvm/ADT/StringRef.h>

#include <string>

namespace serene {

namespace exprs {

/// This data structure represent the Lisp symbol. Just a symbol
/// in the context of the AST and nothing else.
class Symbol : public Expression {

public:
  std::string name;

  Symbol(reader::LocationRange &loc, llvm::StringRef name)
      : Expression(loc), name(name){};

  Symbol(Symbol &s) : Expression(s.location) { this->name = s.name; }

  ExprType getType() const override;
  std::string toString() const override;

  MaybeNode analyze(SereneContext & /*ctx*/) override;
  void generateIR(serene::Namespace & /*ns*/,
                  mlir::ModuleOp & /*m*/) override{};

  ~Symbol() = default;

  static bool classof(const Expression *e);
};

} // namespace exprs
} // namespace serene

#endif

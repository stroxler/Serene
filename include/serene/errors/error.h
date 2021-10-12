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

#ifndef SERENE_ERRORS_ERROR_H
#define SERENE_ERRORS_ERROR_H

#include "serene/context.h"
#include "serene/errors/constants.h"
#include "serene/errors/traits.h"
#include "serene/reader/location.h"
#include "serene/reader/traits.h"
#include "serene/traits.h"
//#include "serene/exprs/expression.h"
#include <llvm/Support/Error.h>

namespace serene::errors {

/// This enum represent the expression type and **not** the value type.
enum class ErrType {
  Syntax,
  Semantic,
  Compile,
};

/// This data structure represent the Lisp error. This type of expression
/// doesn't show up in the AST but the compiler might rewrite the AST
/// to contains error expressions
class Error
    : public WithTrait<Error, IError, reader::ILocatable, serene::IDebuggable> {
  reader::LocationRange location;
  ErrorVariant *variant;
  std::string message;

public:
  Error(reader::LocationRange &loc, ErrorVariant *err, llvm::StringRef msg)
      : location(loc), variant(err), message(msg){};

  std::string toString() const;
  reader::LocationRange &where();
  ErrorVariant *getVariant();
  std::string getMessage();
  ~Error() = default;
};

}; // namespace serene::errors

#endif

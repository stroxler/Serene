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

#include "serene/errors/constants.h"
#include "serene/errors/traits.h"
#include "serene/reader/traits.h"
#include "serene/traits.h"

#include <serene/export.h>

#include <llvm/ADT/Optional.h>

namespace serene::reader {
class LocationRange;
} // namespace serene::reader

namespace serene::errors {
class Error;

using ErrorPtr = std::shared_ptr<errors::Error>;

// tree? Yupe, Errors can be stackable which makes a vector of them a tree
using ErrorTree      = std::vector<ErrorPtr>;
using OptionalErrors = llvm::Optional<ErrorTree>;

/// This data structure represent the Lisp error. This type of expression
/// doesn't show up in the AST but the compiler might rewrite the AST
/// to contains error expressions
class SERENE_EXPORT Error
    : public WithTrait<Error, IError, reader::ILocatable, serene::IDebuggable> {
  reader::LocationRange location;
  ErrorVariant *variant;
  std::string message;

public:
  Error(reader::LocationRange &loc, ErrorVariant &err, llvm::StringRef msg)
      : location(loc), variant(&err), message(msg){};

  Error(reader::LocationRange &loc, ErrorVariant &err)
      : location(loc), variant(&err){};

  std::string toString() const;
  reader::LocationRange &where();
  ErrorVariant *getVariant();
  std::string &getMessage();
  ~Error() = default;
};

/// Creates a new Error in the give location \p loc, with the given
/// variant \p and an optional message \p msg and retuns a shared ptr
/// to the error. This is the official API to make errors.
ErrorPtr makeError(reader::LocationRange &loc, ErrorVariant &err,
                   llvm::StringRef msg = "msg");

/// Creates a new ErrorTree out of a new Error that creats from the input
/// argument and pass them to `makeError` function.
ErrorTree makeErrorTree(reader::LocationRange &loc, ErrorVariant &err,
                        llvm::StringRef msg = "msg");

}; // namespace serene::errors

#endif

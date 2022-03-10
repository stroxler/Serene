/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef SERENE_ERRORS_H
#define SERENE_ERRORS_H

#include "serene/errors/base.h"
#include "serene/errors/errc.h"
#include "serene/export.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>

namespace serene {
class SereneContext;
} // namespace serene

namespace serene::errors {

/// Create and return a Serene flavored `llvm::Error` by passing the parameters
/// directly to the constructor of type `E`.
///
/// This is the official way of creating error objects in Serene.
template <typename... Args>
SERENE_EXPORT llvm::Error makeError(SereneContext &ctx, ErrorType errtype,
                                    Args &&...args) {
  return llvm::make_error<Error>(ctx, errtype, std::forward<Args>(args)...);
};

/// Returns the messange that the given error \p e is holding. It doesn't cast
/// the error to a concrete error type.
SERENE_EXPORT std::string getMessage(const llvm::Error &e);

SERENE_EXPORT const ErrorVariant *getVariant(ErrorType t);
} // namespace serene::errors

#endif

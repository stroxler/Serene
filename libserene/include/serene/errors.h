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

#define GET_CLASS_DEFS
#include "serene/errors/errs.h.inc"

#include <llvm/Support/Error.h>

namespace serene::errors {

/// Create and return a Serene flavored `llvm::Error` by passing the parameters
/// directly to the constructor of type `E`.
///
/// This is the official way of creating error objects in Serene.
template <typename E, typename... Args>
SERENE_EXPORT llvm::Error makeError(Args &&...args) {
  return llvm::make_error<E>(std::forward<Args>(args)...);
};

SERENE_EXPORT std::string getMessage(const llvm::Error &e);
} // namespace serene::errors

#endif

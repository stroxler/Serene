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

#include "serene/errors.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>

namespace serene::errors {

// We need this to make Error class a llvm::Error friendy implementation
char SereneError::ID;

std::string getMessage(const llvm::Error &e) {
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << e;
  return os.str();
};

const ErrorVariant *getVariant(ErrorType t) {
  if ((0 <= (int)t) && (t < NUMBER_OF_ERRORS)) {
    return &errorVariants[t];
  }
  return nullptr;
};
} // namespace serene::errors

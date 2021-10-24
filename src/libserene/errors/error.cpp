/*
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

#include "serene/errors/error.h"

#include <llvm/Support/FormatVariadic.h>

#include <memory>

namespace serene {
namespace errors {

std::string Error::toString() const {
  return llvm::formatv("<Error E{0}: {1}>", this->variant->id, this->message);
}

reader::LocationRange &Error::where() { return this->location; };

ErrorVariant *Error::getVariant() { return this->variant; }

std::string &Error::getMessage() { return this->message; }

ErrorPtr makeError(reader::LocationRange &loc, ErrorVariant &err,
                   llvm::StringRef msg) {
  return std::make_shared<Error>(loc, err, msg);
};

ErrorTree makeErrorTree(reader::LocationRange &loc, ErrorVariant &err,
                        llvm::StringRef msg) {
  std::vector errs{makeError(loc, err, msg)};
  return errs;
};

} // namespace errors
} // namespace serene

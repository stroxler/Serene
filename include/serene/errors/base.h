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

#ifndef SERENE_ERRORS_BASE_H
#define SERENE_ERRORS_BASE_H

#include "serene/reader/location.h"

#include <system_error>

#include <llvm/Support/Error.h>

namespace serene::errors {

template <typename T>
class SereneError : public llvm::ErrorInfo<T> {
  reader::LocationRange location;
  std::string msg;

public:
  static const int ID = -1;

  void log(llvm::raw_ostream &os) const { os << msg; };
  std::string message() const override { return msg; };
  std::string &getTitle() const override { return T::title; };
  std::string &getDesc() const override { return T::description; };
  std::error_code convertToErrorCode() const { return std::error_code(); };

  SereneError(reader::LocationRange &loc, std::string &msg)
      : location(loc), msg(msg){};

  reader::LocationRange &where() { return location; };

  static const void *classID() { return &T::ID; }

  bool isA(const void *const id) const override {
    // the check with -1 is a shortcut for us to figure
    // out whether we're dealing with an Serene error or
    // LLVM error
    return *(const int *)id == -1 || id == classID() ||
           llvm::ErrorInfoBase::isA(id);
  }
};

}; // namespace serene::errors
#endif

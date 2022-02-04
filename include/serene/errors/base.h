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
#include <llvm/Support/FormatVariadic.h>

namespace serene::errors {

struct ErrorVariant {
  int id;
  const std::string title;
  const std::string desc;
  const std::string help;

  static ErrorVariant make(int id, const char *t, const char *d,
                           const char *h) {
    return ErrorVariant(id, t, d, h);
  };

private:
  ErrorVariant(int id, const char *t, const char *d, const char *h)
      : id(id), title(t), desc(d), help(h){};
};

class SereneError : public llvm::ErrorInfoBase {
  reader::LocationRange location;
  std::string msg;

public:
  static const int ID = -1;

  void log(llvm::raw_ostream &os) const override { os << msg; };
  std::string message() const override { return msg; };

  std::error_code convertToErrorCode() const override {
    return std::error_code();
  };

  SereneError(reader::LocationRange &loc, std::string &msg)
      : location(loc), msg(msg){};

  SereneError(reader::LocationRange &loc, const char *msg)
      : location(loc), msg(msg){};

  SereneError(reader::LocationRange &loc, llvm::StringRef msg)
      : location(loc), msg(msg.str()){};

  SereneError(reader::LocationRange &loc) : location(loc){};

  SereneError(SereneError &e) = delete;

  reader::LocationRange &where() { return location; };

  static const void *classID() { return &ID; }

  bool isA(const void *const id) const override {
    return id == classID() || llvm::ErrorInfoBase::isA(id);
  }

  ~SereneError() = default;
};

}; // namespace serene::errors
#endif

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

#ifndef TEST_HEALPERS_H
#define TEST_HEALPERS_H

#include "serene/reader/location.h"

// *IMPORTANT NOTE:* The `READ` macro is just a quick way to eliminate
//                   the overhead of writing the same function signature
//                   over and over again. Nothing special about it.
#define READ(input) reader::read(*ctx, input, "user", llvm::None)

#define CHECK_ERR_MSG(e, s) CHECK(serene::errors::getMessage(e) == s)
// `llvm::Error`s has to be checked in the same scope. This macro makes
// the check easy while were testing the other aspects of the error.
// `t` is the concrete error type and `e` is the error instance.
#define CHECK_ERR(t, e)                                                        \
  auto unhandled =                                                             \
      llvm::handleErrors(e, [&](const Error &x) { CHECK(x.errorType == t); }); \
  CHECK(!unhandled);

namespace serene {

reader::LocationRange *dummyLocation() {
  reader::Location start("serene.test.ns");
  reader::Location end("serene.test.ns");

  constexpr const int line1 = 2;
  constexpr const int line2 = 3;
  constexpr const int col1  = 20;
  constexpr const int col2  = 30;

  start.line = line1;
  start.col  = col1;
  end.line   = line2;
  end.col    = col2;

  return new reader::LocationRange(start, end);
};

} // namespace serene

#endif

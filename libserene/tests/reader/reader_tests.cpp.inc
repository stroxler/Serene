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

#ifndef SERENE_TEST_READER_H
#define SERENE_TEST_READER_H

#include "serene/reader/reader.h"

#include <catch2/catch_test_macros.hpp>

// *IMPORTANT NOTE:* The `READ` macro is just a quick way to eliminate
//                   the overhead of writing the same function signature
//                   over and over again. Nothing special about it.
#define READ(input) reader::read(*ctx, input, "user", llvm::None)

namespace serene {
namespace reader {

TEST_CASE("Read numbers", "[reader]") {
  auto ctx      = makeSereneContext();
  auto maybeAst = READ("3");

  if (!maybeAst) {
    FAIL();
  }

  auto ast = *maybeAst;
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<Number 3>");

  maybeAst = READ("-34");

  if (!maybeAst) {
    FAIL();
  }

  ast = *maybeAst;
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<Number -34>");

  maybeAst = READ("-3.5434");

  if (!maybeAst) {
    FAIL();
  }

  ast = *maybeAst;
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<Number -3.5434>");

  maybeAst = READ("444323 2123 123123");

  if (!maybeAst) {
    FAIL();
  }

  ast = *maybeAst;
  REQUIRE(ast.size() == 3);
  CHECK(ast.front()->toString() == "<Number 444323>");
  CHECK(ast[1]->toString() == "<Number 2123>");
  CHECK(ast[2]->toString() == "<Number 123123>");
};

TEST_CASE("Read Lists and Symbols", "[reader]") {
  auto ctx      = makeSereneContext();
  auto maybeAst = READ("(x 1)");

  if (!maybeAst) {
    FAIL();
  }

  auto ast = *maybeAst;
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol user/x> <Number 1>>");

  maybeAst = READ("(x (y (z)))");

  if (!maybeAst) {
    FAIL();
  }

  ast = *maybeAst;
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol user/x> <List  <Symbol "
                                   "user/y> <List  <Symbol user/z>>>>");

  maybeAst = READ("(x \n y)");

  if (!maybeAst) {
    FAIL();
  }

  ast = *maybeAst;
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol user/x> <Symbol user/y>>");
};
} // namespace reader
} // namespace serene
#endif

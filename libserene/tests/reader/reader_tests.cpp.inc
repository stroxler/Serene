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

#include "serene/reader/reader.h"

#include "../test_helpers.cpp.inc"
#include <catch2/catch_all.hpp>

namespace serene {
namespace reader {

TEST_CASE("Read numbers", "[reader]") {
  auto maybeAst = reader::read("3");

  if (!maybeAst) {
    FAIL();
  }

  auto ast = std::move(maybeAst).getValue();
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<Number 3>");

  maybeAst = reader::read("-34");

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<Number -34>");

  maybeAst = reader::read("-3.5434");

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<Number -3.5434>");

  maybeAst = reader::read("444323 2123 123123");

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE(ast.size() == 3);
  CHECK(ast.front()->toString() == "<Number 444323>");
  CHECK(ast[1]->toString() == "<Number 2123>");
  CHECK(ast[2]->toString() == "<Number 123123>");
};
TEST_CASE("Read Lists and Symbols", "[reader]") {
  auto maybeAst = reader::read("(x 1)");

  if (!maybeAst) {
    FAIL();
  }

  auto ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol x> <Number 1>>");

  maybeAst = reader::read("(x (y (z)))");

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() ==
        "<List  <Symbol x> <List  <Symbol y> <List  <Symbol z>>>>");

  maybeAst = reader::read("(x \n y)");

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol x> <Symbol y>>");
};
} // namespace reader
} // namespace serene

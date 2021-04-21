/* -*- C++ -*-
 * Serene programming language.
 *
 *  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "../test_helpers.cpp.inc"
#include "serene/reader/reader.h"
#include <catch2/catch.hpp>

namespace serene {
namespace reader {

TEST_CASE("Read numbers", "[reader]") {
  auto r = new reader::Reader("3");
  auto maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  auto ast = std::move(maybeAst).getValue();
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() ==  "<Number 3>");

  r->setInput("-34");
  maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() ==  "<Number -34>");

  r->setInput("-3.5434");
  maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() ==  "<Number -3.5434>");

  r->setInput("444323 2123 123123");
  maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE(ast.size() == 3);
  CHECK(ast.front()->toString() ==  "<Number 444323>");
  CHECK(ast[1]->toString() == "<Number 2123>" );
  CHECK(ast[2]->toString() == "<Number 123123>");

  delete r;
};
TEST_CASE("Read Lists and Symbols", "[reader]") {
  auto r = new reader::Reader("(x 1)");

  auto maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  auto ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() ==  "<List  <Symbol x> <Number 1>>");

  r->setInput("(x (y (z)))");
  maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol x> <List  <Symbol y> <List  <Symbol z>>>>");

  r->setInput("(x \n y)");
  maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  ast = std::move(maybeAst.getValue());
  REQUIRE_FALSE(ast.empty());
  CHECK(ast.front()->toString() == "<List  <Symbol x> <Symbol y>>");

  delete r;
};
} // namespace reader
} // namespace serene

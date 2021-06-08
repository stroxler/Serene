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

#include "./test_helpers.cpp.inc"
#include "serene/context.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"
#include "serene/reader/reader.h"
#include <catch2/catch.hpp>

namespace serene {

TEST_CASE("Namespace tests", "[namespace]") {
  auto ctx = makeSereneContext();
  auto userNs =
      makeNamespace(*ctx, "user", llvm::Optional<llvm::StringRef>("/some/file"));

  auto r = new reader::Reader("(x 1) (def b a)");

  auto maybeAst = r->read();

  if (!maybeAst) {
    FAIL();
  }

  auto result = userNs->setTree(maybeAst.getValue());

  REQUIRE(result.succeeded());

  REQUIRE_FALSE(userNs->getTree().empty());
  CHECK(exprs::astToString(&userNs->getTree()) ==
        "<List  <Symbol x> <Number 1>> <List  <Symbol def> <Symbol b> <Symbol "
        "a>>");
  delete r;
};

} // namespace serene

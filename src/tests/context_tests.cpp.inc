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
#include "serene/namespace.h"
#include <catch2/catch.hpp>

namespace serene {

TEST_CASE("Context tests", "[context]") {
  auto ctx = makeSereneContext();
  auto ns = ctx->getNS("blah");

  REQUIRE_FALSE(ns);

  auto userNs = makeNamespace(*ctx, "user", llvm::Optional<llvm::StringRef>("/some/file"));

  CHECK(userNs->name == "user");
  REQUIRE(userNs->filename);
  CHECK(userNs->filename.getValue() == "/some/file");

  ns = ctx->getNS("user");

  REQUIRE(ns);
  CHECK(ns->name == userNs->name);


  /// Creating new ns with the same name overrides the old one
  auto userNs1 = makeNamespace(*ctx, "user", llvm::Optional<llvm::StringRef>("/some/other/file"));

  ns = ctx->getNS("user");

  REQUIRE(ns);
  CHECK(ns->name == userNs1->name);
  REQUIRE(ns->filename);
  CHECK(ns->filename.getValue() == "/some/other/file");

};

TEST_CASE("Get and Set current namespace", "[context]") {
  auto ctx = makeSereneContext();
  auto userNs = makeNamespace(*ctx, "user", llvm::Optional<llvm::StringRef>("/some/file"));

  auto isSet = ctx->setCurrentNS("user");

  REQUIRE(isSet);
  CHECK(ctx->getCurrentNS() == userNs);

  isSet = ctx->setCurrentNS("user1");
  REQUIRE_FALSE(isSet);
  CHECK(ctx->getCurrentNS() == userNs);
};
} // namespace serene

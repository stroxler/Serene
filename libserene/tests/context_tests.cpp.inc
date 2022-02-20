/* -*- C++ -*-
 * Serene programming language.
 *
 *  Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

#include "serene/context.h"
#include "serene/namespace.h"
#include "serene/reader/location.h"
#include "serene/serene.h"

#include "./test_helpers.cpp.inc"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <llvm/ADT/None.h>

namespace serene {

TEST_CASE("Compiler options", "[context]") {
  auto opts    = Options();
  opts.JITLazy = !opts.JITLazy;
  opts.JITenableGDBNotificationListener =
      !opts.JITenableGDBNotificationListener;

  auto ctx = makeSereneContext(opts);

  CHECK(ctx->opts.JITLazy == opts.JITLazy);
  CHECK(ctx->opts.JITenableGDBNotificationListener ==
        opts.JITenableGDBNotificationListener);
};

TEST_CASE("makeNamespace & getNS", "[context]") {
  auto ctx = makeSereneContext();
  auto *ns = ctx->getNS("blah");

  REQUIRE_FALSE(ns);

  auto userNs =
      ctx->makeNamespace("user", llvm::Optional<llvm::StringRef>("/some/file"));

  CHECK(userNs->name == "user");
  REQUIRE(userNs->filename);
  CHECK(userNs->filename.getValue() == "/some/file");

  ns = ctx->getNS("user");

  REQUIRE(ns);
  CHECK(ns->name == userNs->name);

  /// Creating new ns with the same name overrides the old one
  auto userNs1 = ctx->makeNamespace(
      "user", llvm::Optional<llvm::StringRef>("/some/other/file"));

  ns = ctx->getNS("user");

  REQUIRE(ns);
  CHECK(ns->name == userNs1->name);
  REQUIRE(ns->filename);
  CHECK(ns->filename.getValue() == "/some/other/file");
};

TEST_CASE(
    "withCurrentNS run a function with the given namespace as the current NS",
    "[context]") {
  auto ctx = makeSereneContext();
  auto userNs =
      ctx->makeNamespace("user", llvm::Optional<llvm::StringRef>("/some/file"));

  ctx->withCurrentNS<int>("user", [&]() -> int {
    CHECK(ctx->getCurrentNS().name == userNs->name);
    return 0;
  });

  // Checking the `void` type specialization
  ctx->withCurrentNS<void>("user", [&]() -> void {
    CHECK(ctx->getCurrentNS().name == userNs->name);
  });

  CHECK(ctx->getCurrentNS().name == DEFAULT_NS_NAME);
};

TEST_CASE("getSharedPtrToNS returns a shared ptr to the NS", "[context]") {
  auto ctx = makeSereneContext();
  auto userNs =
      ctx->makeNamespace("user", llvm::Optional<llvm::StringRef>("/some/file"));

  auto userNs1 = ctx->makeNamespace(
      "user1", llvm::Optional<llvm::StringRef>("/some/file1"));

  CHECK(ctx->getCurrentNS().name == DEFAULT_NS_NAME);

  NSPtr shouldBeUser1 = ctx->getSharedPtrToNS("user");

  REQUIRE(shouldBeUser1);
  CHECK(shouldBeUser1->name == userNs->name);
};

TEST_CASE("Compilation phase", "[context]") {
  auto ctx = makeSereneContext();
  auto cp  = CompilationPhase::O3;

  ctx->setOperationPhase(cp);

  CHECK(ctx->getTargetPhase() == cp);
  CHECK(ctx->getOptimizatioLevel() == 3);

  // Anything below 0 is 0
  cp = CompilationPhase::MLIR;
  ctx->setOperationPhase(cp);
  CHECK(ctx->getOptimizatioLevel() == 0);
};

TEST_CASE("makeNamespace", "[context]") {
  auto ctx = makeSereneContext();

  auto ns = ctx->makeNamespace("example.ns", llvm::None);

  // Namespace has to be empty
  CHECK(ns->name == "example.ns");
  CHECK(ns->getTree().empty());
  CHECK(ns->filename == llvm::None);
};

TEST_CASE("context and jit", "[context]") {
  auto ctx = makeSereneContext();

  auto ns = ctx->makeNamespace("example.ns", llvm::None);

  REQUIRE(ctx->jit);

  // no JITDylib should be defined for an empty NS
  CHECK(ctx->getNumberOfJITDylibs(*ns) == 0);

  SECTION("JITDylib management") {
    auto unknown = reader::LocationRange::UnknownLocation(DEFAULT_NS_NAME);
    REQUIRE(!ctx->jit->addNS(*ns, unknown));

    CHECK(ctx->getNumberOfJITDylibs(*ns) == 1);

    REQUIRE(ctx->getLatestJITDylib(*ns));
    CHECK(ctx->getLatestJITDylib(*ns)->getName() == "example.ns#1");
  };
};

} // namespace serene

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

#include "serene/environment.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/symbol.h"
#include "serene/reader/location.h"
#include "serene/serene.h"

#include "./test_helpers.cpp.inc"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <llvm/Support/Casting.h>

namespace serene {

TEST_CASE("Environment tests", "[environment]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());

  exprs::Node sym = exprs::make<exprs::Symbol>(*range, "example", "ns");

  Environment<exprs::Node> e;
  llvm::Optional<exprs::Node> result = e.lookup("a");

  REQUIRE_FALSE(result.hasValue());

  auto status = e.insert_symbol("a", sym);

  REQUIRE(status.succeeded());

  result = e.lookup("a");

  REQUIRE(result.hasValue());
  CHECK(result.getValue() == sym);

  auto *fetchedSym = llvm::dyn_cast<exprs::Symbol>(result.getValue().get());
  REQUIRE(fetchedSym != nullptr);
  CHECK(fetchedSym->name == "example");
  CHECK(fetchedSym->nsName == "ns");

  SECTION("Testing the environment copy") {
    Environment<exprs::Node> e1(&e);

    result = e1.lookup("b");
    REQUIRE_FALSE(result.hasValue());

    // It should lookup the value in the parent environment
    result = e1.lookup("a");
    REQUIRE(result.hasValue());
    CHECK(result.getValue() == sym);
  };
};

TEST_CASE("Test generic Environment usage", "[environment]") {
  Environment<char> env;

  auto result = env.lookup("blah");
  REQUIRE_FALSE(result);

  auto status = env.insert_symbol("blah", 'A');
  REQUIRE(status.succeeded());

  result = env.lookup("blah");
  REQUIRE(result);
  CHECK(*result == 'A');

  result = env.lookup("blah");
  REQUIRE(result);
  CHECK(*result == 'A');

  // Test the overwrite functionality
  status = env.insert_symbol("blah", 'B');
  REQUIRE(status.succeeded());

  result = env.lookup("blah");
  REQUIRE(result);
  CHECK(*result == 'B');
}
} // namespace serene

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

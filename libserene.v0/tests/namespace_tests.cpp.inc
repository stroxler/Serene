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

#include "serene/context.h"
#include "serene/exprs/expression.h"
#include "serene/namespace.h"
#include "serene/reader/reader.h"

#include "./test_helpers.cpp.inc"
#include <catch2/catch_all.hpp>

namespace serene {

TEST_CASE("Namespace tests", "[namespace]") {
  auto ctx    = makeSereneContext();
  auto userNs = makeNamespace(*ctx, "user",
                              llvm::Optional<llvm::StringRef>("/some/file"));

  auto maybeAst = reader::read("(x 1) (def b a)");

  if (!maybeAst) {
    FAIL();
  }

  auto result = userNs->setTree(maybeAst.getValue());

  REQUIRE(result.succeeded());

  REQUIRE_FALSE(userNs->getTree().empty());
  CHECK(exprs::astToString(&userNs->getTree()) ==
        "<List  <Symbol x> <Number 1>> <List  <Symbol def> <Symbol b> <Symbol "
        "a>>");
};

} // namespace serene

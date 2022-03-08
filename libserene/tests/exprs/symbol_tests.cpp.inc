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

#include "serene/exprs/expression.h"
#include "serene/exprs/symbol.h"

#include "../test_helpers.cpp.inc"

namespace serene {
namespace exprs {

TEST_CASE("Public Symbol API", "[expression]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());
  auto sym = make<Symbol>(*range.get(), "example");

  REQUIRE(sym->getType() == ExprType::Symbol);
  CHECK(sym->toString() == "<Symbol example>");
};
} // namespace exprs
} // namespace serene

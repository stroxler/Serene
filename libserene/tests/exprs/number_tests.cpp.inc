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

#include "serene/exprs/number.h"

#include "../test_helpers.cpp.inc"
#include <catch2/catch_all.hpp>

namespace serene {
namespace exprs {
TEST_CASE("Number Expression", "[expression]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());

  auto num1 = makeAndCast<Number>(*range.get(), "3", false, false);
  auto num2 = makeAndCast<Number>(*range.get(), "3.4", false, true);
  // Hence the `isNeg` being true. We need to provide the sign as the input
  // anyway
  auto num3 = makeAndCast<Number>(*range.get(), "3", true, false);
  auto num4 = makeAndCast<Number>(*range.get(), "-3", true, false);

  CHECK(num1->toString() == "<Number 3>");
  CHECK(num2->toString() == "<Number 3.4>");
  CHECK(num3->toString() == "<Number 3>");
  CHECK(num4->toString() == "<Number -3>");
};

} // namespace exprs
} // namespace serene

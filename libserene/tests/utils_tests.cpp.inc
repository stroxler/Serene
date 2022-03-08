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
#include "serene/utils.h"

#include "./test_helpers.cpp.inc"

namespace serene {
TEST_CASE("Result Type", "[utils]") {
  auto r = Result<int>::success(4);

  REQUIRE(r == true);
  CHECK(r.getValue() == 4);

  auto r1 = Result<int, char>::error('c');

  REQUIRE_FALSE(r1);
  CHECK(r1.getError() == 'c');
};
} // namespace serene

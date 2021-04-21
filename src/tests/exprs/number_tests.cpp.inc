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
#include "serene/exprs/number.h"
#include <catch2/catch.hpp>

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
  CHECK(num3->toString() ==  "<Number 3>");
  CHECK(num4->toString() ==  "<Number -3>");
};

} // namespace exprs
} // namespace serene

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

#ifndef TEST_HEALPERS_H
#define TEST_HEALPERS_H

#include "serene/reader/location.h"

#include <catch2/catch_all.hpp>

namespace serene {

reader::LocationRange *dummyLocation() {
  reader::Location start("serene.test.ns");
  reader::Location end("serene.test.ns");

  constexpr const int line1 = 2;
  constexpr const int line2 = 3;
  constexpr const int col1  = 20;
  constexpr const int col2  = 30;

  start.line = line1;
  start.col  = col1;
  end.line   = line2;
  end.col    = col2;

  return new reader::LocationRange(start, end);
};

} // namespace serene

#endif

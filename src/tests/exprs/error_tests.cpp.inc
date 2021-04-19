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
#include "serene/exprs/symbol.h"
#include "serene/exprs/error.h"
#include "llvm/Support/Casting.h"
#include <catch2/catch.hpp>

namespace serene {
namespace exprs {

TEST_CASE("Error Expression", "[expression]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());

  node sym = make<Symbol>(*range.get(), llvm::StringRef("example"));
  node err = make<Error>(ErrType::Semantic, sym, "Something Failed");

  REQUIRE(err->getType() == ExprType::Error);
  CHECK(err->toString() == "<Error [loc: 2:20:40 | 3:30:80]: Something Failed>");

  auto error = llvm::dyn_cast<Error>(err.get());
  CHECK(error->errorType == ErrType::Semantic);
  CHECK(error->target == sym);
};

} // namespace exprs
} // namespace serene

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
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"

namespace serene {
namespace exprs {

TEST_CASE("List Expression", "[expression]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());

  auto sym = make<Symbol>(*range.get(), llvm::StringRef("example"));
  auto list = make<List>(*range.get());
  auto list2 = make<List>(*range.get(), list);
  auto list3 = make<List>(*range.get(), llvm::ArrayRef<node>{list, list2, sym});

  REQUIRE(list->getType() == ExprType::List);
  CHECK(list->toString() == "<List [loc: 2:20:40 | 3:30:80]: ->");

  CHECK(list2->toString() ==
        "<List [loc: 2:20:40 | 3:30:80]:  <List [loc: 2:20:40 | 3:30:80]: ->>");
  CHECK(list3->toString() ==
        "<List [loc: 2:20:40 | 3:30:80]:  <List [loc: 2:20:40 | 3:30:80]: -> "
        "<List [loc: 2:20:40 | 3:30:80]:  <List [loc: 2:20:40 | 3:30:80]: "
        "->> <Symbol [loc: 2:20:40 | 3:30:80]: example>>");

  auto l = llvm::dyn_cast<List>(list);

  l.append(sym);

  REQUIRE(list->getType() == ExprType::List);
  CHECK(list->toString() == "<List [loc: 2:20:40 | 3:30:80]:  <Symbol [loc: "
                            "2:20:40 | 3:30:80]: example>>");
};

} // namespace exprs
} // namespace serene

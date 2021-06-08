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
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "serene/namespace.h"
#include "serene/reader/reader.h"
#include "serene/reader/semantics.h"
#include <catch2/catch.hpp>

namespace serene {
namespace exprs {

TEST_CASE("List Expression", "[expression]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());

  Node sym = make<Symbol>(*range.get(), llvm::StringRef("example"));
  Node sym1 = make<Symbol>(*range.get(), llvm::StringRef("example1"));

  Node list = make<List>(*range.get());

  REQUIRE(list->getType() == ExprType::List);
  CHECK(list->toString() == "<List ->");

  Node list2 = make<List>(*range.get(), list);
  CHECK(list2->toString() == "<List  <List ->>");

  Ast elements;
  elements.push_back(list);
  elements.push_back(list2);
  elements.push_back(sym);

  auto list3 = make<List>(*range.get(), elements);

  CHECK(list3->toString() ==
        "<List  <List -> <List  <List ->> <Symbol example>>");

  auto l = llvm::dyn_cast<List>(list.get());

  l->append(sym1);

  REQUIRE(list->getType() == ExprType::List);
  CHECK(list->toString() == "<List  <Symbol example1>>");

  l->append(sym);
  REQUIRE(l->count() == 2);

  auto expr = l->at(1);
  REQUIRE(expr.hasValue());
  CHECK(expr.getValue()->toString() == "<Symbol example>");

  expr = l->at(2);
  REQUIRE_FALSE(expr.hasValue());

  for (auto x : *l) {
    CHECK(x->getType() == ExprType::Symbol);
  }
};

TEST_CASE("List semantic analysis of 'def'", "[semantic]") {
  auto ctx = makeSereneContext();
  auto ns = makeNamespace(*ctx, "user", llvm::None);
  auto ast = reader::read("(def (a) b)");
  auto afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE_FALSE(afterAst);
  // Fetch the first error
  CHECK(afterAst.getError()[0]->toString() == "<Error E1: >");

  ast = reader::read("(def a)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE_FALSE(afterAst);
  CHECK(afterAst.getError()[0]->toString() == "<Error E2: Expected 3 got 2>");

  ast = reader::read("(def a b c)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE_FALSE(afterAst);
  CHECK(afterAst.getError()[0]->toString() == "<Error E2: Expected 3 got 4>");

  ast = reader::read("(def a b)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) == "<Def a -> <Symbol b>>");

  ast = reader::read("(def a (fn () a))");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) ==
        "<Def a -> <Fn a <List -> to <Symbol a>>>");
}

TEST_CASE("List semantic analysis for 'fn'", "[semantic]") {
  auto ctx = makeSereneContext();
  auto ns = makeNamespace(*ctx, "user", llvm::None);
  auto ast = reader::read("(fn)");
  auto afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE_FALSE(afterAst);
  REQUIRE(afterAst.getError().size() == 1);
  CHECK(afterAst.getError()[0]->toString() ==
        "<Error E3: The argument list is mandatory.>");

  ast = reader::read("(fn ())");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) == "<Fn ___fn___0 <List -> to <>>");

  ast = reader::read("(fn (a b c) a a a)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) ==
        "<Fn ___fn___1 <List  <Symbol a> <Symbol b> <Symbol c>> to <Symbol a> <Symbol a> <Symbol a>>"
        );

  ast = reader::read("(fn () a b)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) ==
        "<Fn ___fn___2 <List -> to <Symbol a> <Symbol b>>");

  ast = reader::read("(fn (x) (fn (y) x) z)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) ==
        "<Fn ___fn___4 <List  <Symbol x>> to <Fn ___fn___3 <List  <Symbol y>> to <Symbol x>> <Symbol z>>"
  );

  ast = reader::read("(fn (x) (def a b) (def b c))");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) ==
        "<Fn ___fn___5 <List  <Symbol x>> to <Def a -> <Symbol b>> <Def b -> <Symbol c>>>");
}

TEST_CASE("Complex semantic analysis", "[semantic]") {
  auto ctx = makeSereneContext();
  auto ns = makeNamespace(*ctx, "user", llvm::None);
  auto ast =
      reader::read("(def a (fn (x) x))\n((def b (fn (x) (fn (y) y))))\n\n");
  auto afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);
  CHECK(astToString(&afterAst.getValue()) ==
        "<Def a -> <Fn a <List  <Symbol x>> to <Symbol x>>> <Call <Def b -> <Fn b <List  <Symbol x>> to <Fn ___fn___1 <List  <Symbol y>> to <Symbol y>>>> >");

  ctx = makeSereneContext();
  ns = makeNamespace(*ctx, "user", llvm::None);
  ast = reader::read("((a b))");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE_FALSE(afterAst);
  auto errs = afterAst.getError();
  CHECK(errs[0]->toString() == "<Error E5: Can't resolve the symbol 'a'>");

  ctx = makeSereneContext();
  ns = makeNamespace(*ctx, "user", llvm::None);
  ast = reader::read("(def a (fn (x) x)) (a b)");
  afterAst = reader::analyze(*ctx, ast.getValue());
  REQUIRE(afterAst);

  CHECK(astToString(&afterAst.getValue()) ==
        "<Def a -> <Fn a <List  <Symbol x>> to <Symbol x>>> <Call <Fn a <List  <Symbol x>> to <Symbol x>> <Symbol b>>");
}
} // namespace exprs
} // namespace serene

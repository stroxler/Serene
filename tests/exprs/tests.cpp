#include "catch2/catch.hpp"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"
#include "llvm/ADT/ArrayRef.h"
#include <iostream>

namespace serene {
namespace exprs {

reader::LocationRange *dummyLocation() {
  reader::Location start;
  reader::Location end;

  start.line = 2;
  start.col = 20;
  start.pos = 40;

  end.line = 3;
  end.col = 30;
  end.pos = 80;

  return new reader::LocationRange(start, end);
}

TEST_CASE("Symbol Expression", "[expression]") {
  // return new reader::LocationRange(start, end);
  std::unique_ptr<reader::LocationRange> range(dummyLocation());
  Expression sym = Expression::make<Symbol>(*range.get(), "example");

  REQUIRE(sym.getType() == ExprType::Symbol);
  REQUIRE(sym.toString() == "<Symbol [loc: 2:20:40 | 3:30:80]: example>");
};

TEST_CASE("List Expression", "[expression]") {
  // return new reader::LocationRange(start, end);
  std::unique_ptr<reader::LocationRange> range(dummyLocation());
  Expression sym = Expression::make<Symbol>(*range.get(), "example");

  Expression list = Expression::make<List>(*range.get());
  Expression list2 = Expression::make<List>(*range.get(), list);
  Expression list3 = Expression::make<List>(
      *range.get(), llvm::ArrayRef<Expression>{list, list2, sym});

  REQUIRE(list.toString() == "<List [loc: 2:20:40 | 3:30:80]: ->");
  REQUIRE(list.getType() == ExprType::List);

  REQUIRE(
      list2.toString() ==
      "<List [loc: 2:20:40 | 3:30:80]:  <List [loc: 2:20:40 | 3:30:80]: ->>");
  REQUIRE(list3.toString() ==
          "<List [loc: 2:20:40 | 3:30:80]:  <List [loc: 2:20:40 | 3:30:80]: -> "
          "<List [loc: 2:20:40 | 3:30:80]:  <List [loc: 2:20:40 | 3:30:80]: "
          "->> <Symbol [loc: 2:20:40 | 3:30:80]: example>>");

  auto l = list.to<List>();

  l->get()->elements.push_back(sym);

  REQUIRE(list.getType() == ExprType::List);
  REQUIRE(list.toString() == "<List [loc: 2:20:40 | 3:30:80]: ->");
};

} // namespace exprs
} // namespace serene

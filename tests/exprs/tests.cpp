#include "catch2/catch.hpp"
#include "serene/exprs/expression.h"
#include "serene/exprs/list.h"
#include "serene/exprs/symbol.h"

namespace serene {
namespace exprs {
TEST_CASE("Expressions", "[cabc]") {
  auto list = Expression::make<List>();
  REQUIRE(list.getType() == ExprType::List);


}
} // namespace exprs
} // namespace serene

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

TEST_CASE("Test stuff", "[abc]") { REQUIRE(1 == 2); }

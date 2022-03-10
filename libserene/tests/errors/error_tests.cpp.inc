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

#ifndef SERENE_TEST_ERRORS_H
#define SERENE_TEST_ERRORS_H

#include "serene/context.h"
#include "serene/errors.h"

#include "../test_helpers.cpp.inc"
#include <catch2/catch_test_macros.hpp>

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>

namespace serene {
namespace errors {

TEST_CASE("Serene Error construction", "[errors]") {

  std::unique_ptr<reader::LocationRange> range(dummyLocation());
  auto ctx        = makeSereneContext();
  llvm::Error err = makeError(*ctx, PassFailureError, *range, "test error");

  auto unhandled =
      llvm::handleErrors(std::move(err), [&](const SereneError &e) {
        REQUIRE(e.message() == "test error");
        const auto *v = getVariant(e.errorType);
        REQUIRE(v != nullptr);
        CHECK(v->title == "PassFailureError");
        CHECK(v->desc == "Pass Failure.");
        CHECK(v->help.empty());
      });

  CHECK(!unhandled);
}

TEST_CASE("getMessage function", "[errors]") {
  std::unique_ptr<reader::LocationRange> range(dummyLocation());
  auto ctx        = makeSereneContext();
  llvm::Error err = makeError(*ctx, PassFailureError, *range, "test error");

  CHECK(getMessage(err) == "test error");

  CHECK_ERR(PassFailureError, std::move(err));
}

}; // namespace errors
} // namespace serene

#endif

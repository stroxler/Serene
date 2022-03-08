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

#ifndef SERENE_TEST_ERRORS_H
#define SERENE_TEST_ERRORS_H

#include "serene/errors.h"

#include <catch2/catch_test_macros.hpp>

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>

namespace serene {
namespace errors {

TEST_CASE("Serene Error construction", "[errors]") {
  {
    std::unique_ptr<reader::LocationRange> range(dummyLocation());
    llvm::Error err = llvm::make_error<PassFailureError>(*range, "test error");

    auto unhandled =
        llvm::handleErrors(std::move(err), [&](const PassFailureError &e) {
          REQUIRE(e.message() == "test error");
          CHECK(errorVariants[e.ID].title == "PassFailureError");
          CHECK(errorVariants[e.ID].desc == "Pass Failure.");
          CHECK(errorVariants[e.ID].help.empty());
        });

    CHECK(!unhandled);
  };
} // namespace errors
}; // namespace errors
} // namespace serene

#endif

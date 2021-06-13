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

#ifndef SERENE_UTILS_H
#define SERENE_UTILS_H

#include "llvm/Support/Error.h"

#include <variant>

// C++17 required. We can't go back to 14 any more :))

namespace serene {

/// A similar type to Rust's Result data structure. It either holds a value of
/// type `T` successfully or holds a value of type `E` errorfully. It is
/// designed to be used in situations which the return value of a function might
/// contains some errors. The official way to use this type is to use the
/// factory functions `Success` and `Error`. For example:
///
/// \code
/// auto successfulResult = Result<int>::success(3);
/// auto notOkResult = Result<int>::error(SomeLLVMError());
// \endcode
///
/// In order check for a value being errorful or successful checkout the `ok`
/// method or simply use the value as a conditiona.
template <typename T, typename E = llvm::Error>
class Result {

  // The actual data container
  std::variant<T, E> contents;

  /// The main constructor which we made private to avoid ambiguousness in
  /// input type. `Success` and `Error` call this ctor.
  template <typename InPlace, typename Content>
  Result(InPlace i, Content &&c) : contents(i, std::forward<Content>(c)){};

public:
  /// Create a succesfull result with the given value of type `T`.
  static Result success(T v) {
    return Result(std::in_place_index_t<0>(), std::move(v));
  }

  /// Create an errorful result with the given value of type `E` (default
  /// `llvm::Error`).
  static Result error(E e) {
    return Result(std::in_place_index_t<1>(), std::move(e));
  }

  /// Return the value if it's successful otherwise throw an error
  T &&getValue() && { return std::move(std::get<0>(contents)); };

  /// Return the error value if it's errorful otherwise throw an error
  E &&getError() && { return std::move(std::get<1>(contents)); };

  // using std::get, it'll throw if contents doesn't contain what you ask for

  /// Return the value if it's successful otherwise throw an error
  T &getValue() & { return std::get<0>(contents); };

  /// Return the error value if it's errorful otherwise throw an error
  E &getError() & { return std::get<1>(contents); };

  const T &getValue() const & { return std::get<0>(contents); }
  const E &getError() const & { return std::get<1>(contents); }

  /// Return the a boolean value indicating whether the value is succesful
  /// or errorful.
  bool ok() const { return std::holds_alternative<T>(contents); };

  operator bool() const { return ok(); }
};

} // namespace serene
#endif

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

#include "serene/reader/location.h"

#include "mlir/IR/Identifier.h"

#include "llvm/Support/FormatVariadic.h"

namespace serene {
namespace reader {

LocationRange::LocationRange(const LocationRange &loc) {
  start = loc.start;
  end   = loc.end;
}

/// Return the string represenation of the location.
std::string Location::toString() const {
  return llvm::formatv("{0}:{1}:{2}", line, col, pos);
};

/// Increase the given location by one and set the line/col value in respect to
/// the `newline` in place.
/// \param loc The `Location` data
/// \param newline Whether or not we reached a new line
void inc_location(Location &loc, bool newline) {
  loc.pos++;

  if (!newline) {
    loc.col++;
  } else {
    loc.col = 0;
    loc.line++;
  }
}

/// decrease the given location by one and set the line/col value in respect to
/// the `newline` in place.
/// \param loc The `Location` data
/// \param newline Whether or not we reached a new line
void dec_location(Location &loc, bool newline) {
  loc.pos = loc.pos == 0 ? 0 : loc.pos - 1;

  if (newline) {
    loc.line = loc.line == 0 ? 0 : loc.line - 1;
    // We don't move back the `col` value because we simply don't know it
  } else {
    loc.col = loc.col == 0 ? 0 : loc.col - 1;
  }
}

} // namespace reader
} // namespace serene

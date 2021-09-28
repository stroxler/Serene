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

#include "serene/context.h"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Identifier.h>

namespace serene {
namespace reader {

// LocationRange::LocationRange(const LocationRange &loc) {
//   start = loc.start.clone();
//   end   = loc.end.clone();
// }

/// Return the string represenation of the location.
std::string Location::toString() const {
  return llvm::formatv("{0}:{1}", line, col);
};

Location Location::clone() {
  return Location{ns, filename, c, line, col, knownLocation};
}

Location Location::clone() const {
  return Location{ns, filename, c, line, col, knownLocation};
}

mlir::Location Location::toMLIRLocation(SereneContext &ctx) {
  // TODO: Create a new Location attribute that is namespace base
  if (filename.hasValue()) {
    return mlir::FileLineColLoc::get(&ctx.mlirContext, filename.getValue(),
                                     line, col);
  }
  return mlir::FileLineColLoc::get(&ctx.mlirContext, ns, line, col);
}
/// Increase the given location by one and set the line/col value in respect to
/// the `newline` in place.
/// \param loc The `Location` data
/// \param c A pointer to the current char that the location has to point to
void incLocation(Location &loc, const char *c) {
  // TODO: Handle the end of line with respect to the OS.
  // increase the current position in the buffer with respect to the end
  // of line.
  auto newline = *c == '\n';

  if (!newline) {
    loc.col++;
  } else {
    loc.line++;
    loc.col = 0;
  }
}

/// decrease the given location by one and set the line/col value in respect to
/// the `newline` in place.
/// \param loc The `Location` data
/// \param c A pointer to the current char that the location has to point to
void decLocation(Location &loc, const char *c) {
  // TODO: Handle the end of line with respect to the OS.
  // increase the current position in the buffer with respect to the end
  // of line.
  auto newline = *c == '\n';

  if (newline) {
    loc.line = loc.line == 0 ? 0 : loc.line - 1;

    // We don't move back the `col` value because we simply don't know it
  } else {
    loc.col = loc.col == 0 ? 0 : loc.col - 1;
  }
}

} // namespace reader
} // namespace serene

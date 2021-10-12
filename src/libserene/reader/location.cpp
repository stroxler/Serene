/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

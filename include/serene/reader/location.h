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

#ifndef LOCATION_H
#define LOCATION_H

#include "mlir/IR/Diagnostics.h"

#include <mlir/IR/Location.h>
#include <string>

namespace serene {
class SereneContext;

namespace reader {

/// It represents a location in the input string to the parser via `line`,
struct Location {
  llvm::StringRef ns;
  /// A pointer to the character that this location is pointing to
  /// it the input buffer
  const char *c = nullptr;

  /// At this stage we only support 65535 lines of code in each file
  unsigned short int line;
  /// At this stage we only support 65535 chars in each line
  unsigned short int col;

  ::std::string toString() const;
  Location() = default;
  Location clone();

  mlir::Location toMLIRLocation(SereneContext &ctx, llvm::StringRef ns);
};

class LocationRange {
public:
  Location start;
  Location end;

  LocationRange() = default;
  LocationRange(Location _start) : start(_start), end(_start){};
  LocationRange(Location _start, Location _end) : start(_start), end(_end){};
  LocationRange(const LocationRange &);
};

void incLocation(Location &, const char *);
void decLocation(Location &, const char *);

} // namespace reader
} // namespace serene
#endif

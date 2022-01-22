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

#include <errors-backend.h>

#include <llvm/Support/Format.h>

#define DEBUG_TYPE "errors-backend"

namespace serene {

// Any helper data structures can be defined here. Some backends use
// structs to collect information from the records.

class ErrorsBackend {
private:
  llvm::RecordKeeper &records;

public:
  ErrorsBackend(llvm::RecordKeeper &rk) : records(rk) {}

  void run(llvm::raw_ostream &os);
}; // emitter class

void ErrorsBackend::run(llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader("Serene's Errors collection", os);

  (void)records;
}

void emitErrors(llvm::RecordKeeper &rk, llvm::raw_ostream &os) {
  ErrorsBackend(rk).run(os);
}

} // namespace serene

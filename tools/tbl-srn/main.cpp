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

/**
 * Commentary:
 * This is the main file for Serene's tablegen instance. All the backends
 * are local to this instance and we use this instance alongside the official
 * instances.
 */

#include "serene/errors-backend.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/SetTheory.h>

namespace cl = llvm::cl;

namespace serene {
enum ActionType { PrintRecords, ErrorsBackend };

cl::opt<ActionType> action(
    cl::desc("Action to perform:"),

    cl::values(
        clEnumValN(ErrorsBackend, "errors-backend",
                   "Generate a C++ file containing errors in the input file"),
        clEnumValN(PrintRecords, "print-records",
                   "Print all records to stdout (default)")));

bool llvmTableGenMain(llvm::raw_ostream &os, llvm::RecordKeeper &records) {
  switch (action) {
  case ErrorsBackend:
    emitErrors(records, os);
    break;
  case PrintRecords:
    os << records;
    break;
  }
  return false;
}
} // namespace serene

int main(int argc, char **argv) {
  llvm::InitLLVM x(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return llvm::TableGenMain(argv[0], &serene::llvmTableGenMain);
}

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__) || \
    __has_feature(leak_sanitizer)

#include <sanitizer/lsan_interface.h>
// Disable LeakSanitizer for this binary as it has too many leaks that are not
// very interesting to fix. See compiler-rt/include/sanitizer/lsan_interface.h .
LLVM_ATTRIBUTE_USED int __lsan_is_turned_off() { return 1; }

#endif

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

/**
 * Commentary:
 * `DiagEngine` is in charge of error handling of the compiler. It receives
 * the incoming errors (`llvm::Error`) and print them out to stderr.
 *
 * Errors might raise from different contextes and the incoming channel might
 * vary. For exmaple in many case we just return `llvm::Error` from functions
 * and propagate them to the highest level call site and deal with them there
 * or during the pass management we call the `emit()` function on operations
 * to report back an error.
 *
 * Serene extends `llvm::Error`. For more info have a look at
 * `serene/errors/base.h` and the `serene/errors/errors.td`.
 */

#ifndef SERENE_DIAGNOSTICS_H
#define SERENE_DIAGNOSTICS_H

#include "serene/errors.h"
#include "serene/export.h"
#include "serene/reader/location.h"
#include "serene/source_mgr.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>

#include <memory>

namespace serene {
class SereneContext;
class DiagnosticEngine;

// TODO: Finish up the Diagnostic interface and utility functions
//       to build diagnostics from Errors

class Diagnostic {
  // TODO: Add support for llvm::SMFixIt
  friend DiagnosticEngine;

  enum Type {
    Error,

    // TODO: Add support for remarks and notes
    Remark,
    Note,
  };

  SereneContext &ctx;
  reader::LocationRange loc;
  std::string fn;
  llvm::Error *err = nullptr;
  Type type        = Type::Error;
  std::string message, lineContents;

  std::string getPrefix(llvm::StringRef prefix = "");

public:
  Diagnostic(SereneContext &ctx, reader::LocationRange loc, llvm::Error *e,
             llvm::StringRef msg, llvm::StringRef fn = "")
      : ctx(ctx), loc(loc), fn(fn), err(e), message(msg){};

protected:
  void print(llvm::raw_ostream &os, llvm::StringRef prefix = "") const;
  void writeColorByType(llvm::raw_ostream &os, llvm::StringRef str);
};

/// DiagnosticEngine is the central hub for dealing with errors in Serene. It
/// integrates with MLIR's diag engine and LLVM's error system to handle error
/// reporting for Serene's compiler
class DiagnosticEngine {
  SereneContext &ctx;

  mlir::DiagnosticEngine &diagEngine;

  Diagnostic toDiagnostic(reader::LocationRange loc, llvm::Error &e,
                          llvm::StringRef msg, llvm::StringRef fn = "");

  void print(llvm::raw_ostream &os, Diagnostic &d);

public:
  DiagnosticEngine(SereneContext &ctx);

  void enqueueError(llvm::StringRef msg);
  void emitSyntaxError(reader::LocationRange loc, llvm::Error &e,
                       llvm::StringRef msg = "");

  void emit(const llvm::Error &err);
};

/// Create a new instance of the `DiagnosticEngine` from the give
/// `SereneContext`
std::unique_ptr<DiagnosticEngine> makeDiagnosticEngine(SereneContext &ctx);

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

/// Throw out an error with the given \p and terminate the execution.
SERENE_EXPORT void panic(SereneContext &ctx, llvm::Twine msg);

/// Throw out the give error \p err and stop the execution.
SERENE_EXPORT void panic(SereneContext &ctx, const llvm::Error &err);

/// Throw the give `llvm::Error` \p errs to the stderr.
SERENE_EXPORT void throwErrors(SereneContext &ctx, const llvm::Error &err);
} // namespace serene

#endif

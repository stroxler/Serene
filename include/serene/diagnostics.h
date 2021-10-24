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

#ifndef SERENE_DIAGNOSTICS_H
#define SERENE_DIAGNOSTICS_H

#include "serene/errors/constants.h"
#include "serene/errors/error.h"
#include "serene/reader/location.h"
#include "serene/source_mgr.h"

#include <serene/export.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>

#include <memory>

namespace serene {
class SereneContext;
class DiagnosticEngine;

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
  errors::ErrorVariant *err = nullptr;
  Type type                 = Type::Error;
  std::string message, lineContents;

  std::string getPrefix(llvm::StringRef prefix = "");

public:
  Diagnostic(SereneContext &ctx, reader::LocationRange loc,
             errors::ErrorVariant *e, llvm::StringRef msg,
             llvm::StringRef fn = "")
      : ctx(ctx), loc(loc), fn(fn), err(e), message(msg){};

protected:
  void print(llvm::raw_ostream &os, llvm::StringRef prefix = "");
  void writeColorByType(llvm::raw_ostream &os, llvm::StringRef str);
};

class DiagnosticEngine {
  SereneContext &ctx;

  mlir::DiagnosticEngine &diagEngine;

  Diagnostic toDiagnostic(reader::LocationRange loc, errors::ErrorVariant &e,
                          llvm::StringRef msg, llvm::StringRef fn = "");

  void print(llvm::raw_ostream &os, Diagnostic &d);

public:
  DiagnosticEngine(SereneContext &ctx);

  void enqueueError(llvm::StringRef msg);
  void emitSyntaxError(reader::LocationRange loc, errors::ErrorVariant &e,
                       llvm::StringRef msg = "");

  void emit(const errors::ErrorPtr &err);
  void emit(const errors::ErrorTree &errs);

  /// Throw out an error with the given `msg` and terminate the execution
  void panic(llvm::StringRef msg);
};

/// Create a new instance of the `DiagnosticEngine` from the give
/// `SereneContext`
std::unique_ptr<DiagnosticEngine> makeDiagnosticEngine(SereneContext &ctx);

/// Throw out an error with the given `msg` and terminate the execution.
SERENE_EXPORT void panic(SereneContext &ctx, llvm::StringRef msg);

/// Throw the give `ErrorTree` \p errs and terminate the execution.
SERENE_EXPORT void throwErrors(SereneContext &ctx, errors::ErrorTree &errs);
} // namespace serene

#endif

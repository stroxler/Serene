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

#ifndef SERENE_DIAGNOSTICS_H
#define SERENE_DIAGNOSTICS_H

#include "serene/errors/constants.h"
#include "serene/reader/location.h"
#include "serene/source_mgr.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/Diagnostics.h>

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
};

std::unique_ptr<DiagnosticEngine> makeDiagnosticEngine(SereneContext &ctx);
} // namespace serene

#endif

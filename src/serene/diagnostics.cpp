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

#include "serene/diagnostics.h"

#include "serene/context.h"
#include "serene/reader/location.h"
#include "serene/source_mgr.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <llvm/Support/FormatVariadic.h>
#include <memory>

namespace serene {
void Diagnostic::writeColorByType(llvm::raw_ostream &os, llvm::StringRef str) {
  llvm::ColorMode mode =
      ctx.opts.withColors ? llvm::ColorMode::Auto : llvm::ColorMode::Disable;

  llvm::WithColor s(os, llvm::raw_ostream::SAVEDCOLOR, true, false, mode);

  switch (type) {
  case Type::Error:
    s.changeColor(llvm::raw_ostream::Colors::RED);
    break;
  // case Type::Warning:
  //   s.changeColor(llvm::raw_ostream::Colors::YELLOW);
  //   break;
  case Type::Note:
    s.changeColor(llvm::raw_ostream::Colors::CYAN);
    break;
  case Type::Remark:
    s.changeColor(llvm::raw_ostream::Colors::MAGENTA);
    break;
  }

  s << str;
  s.resetColor();
};

std::string Diagnostic::getPrefix(llvm::StringRef prefix) {
  if (prefix != "") {
    return prefix.str();
  }

  switch (type) {
  case Type::Error:
    return "Error";
    break;
  case Type::Note:
    return "Note";
  case Type::Remark:
    return "Remark";
  }
};

void Diagnostic::print(llvm::raw_ostream &os, llvm::StringRef prefix) {
  llvm::ColorMode mode =
      ctx.opts.withColors ? llvm::ColorMode::Auto : llvm::ColorMode::Disable;

  llvm::WithColor s(os, llvm::raw_ostream::SAVEDCOLOR, true, false, mode);

  s << "[";
  if (err) {
    writeColorByType(os, err->getErrId());
  }
  s << "] ";
  writeColorByType(os, getPrefix(prefix));
  s << ": ";

  if (err) {
    s << err->description << '\n';
  }

  if (message != "") {
    s.changeColor(llvm::raw_ostream::Colors::YELLOW);
    s << "With message";
    s.resetColor();
    s << ": " << message << "\n";
  }

  s << "In ns '";
  s.changeColor(llvm::raw_ostream::Colors::MAGENTA);
  s << loc.start.ns;
  s.resetColor();

  if (loc.start.filename.hasValue()) {
    s << "' At: ";
    s.changeColor(llvm::raw_ostream::Colors::YELLOW);
    s << loc.start.filename.getValue();
    s.resetColor();
    s << ":";
    s.changeColor(llvm::raw_ostream::Colors::CYAN);
    s << loc.start.line;
    s.resetColor();
    s << ":";
    s.changeColor(llvm::raw_ostream::Colors::CYAN);
    s << loc.start.col;
    s.resetColor();
  }
};

DiagnosticEngine::DiagnosticEngine(SereneContext &ctx)
    : ctx(ctx), diagEngine(ctx.mlirContext.getDiagEngine()){};

void DiagnosticEngine::print(llvm::raw_ostream &os, Diagnostic &d){};

Diagnostic DiagnosticEngine::toDiagnostic(reader::LocationRange loc,
                                          errors::ErrorVariant &e,
                                          llvm::StringRef msg,
                                          llvm::StringRef fn) {

  return Diagnostic(ctx, loc, &e, msg, fn);
};

void DiagnosticEngine::emitSyntaxError(reader::LocationRange loc,
                                       errors::ErrorVariant &e,
                                       llvm::StringRef msg) {
  Diagnostic diag(ctx, loc, &e, msg);

  diag.print(llvm::errs(), "SyntaxError");
  exit(1);
};

std::unique_ptr<DiagnosticEngine> makeDiagnosticEngine(SereneContext &ctx) {
  return std::make_unique<DiagnosticEngine>(ctx);
}
} // namespace serene

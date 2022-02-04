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

#include "serene/diagnostics.h"

#include "serene/context.h"
#include "serene/reader/location.h"
#include "serene/source_mgr.h"
#include "serene/utils.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatAdapters.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>

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
  if (!prefix.empty()) {
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

  s << "\n[";
  writeColorByType(os, "Error");
  s << "]>\n";

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

  s << "\n\n";

  const auto &srcBuf = ctx.sourceManager.getBufferInfo(loc.start.ns);
  const char *line   = srcBuf.getPointerForLineNumber(loc.start.line);

  while (*line != '\n' && line != srcBuf.buffer->getBufferEnd()) {
    s << *line;
    line++;
  }

  s << '\n';

  s.changeColor(llvm::raw_ostream::Colors::GREEN);
  s << llvm::formatv("{0}", llvm::fmt_pad("^", (size_t)loc.start.col - 1, 0));
  s.resetColor();

  s << '\n';

  s << "[";
  if (err != nullptr) {
    writeColorByType(os, err->getErrId());
  }
  s << "] ";
  writeColorByType(os, getPrefix(prefix));
  s << ": ";

  if (err != nullptr) {
    s << err->description << '\n';
  }

  if (!message.empty()) {
    s.changeColor(llvm::raw_ostream::Colors::YELLOW);
    s << "With message";
    s.resetColor();
    s << ": " << message << "\n";
  }

  if (err != nullptr) {
    s << "For more information checkout";
    s.changeColor(llvm::raw_ostream::Colors::CYAN);
    s << " `serenec --explain ";
    s << err->getErrId() << "`\n";
  }
};

DiagnosticEngine::DiagnosticEngine(SereneContext &ctx)
    : ctx(ctx), diagEngine(ctx.mlirContext.getDiagEngine()){};

void DiagnosticEngine::print(llvm::raw_ostream &os, Diagnostic &d) {
  UNUSED(ctx);
  UNUSED(os);
  UNUSED(d);
};

Diagnostic DiagnosticEngine::toDiagnostic(reader::LocationRange loc,
                                          llvm::Error &e, llvm::StringRef msg,
                                          llvm::StringRef fn) {

  return Diagnostic(ctx, loc, &e, msg, fn);
};

void DiagnosticEngine::enqueueError(llvm::StringRef msg) {
  llvm::errs() << llvm::formatv("FIX ME (better emit error): {0}\n", msg);
  terminate(ctx, 1);
};

void DiagnosticEngine::emitSyntaxError(reader::LocationRange loc,
                                       llvm::Error &e, llvm::StringRef msg) {
  Diagnostic diag(ctx, loc, &e, msg);

  diag.print(llvm::errs(), "SyntaxError");
  terminate(ctx, 1);
};

void DiagnosticEngine::panic(llvm::StringRef msg) {
  // TODO: Use Diagnostic class here instead
  // TODO: Provide a trace if possible

  llvm::ColorMode mode =
      ctx.opts.withColors ? llvm::ColorMode::Auto : llvm::ColorMode::Disable;

  llvm::WithColor s(llvm::errs(), llvm::raw_ostream::SAVEDCOLOR, true, false,
                    mode);
  s << "\n[";
  s.changeColor(llvm::raw_ostream::Colors::RED);
  s << "Panic";
  s.resetColor();
  s << "]: ";

  s << msg << "\n";
  // TODO: Use a proper error code
  terminate(ctx, 1);
};

std::unique_ptr<DiagnosticEngine> makeDiagnosticEngine(SereneContext &ctx) {
  return std::make_unique<DiagnosticEngine>(ctx);
}

void DiagnosticEngine::emit(const llvm::Error &err) {
  UNUSED(ctx);
  // TODO: create a diag and print it
  llvm::errs() << err << "\n";
};

// void DiagnosticEngine::emit(const llvm::Error &errs) {
//
//   // for (const auto &e : errs) {
//   //   emit(e);
//   // }
//
// };

void panic(SereneContext &ctx, llvm::StringRef msg) {
  ctx.diagEngine->panic(msg);
};

void throwErrors(SereneContext &ctx, llvm::Error &errs) {
  // llvm::handleErrors(errs,
  //                    [](const errors::SereneError &e){});
  ctx.diagEngine->emit(errs);
};

} // namespace serene

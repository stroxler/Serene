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

// The "serene/" part is due to a convention that we use in the project
#include "serene/errors-backend.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LineIterator.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

#define DEBUG_TYPE "errors-backend"

namespace serene {

// Any helper data structures can be defined here. Some backends use
// structs to collect information from the records.

class ErrorsBackend {
private:
  llvm::RecordKeeper &records;

public:
  ErrorsBackend(llvm::RecordKeeper &rk) : records(rk) {}

  void createNSBody(llvm::raw_ostream &os);
  void createErrorClass(const int id, llvm::Record &defRec,
                        llvm::raw_ostream &os);
  void run(llvm::raw_ostream &os);
}; // emitter class

static void inNamespace(llvm::StringRef name, llvm::raw_ostream &os,
                        std::function<void(llvm::raw_ostream &)> f) {
  os << "namespace " << name << " {\n\n";
  f(os);
  os << "} // namespace " << name << "\n";
};

void ErrorsBackend::createErrorClass(const int id, llvm::Record &defRec,
                                     llvm::raw_ostream &os) {
  (void)records;

  const auto recName = defRec.getName();

  os << "class " << recName << " : public llvm::ErrorInfo<" << recName
     << "> {\n";
  os << "  static int ID = " << id << ";\n";

  for (const auto &val : defRec.getValues()) {
    auto valName = val.getName();

    if (!(valName == "title" || valName == "description")) {
      llvm::PrintWarning("Only 'title' and 'description' are allowed.");
      llvm::PrintWarning("Record: " + recName);
      continue;
    }

    auto *stringVal = llvm::dyn_cast<llvm::StringInit>(val.getValue());

    if (stringVal == nullptr) {
      llvm::PrintError("The value of " + valName + " is not string.");
      llvm::PrintError("Record: " + recName);
      continue;
    }

    if (stringVal->getValue().empty()) {
      llvm::PrintError("The value of " + valName + " is an empty string.");
      llvm::PrintError("Record: " + recName);
      continue;
    }

    os << "  static std::string " << valName << " = ";

    const llvm::MemoryBufferRef value(stringVal->getValue(), valName);
    llvm::line_iterator lines(value, false);
    while (!lines.is_at_end()) {
      if (lines.line_number() != 1) {
        os << '\t';
      }
      auto prevLine = *lines;
      lines++;
      os << '"' << prevLine << '"';

      if (lines.is_at_end()) {
        os << ";\n";
      } else {
        os << '\n';
      }
    }
  }
  os << "};\n\n";
};

void ErrorsBackend::createNSBody(llvm::raw_ostream &os) {
  int counter = 1;
  for (const auto &defPair : records.getDefs()) {
    llvm::Record &defRec = *defPair.second;

    if (!defRec.isSubClassOf("Error")) {
      continue;
    }

    createErrorClass(counter, defRec, os);

    counter++;
  }

  (void)records;
}

void ErrorsBackend::run(llvm::raw_ostream &os) {
  (void)records;
  llvm::emitSourceFileHeader("Serene's Errors collection", os);

  os << "#include <llvm/Support/Error.h>\n\n";
  inNamespace("serene::errors", os,
              [&](llvm::raw_ostream &os) { createNSBody(os); });
}

void emitErrors(llvm::RecordKeeper &rk, llvm::raw_ostream &os) {
  ErrorsBackend(rk).run(os);
}

} // namespace serene

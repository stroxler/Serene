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
 * This is a `tablegen` backend to read from generate error definitions
 * from the given tablegen records defined in a `.td` file. It relies on
 * Two main classes to be available in the target source code. `SereneError`
 * and `ErrorVariant`. Checkout `libserene/include/serene/errors/base.h`.
 */

// The "serene/" part is due to a convention that we use in the project
#include "serene/errors-backend.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/LineIterator.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

#define DEBUG_TYPE      "errors-backend"
#define INSTANCE_SUFFIX "Instance"

namespace serene {

// Any helper data structures can be defined here. Some backends use
// structs to collect information from the records.

class ErrorsBackend {
private:
  llvm::RecordKeeper &records;

public:
  explicit ErrorsBackend(llvm::RecordKeeper &rk) : records(rk) {}

  void createNSBody(llvm::raw_ostream &os);
  void createErrorClass(int id, llvm::Record &defRec, llvm::raw_ostream &os);
  void run(llvm::raw_ostream &os);
}; // emitter class

static void inNamespace(llvm::StringRef name, llvm::raw_ostream &os,
                        std::function<void(llvm::raw_ostream &)> f) {

  os << "namespace " << name << " {\n\n";
  f(os);
  os << "}; // namespace " << name << "\n";
};

void ErrorsBackend::createErrorClass(const int id, llvm::Record &defRec,
                                     llvm::raw_ostream &os) {
  (void)records;
  (void)id;

  const auto recName = defRec.getName();

  os << "  " << recName << ",\n";
  // os << "class " << recName << " : public llvm::ErrorInfo<" << recName << ",
  // "
  //    << "SereneError> {\n"
  //    << "public:\n"
  //    << "  using llvm::ErrorInfo<" << recName << ", "
  //    << "SereneError>::ErrorInfo;\n"
  //    << "  constexpr static const int ID = " << id << ";\n};\n\n";
};

void ErrorsBackend::createNSBody(llvm::raw_ostream &os) {
  auto *index = records.getGlobal("errorsIndex");

  if (index == nullptr) {
    llvm::PrintError("'errorsIndex' var is missing!");
    return;
  }

  auto *indexList = llvm::dyn_cast<llvm::ListInit>(index);

  if (indexList == nullptr) {
    llvm::PrintError("'errorsIndex' has to be a list!");
    return;
  }

  os << "#ifdef GET_CLASS_DEFS\n";
  inNamespace("serene::errors", os, [&](llvm::raw_ostream &os) {
    os << "enum ErrorType {\n";
    for (size_t i = 0; i < indexList->size(); i++) {
      llvm::Record *defRec = indexList->getElementAsRecord(i);

      if (!defRec->isSubClassOf("Error")) {
        continue;
      }

      createErrorClass(i, *defRec, os);
    }
    os << "};\n\n";

    os << "#define NUMBER_OF_ERRORS " << indexList->size() << "\n";
    os << "static const ErrorVariant errorVariants[" << indexList->size()
       << "] = {\n";

    for (size_t i = 0; i < indexList->size(); i++) {
      llvm::Record *defRec = indexList->getElementAsRecord(i);
      auto recName         = defRec->getName();

      if (!defRec->isSubClassOf("Error")) {
        continue;
      }

      os << "  ErrorVariant::make(" << i << ", \n";
      os << "  \"" << recName << "\",\n";

      auto desc = defRec->getValueAsString("desc");

      if (desc.empty()) {
        llvm::PrintError("'desc' field is empty for " + recName);
      }

      os << "  \"" << desc << "\",\n";

      auto help = defRec->getValueAsString("help");

      if (!help.empty()) {

        const llvm::MemoryBufferRef value(help, "help");

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
      } else {
        os << "  \"\"";
      }

      os << "),\n";
    }

    os << "};\n";
  });

  os << "#undef GET_CLASS_DEFS\n#endif\n\n";
}

void ErrorsBackend::run(llvm::raw_ostream &os) {
  (void)records;
  llvm::emitSourceFileHeader("Serene's Errors collection", os);

  // DO NOT GUARD THE HEADER WITH #ifndef ...
  os << "#include \"serene/errors/variant.h\"\n\n#include "
        "<llvm/Support/Error.h>\n\n";

  createNSBody(os);
}

void emitErrors(llvm::RecordKeeper &rk, llvm::raw_ostream &os) {
  ErrorsBackend(rk).run(os);
}

} // namespace serene

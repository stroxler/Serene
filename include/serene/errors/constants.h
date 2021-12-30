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

#ifndef SERENE_ERRORS_CONSTANTS_H
#define SERENE_ERRORS_CONSTANTS_H

#include <llvm/Support/FormatVariadic.h>

#include <map>
#include <string>
#include <utility>

namespace serene {
namespace errors {

/// This enum represent the expression type and **not** the value type.
enum class ErrType {
  Syntax,
  Semantic,
  Compile,
};

enum ErrID {
  E0000 = 0,
  E0001,
  E0002,
  E0003,
  E0004,
  E0005,
  E0006,
  E0007,
  E0008,
  E0009,
  E0010,
  E0011,
  E0012,
  E0013,
  E0014,
};

struct ErrorVariant {
  ErrID id;

  std::string description;
  std::string longDescription;

  ErrorVariant(ErrID id, std::string desc, std::string longDesc)
      : id(id), description(std::move(desc)),
        longDescription(std::move(longDesc)){};

  std::string getErrId() { return llvm::formatv("E{0:d}", id); };
};

static ErrorVariant
    UnknownError(E0000, "Can't find any description for this error.", "");
static ErrorVariant
    DefExpectSymbol(E0001, "The first argument to 'def' has to be a Symbol.",
                    "");

static ErrorVariant DefWrongNumberOfArgs(
    E0002, "Wrong number of arguments is passed to the 'def' form.", "");

static ErrorVariant FnNoArgsList(E0003, "'fn' form requires an argument list.",
                                 "");

static ErrorVariant FnArgsMustBeList(E0004, "'fn' arguments should be a list.",
                                     "");

static ErrorVariant CantResolveSymbol(E0005, "Can't resolve the given name.",
                                      "");
static ErrorVariant
    DontKnowHowToCallNode(E0006, "Don't know how to call the given expression.",
                          "");

static ErrorVariant PassFailureError(E0007, "Pass Failure.", "");

static ErrorVariant NSLoadError(E0008, "Faild to find a namespace.", "");

static ErrorVariant
    NSAddToSMError(E0009, "Faild to add the namespace to the source manager.",
                   "");

static ErrorVariant
    EOFWhileScaningAList(E0010, "EOF reached before closing of list", "");

static ErrorVariant InvalidDigitForNumber(E0011, "Invalid digit for a number.",
                                          "");

static ErrorVariant
    TwoFloatPoints(E0012, "Two or more float point characters in a number", "");

static ErrorVariant
    InvalidCharacterForSymbol(E0013, "Invalid character for a symbol", "");

static ErrorVariant CompilationError(E0014, "Compilation error!", "");

static std::map<ErrID, ErrorVariant *> ErrDesc = {
    {E0000, &UnknownError},          {E0001, &DefExpectSymbol},
    {E0002, &DefWrongNumberOfArgs},  {E0003, &FnNoArgsList},
    {E0004, &FnArgsMustBeList},      {E0005, &CantResolveSymbol},
    {E0006, &DontKnowHowToCallNode}, {E0007, &PassFailureError},
    {E0008, &NSLoadError},           {E0009, &NSAddToSMError},
    {E0010, &EOFWhileScaningAList},  {E0011, &InvalidDigitForNumber},
    {E0012, &TwoFloatPoints},        {E0013, &InvalidCharacterForSymbol},
    {E0014, &CompilationError}};

} // namespace errors
} // namespace serene
#endif

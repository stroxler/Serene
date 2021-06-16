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

#ifndef SERENE_SLIR_GENERATABLE_H
#define SERENE_SLIR_GENERATABLE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "serene/reader/location.h"
#include "serene/traits.h"

#include <utility>

namespace serene {
class Namespace;
}

namespace serene::slir {

template <typename T>
class GeneratableUnit : public TraitBase<T, GeneratableUnit> {
public:
  GeneratableUnit(){};
  GeneratableUnit(const GeneratableUnit &) = delete;

  void generate(serene::Namespace &ns) {
    // TODO: should we return any status or possible error here or
    //       should we just populate them in a ns wide state?
    this->Object().generateIR(ns);
  };
};

template <typename T>
class Generatable : public TraitBase<T, Generatable> {
public:
  Generatable(){};
  Generatable(const Generatable &) = delete;

  mlir::ModuleOp &generate();
  mlir::LogicalResult runPasses();

  void dump() { this->Object().dump(); };
};

template <typename T>
void dump(Generatable<T> &t) {
  t.dump();
};

} // namespace serene::slir

#endif

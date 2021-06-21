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

#include "serene/context.h"

#include "serene/namespace.h"
#include "serene/passes.h"
#include "serene/slir/generatable.h"

namespace serene {

void SereneContext::insertNS(std::shared_ptr<Namespace> ns) {
  namespaces[ns->name.str()] = ns;
};

std::shared_ptr<Namespace> SereneContext::getNS(llvm::StringRef ns_name) {
  if (namespaces.count(ns_name.str())) {
    return namespaces[ns_name.str()];
  }

  return nullptr;
};

bool SereneContext::setCurrentNS(llvm::StringRef ns_name) {
  if (namespaces.count(ns_name.str())) {
    this->current_ns = ns_name;
    return true;
  }

  return false;
};

std::shared_ptr<Namespace> SereneContext::getCurrentNS() {
  assert(!this->current_ns.empty() && "Current namespace is not set");

  if (namespaces.count(this->current_ns)) {
    return namespaces[this->current_ns];
  }

  return nullptr;
};

void SereneContext::setOperationPhase(CompilationPhase phase) {
  this->targetPhase = phase;

  if (phase == CompilationPhase::SLIR) {
    return;
  }

  if (phase >= CompilationPhase::MLIR) {
    pm.addPass(serene::passes::createSLIRLowerToAffinePass());
  }

  if (phase >= CompilationPhase::LIR) {
    pm.addPass(serene::passes::createSLIRLowerToLLVMDialectPass());
  }
};

int SereneContext::getOptimizatioLevel() {
  if (targetPhase <= CompilationPhase::NoOptimization) {
    return 0;
  }

  if (targetPhase == CompilationPhase::O1) {
    return 1;
  }
  if (targetPhase == CompilationPhase::O2) {
    return 2;
  }
  return 3;
}

std::unique_ptr<SereneContext> makeSereneContext() {
  return std::make_unique<SereneContext>();
};
}; // namespace serene

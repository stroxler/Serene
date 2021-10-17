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

#include "serene/context.h"

#include "serene/namespace.h"
#include "serene/passes.h"
#include "serene/reader/location.h"
#include "serene/slir/generatable.h"

#include <llvm/Support/FormatVariadic.h>
#include <utility>

namespace serene {

void SereneContext::insertNS(const std::shared_ptr<Namespace> &ns) {
  namespaces[ns->name] = ns;
};

std::shared_ptr<Namespace> SereneContext::getNS(llvm::StringRef ns_name) {
  if (namespaces.count(ns_name.str()) != 0) {
    return namespaces[ns_name.str()];
  }

  return nullptr;
};

bool SereneContext::setCurrentNS(llvm::StringRef ns_name) {
  if (namespaces.count(ns_name.str()) != 0) {
    this->current_ns = ns_name;
    return true;
  }

  return false;
};

Namespace &SereneContext::getCurrentNS() {
  if (this->current_ns.empty() || (namespaces.count(this->current_ns) == 0)) {
    panic(*this, llvm::formatv("getCurrentNS: Namespace '{0}' does not exist",
                               this->current_ns)
                     .str());
  }

  return *namespaces[this->current_ns];
};

void SereneContext::setOperationPhase(CompilationPhase phase) {
  this->targetPhase = phase;

  if (phase == CompilationPhase::SLIR) {
    return;
  }

  if (phase >= CompilationPhase::MLIR) {
    pm.addPass(serene::passes::createSLIRLowerToMLIRPass());
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

NSPtr SereneContext::readNamespace(const std::string &name) {
  auto loc = reader::LocationRange::UnknownLocation(name);

  return readNamespace(name, loc);
};

NSPtr SereneContext::readNamespace(std::string name,
                                   reader::LocationRange loc) {
  return sourceManager.readNamespace(*this, std::move(name), loc);
}

std::unique_ptr<SereneContext> makeSereneContext() {
  return std::make_unique<SereneContext>();
};

}; // namespace serene

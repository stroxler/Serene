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

namespace serene {

void SereneContext::insertNS(std::shared_ptr<Namespace> ns) {
  namespaces[ns->name] = ns;
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
  // TODO: replace the assertion with a runtime check
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

NSPtr SereneContext::readNamespace(std::string name) {
  auto loc = reader::LocationRange::UnknownLocation(name);

  return readNamespace(name, loc);
};

NSPtr SereneContext::readNamespace(std::string name,
                                   reader::LocationRange loc) {
  return sourceManager.readNamespace(*this, name, loc);
}

std::unique_ptr<SereneContext> makeSereneContext() {
  return std::make_unique<SereneContext>();
};

}; // namespace serene

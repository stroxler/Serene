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

#include "serene/context.h"

#include "serene/namespace.h"
#include "serene/passes.h"
#include "serene/reader/location.h"
#include "serene/slir/generatable.h"

#include <llvm/Support/FormatVariadic.h>

#include <utility>

namespace serene {

void SereneContext::insertNS(NSPtr &ns) {
  auto nsName        = ns->name;
  namespaces[nsName] = ns;
};

Namespace *SereneContext::getNS(llvm::StringRef nsName) {
  if (namespaces.count(nsName.str()) != 0) {
    return namespaces[nsName.str()].get();
  }

  return nullptr;
};

Namespace &SereneContext::getCurrentNS() {
  if (this->currentNS.empty() || (namespaces.count(this->currentNS) == 0)) {
    panic(*this, llvm::formatv("getCurrentNS: Namespace '{0}' does not exist",
                               this->currentNS)
                     .str());
  }

  return *namespaces[this->currentNS];
};

NSPtr SereneContext::getSharedPtrToNS(llvm::StringRef nsName) {
  if (namespaces.count(nsName.str()) == 0) {
    return nullptr;
  }

  return namespaces[nsName.str()];
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

NSPtr SereneContext::makeNamespace(llvm::StringRef name,
                                   llvm::Optional<llvm::StringRef> filename) {
  auto ns = Namespace::make(*this, name, filename);

  if (ns != nullptr) {
    insertNS(ns);
  }

  return ns;
};

MaybeNS SereneContext::readNamespace(const std::string &name) {
  // TODO: Replace this location with a proper location indicating
  //       the reason why the location data is not available.
  //       For example, because the ns might be the entry point ns.
  auto loc = reader::LocationRange::UnknownLocation(name);

  return readNamespace(name, loc);
};

MaybeNS SereneContext::readNamespace(const std::string &name,
                                     reader::LocationRange loc) {
  return withCurrentNS<MaybeNS>(name, [&]() -> MaybeNS {
    return this->sourceManager.readNamespace(*this, name, loc);
  });
}

MaybeNS SereneContext::importNamespace(const std::string &name) {
  auto loc = reader::LocationRange::UnknownLocation(name);
  return importNamespace(name, loc);
}

MaybeNS SereneContext::importNamespace(const std::string &name,
                                       reader::LocationRange loc) {
  auto maybeNS = readNamespace(name, loc);

  if (maybeNS) {
    auto &ns = *maybeNS;
    if (auto err = jit->addNS(*ns, loc)) {
      return err;
    }

    insertNS(ns);
  }

  return maybeNS;
}

llvm::orc::JITDylib *SereneContext::getLatestJITDylib(Namespace &ns) {

  if (jitDylibs.count(ns.name) == 0) {
    return nullptr;
  }

  auto vec = jitDylibs[ns.name];
  // TODO: Make sure that the returning Dylib still exists in the JIT
  //       by calling jit->engine->getJITDylibByName(dylib_name);
  return vec.empty() ? nullptr : vec.back();
};

void SereneContext::pushJITDylib(Namespace &ns, llvm::orc::JITDylib *l) {
  if (jitDylibs.count(ns.name) == 0) {
    llvm::SmallVector<llvm::orc::JITDylib *, 1> vec{l};
    jitDylibs[ns.name] = vec;
    return;
  }
  auto vec = jitDylibs[ns.name];
  vec.push_back(l);
  jitDylibs[ns.name] = vec;
}

size_t SereneContext::getNumberOfJITDylibs(Namespace &ns) {
  if (jitDylibs.count(ns.name) == 0) {
    return 0;
  }
  auto vec = jitDylibs[ns.name];
  return vec.size();
};

void terminate(SereneContext &ctx, int exitCode) {
  UNUSED(ctx);
  // TODO: Since we are running in a single thread for now using exit is fine
  // but we need to adjust and change it to a thread safe termination
  // process later on.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  std::exit(exitCode);
}

std::unique_ptr<SereneContext> makeSereneContext(Options opts) {
  return SereneContext::make(opts);
};

}; // namespace serene

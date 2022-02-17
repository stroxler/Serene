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

#include "serene/jit/engine.h"

#include "serene/context.h"
#include "serene/jit/layers.h"
#include "serene/utils.h"

#include <memory>

namespace serene::jit {

static void handleLazyCallThroughError() {
  // TODO: Report to the diag engine
  llvm::errs() << "LazyCallThrough error: Could not find function body";
  // TODO: terminate ?
}

SereneJIT::SereneJIT(Namespace &entryNS,
                     std::unique_ptr<orc::ExecutionSession> es,
                     std::unique_ptr<orc::EPCIndirectionUtils> epciu,
                     orc::JITTargetMachineBuilder jtmb, llvm::DataLayout &&dl)

    : es(std::move(es)), epciu(std::move(epciu)), dl(dl),
      mangler(*this->es, this->dl),
      objectLayer(
          *this->es,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      compileLayer(
          *this->es, objectLayer,
          std::make_unique<orc::ConcurrentIRCompiler>(std::move(jtmb))),
      transformLayer(*this->es, compileLayer, optimizeModule),
      astLayer(transformLayer, mangler),
      nsLayer(entryNS.getContext(), transformLayer, mangler, dl),
      mainJD(this->es->createBareJITDylib(entryNS.name)),
      ctx(entryNS.getContext()), currentNS(entryNS) {
  UNUSED(this->ctx);
  mainJD.addGenerator(
      cantFail(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dl.getGlobalPrefix())));

  // if (numCompileThreads > 0) {
  //   compileOnDemandLayer.setCloneToNewContextOnEmit(true);
  // }
}

llvm::Error SereneJIT::addNS(llvm::StringRef nsname,
                             orc::ResourceTrackerSP rt) {
  if (!rt) {
    rt = mainJD.getDefaultResourceTracker();
  }

  return nsLayer.add(rt, nsname);
};

llvm::Error SereneJIT::addAst(exprs::Ast &ast, orc::ResourceTrackerSP rt) {
  if (!rt) {
    rt = mainJD.getDefaultResourceTracker();
  }

  return astLayer.add(rt, getCurrentNS(), ast);
};

llvm::Expected<std::unique_ptr<SereneJIT>> makeSereneJIT(Namespace &ns) {
  auto epc = orc::SelfExecutorProcessControl::Create();
  if (!epc) {
    return epc.takeError();
  }
  auto es = std::make_unique<orc::ExecutionSession>(std::move(*epc));
  auto epciu =
      orc::EPCIndirectionUtils::Create(es->getExecutorProcessControl());
  if (!epciu) {
    return epciu.takeError();
  }

  (*epciu)->createLazyCallThroughManager(
      *es, llvm::pointerToJITTargetAddress(&handleLazyCallThroughError));

  if (auto err = setUpInProcessLCTMReentryViaEPCIU(**epciu)) {
    return std::move(err);
  }

  orc::JITTargetMachineBuilder jtmb(
      es->getExecutorProcessControl().getTargetTriple());

  auto dl = jtmb.getDefaultDataLayoutForTarget();
  if (!dl) {
    return dl.takeError();
  }

  return std::make_unique<SereneJIT>(ns, std::move(es), std::move(*epciu),
                                     std::move(jtmb), std::move(*dl));
};
} // namespace serene::jit

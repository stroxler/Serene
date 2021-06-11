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

#ifndef SERENE_CONTEXT_H
#define SERENE_CONTEXT_H

#include "serene/environment.h"
#include "serene/namespace.h"
#include "serene/passes/slir_lowering.h"
#include "serene/slir/dialect.h"
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

namespace serene {

namespace exprs {
class Expression;
using Node = std::shared_ptr<Expression>;
} // namespace exprs

class SereneContext {
  std::map<std::string, std::shared_ptr<Namespace>> namespaces;

  // Why string vs pointer? We might rewrite the namespace and
  // holding a pointer means that it might point to the old version
  std::string current_ns;

public:
  mlir::MLIRContext mlirContext;
  mlir::PassManager pm;
  /// Insert the given `ns` into the context. The Context object is
  /// the owner of all the namespaces. The `ns` will overwrite any
  /// namespace with the same name.
  void insertNS(std::shared_ptr<Namespace> ns);

  /// Sets the n ame of the current namespace in the context and return
  /// a boolean indicating the status of this operation. The operation
  /// will fail if the namespace does not exist in the namespace table.
  bool setCurrentNS(llvm::StringRef ns_name);

  std::shared_ptr<Namespace> getCurrentNS();

  std::shared_ptr<Namespace> getNS(llvm::StringRef ns_name);

  SereneContext() : pm(&mlirContext) {
    mlirContext.getOrLoadDialect<serene::slir::SereneDialect>();
    pm.addPass(serene::passes::createSLIRLowerToAffinePass());
  };
};

/// Creates a new context object. Contexts are used through out the compilation
/// process to store the state
std::unique_ptr<SereneContext> makeSereneContext();

}; // namespace serene

#endif

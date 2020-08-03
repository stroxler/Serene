/**
 * Serene programming language.
 *
 *  Copyright (c) 2020 Sameer Rahmani <lxsameer@gnu.org>
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

#include "serene/special_forms/def.hpp"
#include "serene/compiler.hpp"
#include "serene/list.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/namespace.hpp"
#include "serene/state.hpp"
#include "serene/symbol.hpp"
#include <assert.h>
#include <fmt/core.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {
namespace special_forms {

ast_node Def::make(Compiler &compiler, State &state, const List *args) {
  auto def_ptr = args->at(0).value_or(nullptr);
  auto name_ptr = args->at(1).value_or(nullptr);
  auto body_ptr = args->at(2).value_or(nullptr);

  if (def_ptr && def_ptr->id() == symbol &&
      static_cast<Symbol *>(def_ptr.get())->name() == "def") {

    if (!name_ptr && def_ptr->id() != symbol) {
      compiler.log_error("First argument of 'def' has to be a symbol.");
      return nullptr;
    }

    if (!body_ptr) {
      compiler.log_error("'def' needs 3 arguments, two has been given.");
      return nullptr;
    }

    return make_unique<Def>(static_cast<Symbol *>(name_ptr.get()),
                            body_ptr.get());
  }

  compiler.log_error("Calling 'def' with wrong parameters");
  return nullptr;
};

Def::Def(Symbol *symbol_, AExpr *value_) : m_sym(symbol_), m_value(value_) {}

string Def::string_repr() const {
  // this method is not going to get called.
  return "Def";
}

Value *Def::codegen(Compiler &compiler, State &state) {
  state.set_in_current_ns_root_scope(m_sym->name(),
                                     m_value->codegen(compiler, state));

  // TODO: Do we need to return the codegen of the symbol instead
  // of the symbol itself?
  return m_sym->codegen(compiler, state);
}

Def::~Def() { EXPR_LOG("Destroying def"); };
} // namespace special_forms
} // namespace serene

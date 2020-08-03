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

#include "serene/list.hpp"
#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/special_forms/def.hpp"
#include "serene/symbol.hpp"
#include <bits/c++config.h>
#include <fmt/core.h>
#include <string>

using namespace llvm;

namespace serene {

std::optional<ast_node> List::at(uint index) {
  if (index >= nodes_.size()) {
    return std::nullopt;
  }

  auto itr = cbegin(nodes_);
  std::advance(itr, index);
  return std::make_optional(*itr);
}

void List::cons(ast_node node) { nodes_.push_front(std::move(node)); }

void List::append(ast_node node) { nodes_.push_back(std::move(node)); }

std::string List::string_repr() const {
  std::string s;

  for (auto &n : nodes_) {
    // TODO: Fix the tailing space for the last element
    s = s + n->string_repr() + " ";
  }

  return fmt::format("({})", s);
}

inline size_t List::length() const { return nodes_.size(); }

Value *List::codegen(Compiler &compiler, State &state) {
  if (length() == 0) {
    compiler.log_error("Can't eveluate empty list.");
    return nullptr;
  }

  auto def_ptr = at(0).value_or(nullptr);
  auto name_ptr = at(1).value_or(nullptr);
  auto body_ptr = at(2).value_or(nullptr);

  if (def_ptr && def_ptr->id() == symbol &&
      static_cast<Symbol *>(def_ptr.get())->name() == "def") {

    if (!name_ptr && def_ptr->id() != symbol) {
      return compiler.log_error("First argument of 'def' has to be a symbol.");
    }

    if (!body_ptr) {
      return compiler.log_error("'def' needs 3 arguments, two has been given.");
    }

    special_forms::Def def(static_cast<Symbol *>(name_ptr.get()),
                           body_ptr.get());
    return def.codegen(compiler, state);
  }

  EXPR_LOG("Not implemented in list.");
  return nullptr;
}
} // namespace serene

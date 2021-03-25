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
#include "serene/symbol.hpp"
#include "llvm/ADT/Optional.h"
#include <bits/c++config.h>
#include <fmt/core.h>
#include <string>

using namespace llvm;

namespace serene {

llvm::Optional<ast_node> List::at(uint index) const {
  if (index >= nodes_.size()) {
    return llvm::None;
  }

  auto itr = cbegin(nodes_);
  std::advance(itr, index);
  return llvm::Optional<ast_node>(*itr);
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

std::string List::dumpAST() const {
  std::string s;

  for (auto &n : nodes_) {
    s = fmt::format("{} {}", s, n->dumpAST());
  }

  return fmt::format("<List: {}>", s);
}

inline size_t List::length() const { return nodes_.size(); }

} // namespace serene

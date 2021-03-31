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

List::List(reader::Location startLoc) {
  this->location.reset(new reader::LocationRange(startLoc));
}

List::List(std::vector<ast_node> elements) {
  auto startLoc = elements[0]->location->start;
  auto endLoc = elements[elements.size() - 1]->location->end;
  this->location.reset(new reader::LocationRange(startLoc, endLoc));
  this->nodes_ = elements;
}

llvm::Optional<ast_node> List::at(uint index) const {
  if (index >= nodes_.size()) {
    return llvm::None;
  }

  return llvm::Optional<ast_node>(this->nodes_[index]);
}

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

  return fmt::format("<List [loc: {} | {}]: {} >",
                     this->location->start.toString(),
                     this->location->end.toString(), s);
}

/**
 * Return a sub set of elements starting from the `begin` index to the end
 * and an empty list otherwise.
 */

std::unique_ptr<List> List::from(uint begin) {
  if (this->count() - begin < 1) {
    return makeList(this->location->end);
  }

  std::vector<ast_node>::const_iterator first = this->nodes_.begin() + begin;
  std::vector<ast_node>::const_iterator last = this->nodes_.end();
  fmt::print("#### {} {} \n", this->nodes_.size(), this->nodes_.max_size());
  fmt::print("MM {}\n", this->string_repr());

  std::vector<ast_node> newCopy(first, last);

  return std::make_unique<List>(newCopy);
};

size_t List::count() const { return nodes_.size(); }

/**
 * `classof` is a enabler static method that belongs to the LLVM RTTI interface
 * `llvm::isa`, `llvm::cast` and `llvm::dyn_cast` use this method.
 */
bool List::classof(const AExpr *expr) {
  return expr->getType() == SereneType::List;
}

/**
 * Make an empty List in starts at the given location `loc`.
 */
std::unique_ptr<List> makeList(reader::Location loc) {
  return std::make_unique<List>(loc);
}
/**
 * Make a list out of the given pointer to a List
 */
std::unique_ptr<List> makeList(List *list) {
  return std::unique_ptr<List>(list);
}

} // namespace serene

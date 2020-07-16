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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <string>
#include <fmt/core.h>
#include "serene/llvm/IR/Value.h"
#include "serene/expr.hpp"
#include "serene/list.hpp"

using namespace std;

namespace serene {
  ast_list_node List::to_list(ast_tree lst) {
    auto l = make_unique<List>();

    // Using a for loop with iterator
    for(auto node = rbegin(lst); node != rend(lst); ++node) {
      l = l->cons(move(*node));
    }

    return l;
  }

  ast_list_node List::cons(ast_node f) {

    return make_unique<List>(move(f), unique_ptr<List>(move(*this)));
  }

  string List::string_repr() {
    return fmt::format("<List: '{}'>", first->string_repr());
  };

  size_t List::length() {
    if (this->len) {
      return this->len;
    }

    return 0;
  }

  List::~List() {};
}

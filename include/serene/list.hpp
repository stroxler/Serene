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

#ifndef LIST_H
#define LIST_H

#include <string>
#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"

namespace serene {


  class List: public AExpr {
  public:
    ast_node first;
    std::unique_ptr<List> rest;
    std::size_t len;

    List(): first(nullptr), rest(nullptr), len(0) {};
    List(ast_node f, std::unique_ptr<List> r): first(std::move(f)),
                                               rest(std::move(r)),
                                               len(r ? r->length() + 1 : 0)
    {};

    List(ast_tree list);

    std::string string_repr();
    std::size_t length();
    std::unique_ptr<List> cons(ast_node f);

    static std::unique_ptr<List> to_list(ast_tree lst);


    ~List();
  };

  typedef std::unique_ptr<List> ast_list_node;
}

#endif

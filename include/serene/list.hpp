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

#ifndef LIST_H
#define LIST_H

#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/reader/location.hpp"
#include "llvm/ADT/Optional.h"
#include <list>
#include <string>

namespace serene {

class List : public AExpr {
  std::list<ast_node> nodes_;

public:
  List(){};
  List(reader::Location start);

  ExprId id() const override { return list; }

  std::string dumpAST() const override;
  std::string string_repr() const override;
  size_t length() const;

  void cons(ast_node f);
  void append(ast_node t);

  llvm::Optional<ast_node> at(uint index) const;
};

std::unique_ptr<List> makeList(reader::Location);
std::unique_ptr<List> makeList(reader::Location, List *);

using ast_list_node = std::unique_ptr<List>;
} // namespace serene

#endif

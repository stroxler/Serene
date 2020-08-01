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
#include <string>

namespace serene {

class ListNode {
public:
  ast_node data;
  ListNode *next;
  ListNode *prev;
  ListNode(ast_node node_data)
      : data{std::move(node_data)}, next{nullptr}, prev{nullptr} {};
};

class List : public AExpr {
public:
  ListNode *head;
  ListNode *tail;
  std::size_t len;
  ExprId id{list};

  List() : head{nullptr}, tail{nullptr}, len{0} {};
  List(const List &list);
  List(List &&list) noexcept;

  List &operator=(const List &other);
  List &operator=(List &&other);

  std::string string_repr();
  std::size_t length();

  void cons(ast_node f);
  void append(ast_node t);

  AExpr *at(const int index);

  void cleanup();

  llvm::Value *codegen(Compiler &compiler, State &state);

  virtual ~List();
};

typedef std::unique_ptr<List> ast_list_node;
} // namespace serene

#endif

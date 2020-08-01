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

using namespace std;
using namespace llvm;

namespace serene {
List::List(const List &list) {
  ListNode *root = list.head;

  ListNode *new_head{nullptr};
  ListNode *prev_new_head{nullptr};

  while (root) {
    ListNode *temp = new ListNode(unique_ptr<AExpr>(root->data.get()));

    if (new_head == nullptr) {
      new_head = temp;
      prev_new_head = new_head;

    } else {
      prev_new_head->next = temp;
      prev_new_head = prev_new_head->next;
    }

    root = root->next;
  }
  head = new_head;
};

List::List(List &&list) noexcept
    : head(list.head), tail(list.tail), len(std::exchange(list.len, 0)) {
  list.head = nullptr;
  list.tail = nullptr;
};

List &List::operator=(const List &list) {
  ListNode *root = list.head;

  ListNode *new_head{nullptr};
  ListNode *prev_new_head{nullptr};

  while (root) {
    ListNode *temp = new ListNode(unique_ptr<AExpr>(root->data.get()));

    if (new_head == nullptr) {
      new_head = temp;
      prev_new_head = new_head;

    } else {
      prev_new_head->next = temp;
      prev_new_head = prev_new_head->next;
    }

    root = root->next;
  }
  head = new_head;
  return *this;
};

List &List::operator=(List &&list) {
  head = list.head;
  tail = list.tail;
  len = std::exchange(list.len, 0);
  list.head = nullptr;
  list.tail = nullptr;
  return *this;
};

AExpr *List::at(const int index) {
  ListNode *x = head;
  int j = 0;

  while (j != index) {
    if (!x) {
      // Index is out of range
      return nullptr;
    }
    x = x->next;
    j++;
  }

  return x->data.get();
};

void List::cons(ast_node f) {
  ListNode *temp = new ListNode(move(f));

  if (head) {
    temp->next = head;
    head->prev = temp;
    head = temp;
  } else {
    head = temp;
  }
  len++;
}

void List::append(ast_node t) {
  // TODO: Should we do it here?
  if (!t) {
    return;
  }

  if (tail) {
    ListNode *temp = new ListNode(move(t));
    temp->prev = tail;
    tail->next = temp;
    tail = temp;
    len++;
  } else {
    if (head) {
      ListNode *temp = new ListNode(move(t));
      head->next = temp;
      tail = temp;
      tail->prev = head;
      len++;
    } else {
      cons(move(t));
    }
  }
}

string List::string_repr() {
  if (head && head->data) {
    string s{"("};

    for (ListNode *current = head, *next; current;) {
      next = current->next;
      s = s + current->data->string_repr();
      current = next;
      if (next) {
        s = s + " ";
      }
    }
    return fmt::format("{})", s);

  } else {
    return "()";
  }
};

size_t List::length() { return len; }

void List::cleanup() {
  for (ListNode *current = head, *next; current;) {
    next = current->next;
    delete current;
    current = next;
  }
};

Value *List::codegen(Compiler &compiler, State &state) {
  if (length() == 0) {
    compiler.log_error("Can't eveluate empty list.");
    return nullptr;
  }

  auto first_expr{head->data.get()};

  if (first_expr->id() == symbol) {
    if (first_expr->name == "def") {
      auto def{make_unique<special_forms::Def>(first_expr, at(2))};
      return def->codegen(compiler, state);
    }
  } else {
    // if it's not symbol, it can be a list or keyword
    // or anything callable.
    EXPR_LOG("TODO: CHECK THE TYPE OF FIRST ELEMENT");
  }

  EXPR_LOG("Not implemented");
  return nullptr;
};

List::~List() {
  EXPR_LOG("Destroying list");
  cleanup();
};
} // namespace serene

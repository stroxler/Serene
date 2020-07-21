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
  void List::cons(ast_node f) {
    auto temp{std::make_unique<ListNode>(move(f))};
    if(head) {
      temp->next = move(head);
      head->prev = move(temp);
      head = move(temp);
    }
    else {
      head = move(temp);
    }
    len++;
  }

  List::List(const List &list) {
    ListNode *root = list.head.get();

    unique_ptr<ListNode> new_head{nullptr};
    ListNode *pnew_head{nullptr};

    while(root) {
      auto temp{std::make_unique<ListNode>(unique_ptr<AExpr>(root->data.get()))};
      if(new_head == nullptr) {
        new_head = move(temp);
        pnew_head = new_head.get();

      } else {
        pnew_head->next = move(temp);
        pnew_head = pnew_head->next.get();
      }

      root = root->next.get();

    }
    head = move(new_head);
  };


  List::List(List &&list) {
    head = move(list.head);
  }

  void List::add_tail(ast_node t) {
    auto temp{std::make_unique<ListNode>(move(t))};
    if(tail) {
      temp->prev = move(tail);
      tail->next = move(temp);
      tail = move(temp);
      len++;
    }
    else {
      if (head) {
        head->next = move(temp);
        len++;
      }
      else {
        cons(move(t));
      }
    }
  }

  string List::string_repr() {
    if (head && head->data) {
      return fmt::format("<List: '{}'>", head->data->string_repr());
    }
    else {
      return "<List: empty>";
    }
  };

  size_t List::length() {
    return len;
  }

  void List::cleanup() {
    while(head) {
      head = move(head->next);
    }
  };

  List::~List() {
    fmt::print("asdsadadsddddddddddddddddddddddddd\n");
    cleanup();
  };
}

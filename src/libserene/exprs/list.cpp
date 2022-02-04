/*
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "serene/exprs/list.h"

#include "serene/errors.h"
#include "serene/exprs/call.h"
#include "serene/exprs/def.h"
#include "serene/exprs/expression.h"
#include "serene/exprs/fn.h"
#include "serene/exprs/symbol.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>

#include <iterator>
#include <utility>

namespace serene {
namespace exprs {

List::List(const List &l) : Expression(l.location) {
  this->elements = l.elements;
};

List::List(const reader::LocationRange &loc, Node &e) : Expression(loc) {
  elements.push_back(e);
};

List::List(const reader::LocationRange &loc, Ast elems)
    : Expression(loc), elements(std::move(elems)){};

ExprType List::getType() const { return ExprType::List; };

std::string List::toString() const {
  std::string s{this->elements.empty() ? "-" : ""};

  for (const auto &n : this->elements) {
    s = llvm::formatv("{0} {1}", s, n->toString());
  }

  return llvm::formatv("<List {0}>", s);
};

MaybeNode List::analyze(semantics::AnalysisState &state) {

  if (!elements.empty()) {
    auto *first = elements[0].get();

    if (first->getType() == ExprType::Symbol) {
      auto *sym = llvm::dyn_cast<Symbol>(first);

      if (sym != nullptr) {
        if (sym->name == "def") {
          return Def::make(state, this);
        }

        if (sym->name == "fn") {
          return Fn::make(state, this);
        }
      }
    }

    return Call::make(state, this);
  }

  return EmptyNode;
};

bool List::classof(const Expression *e) {
  return e->getType() == ExprType::List;
};

std::vector<Node>::const_iterator List::cbegin() { return elements.begin(); }

std::vector<Node>::const_iterator List::cend() { return elements.end(); }

std::vector<Node>::iterator List::begin() { return elements.begin(); }

std::vector<Node>::iterator List::end() { return elements.end(); }

size_t List::count() const { return elements.size(); }

llvm::Optional<Expression *> List::at(uint index) {
  if (index >= elements.size()) {
    return llvm::None;
  }

  return llvm::Optional<Expression *>(this->elements[index].get());
}

Ast List::from(uint index) {
  if (index < elements.size()) {
    return Ast(elements.begin() + index, elements.end());
  }

  return Ast();
}

void List::append(Node n) { elements.push_back(std::move(n)); }
} // namespace exprs
} // namespace serene

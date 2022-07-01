/* -*- C++ -*-
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

#ifndef SERENE_TYPES_TYPE_H
#define SERENE_TYPES_TYPE_H

namespace serene::types {

// ============================================================================
// Expression
// ============================================================================
struct Expression {
  const unsigned char *data;
  explicit Expression(const unsigned char *data) : data(data){};
};

// ============================================================================
// Internal String
// ============================================================================

/// Internal string represts a smaller type of string with limited set of
/// functionalities that we use only for internal usage
struct InternalString {
  // We store the actual string in a "string" data section
  const char *data;
  unsigned int len;

  InternalString(const char *data, const unsigned int len)
      : data(data), len(len){};
};

// ============================================================================
// Symbol
// ============================================================================
struct Symbol {
  const InternalString *ns;
  const InternalString *name;

  Symbol(const InternalString *ns, const InternalString *name)
      : ns(ns), name(name){};
};

// ============================================================================
// Namespace
// ============================================================================
struct Namespace {
  const InternalString *name;

  explicit Namespace(const InternalString *name) : name(name){};
};

}; // namespace serene::types

#endif
